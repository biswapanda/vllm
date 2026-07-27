// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;
use vllm_text::{DecodedTextEvent, Prompt, TextOutputStreamExt as _, TextRequest};

use super::convert::{self, ResponseOpts};
use super::{AdmissionState, InferenceServer, pb};
use crate::state::AppState;

pub(crate) type InferenceGrpcService = InferenceServer<InferenceServiceImpl>;

/// gRPC inference service backed by the shared application state.
pub struct InferenceServiceImpl {
    state: Arc<AppState>,
    admission: Arc<AdmissionState>,
}

impl InferenceServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self::with_admission(state, Arc::new(AdmissionState::default()))
    }

    pub(crate) fn with_admission(state: Arc<AppState>, admission: Arc<AdmissionState>) -> Self {
        Self { state, admission }
    }

    async fn prepare_request(
        &self,
        proto_request: pb::GenerateRequest,
        stream: bool,
    ) -> Result<(TextRequest, crate::lora::LoraLease), Status> {
        let ready = self.state.engine_core_client().ready_response();
        let role = convert::role_from_kv_role(ready.kv_role.as_deref());
        convert::validate_disaggregated_request(&proto_request, role)?;
        let supports_lora = ready.supports_lora;
        let media = convert::media_parts_from_request(&proto_request.media)?;
        let lora_name = proto_request.lora_name.clone();
        let mut text_request =
            convert::to_text_request(proto_request, stream, self.state.served_model_names())?;

        let mut lora_lease = None;
        if !lora_name.is_empty() {
            if !supports_lora {
                return Err(Status::failed_precondition(
                    "engine was not started with LoRA enabled",
                ));
            }
            let mut resolution = self.state.resolve_model_with_loras(Some(&lora_name)).await;
            lora_lease = resolution.lease.take();
            if !self.state.lora_state_is_consistent() {
                return Err(Status::failed_precondition(
                    "LoRA state differs across engine ranks; restart the engine",
                ));
            }
            text_request.lora_request = Some(resolution.lora_request.ok_or_else(|| {
                Status::not_found(format!("LoRA adapter `{lora_name}` is not loaded"))
            })?);
        }
        // Language-only LoRA does not change preprocessed multimodal features.
        // Tower or connector adapters need an explicit cache-identity contract.
        if !media.is_empty() {
            let Prompt::TokenIds(mut token_ids) = text_request.prompt else {
                return Err(Status::invalid_argument(
                    "multimodal gRPC requests must provide token_ids input",
                ));
            };
            let mm_features = self
                .state
                .chat
                .prepare_media(media, &mut token_ids)
                .await
                .map_err(|error| Status::internal(error.to_report_string()))?;
            text_request.prompt = Prompt::TokenIds(token_ids);
            text_request.mm_features = mm_features;
        }

        if role == convert::KvRole::Prefill {
            convert::mark_prefill_request(&mut text_request);
        }

        Ok((text_request, lora_lease))
    }
}

#[tonic::async_trait]
impl pb::inference_server::Inference for InferenceServiceImpl {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    /// Unary generate: collect all output and return a single response.
    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let _guard = self
            .admission
            .try_admit()
            .ok_or_else(|| Status::unavailable("gRPC service is draining"))?;
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let (text_request, lora_lease) = self.prepare_request(proto_req, false).await?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (unary)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;
        let stream = crate::lora::hold_lora_lease(stream, lora_lease);

        let collected = stream.collect_output().await.map_err(text_error_to_status)?;

        // Build the single aggregated response.
        let prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );

        let finish_info = vllm_text::Finished {
            usage: collected.usage,
            finish_reason: collected.finish_reason,
            kv_transfer_params: collected.kv_transfer_params,
            ec_transfer_params: collected.ec_transfer_params,
            routed_experts: collected.routed_experts,
        };

        let outputs = convert::to_sequence_output(
            &collected.text,
            &collected.token_ids,
            collected.logprobs.as_ref(),
            Some(&finish_info),
            &response_opts,
        );

        Ok(Response::new(pb::GenerateResponse {
            prompt_info: Some(prompt_info),
            outputs: Some(outputs),
        }))
    }

    /// Streaming generate: yield incremental responses as tokens are produced.
    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let guard = self
            .admission
            .try_admit()
            .ok_or_else(|| Status::unavailable("gRPC service is draining"))?;
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let (text_request, lora_lease) = self.prepare_request(proto_req, true).await?;

        let request_id = text_request.request_id.clone();
        info!(%request_id, "grpc generate (stream)");

        let stream = self.state.chat.text().generate(text_request).await;
        let stream = stream.map_err(text_error_to_status)?;
        let stream = crate::lora::hold_lora_lease(stream, lora_lease);

        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            let _guard = guard;
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let response = match event {
                    Err(e) => Err(text_error_to_status(e)),
                    Ok(DecodedTextEvent::Start {
                        prompt_token_ids,
                        prompt_logprobs,
                    }) => {
                        let prompt_info = convert::to_prompt_info(
                            &prompt_token_ids,
                            prompt_logprobs.as_ref(),
                            &response_opts,
                        );
                        Ok(pb::GenerateResponse {
                            prompt_info: Some(prompt_info),
                            outputs: None,
                        })
                    }
                    Ok(DecodedTextEvent::TextDelta {
                        delta,
                        token_ids,
                        logprobs,
                        finished,
                    }) => Ok(pb::GenerateResponse {
                        prompt_info: None,
                        outputs: Some(convert::to_sequence_output(
                            &delta,
                            &token_ids,
                            logprobs.as_ref(),
                            finished.as_ref(),
                            &response_opts,
                        )),
                    }),
                };

                if tx.send(response).await.is_err() {
                    // Client disconnected.
                    break;
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }
}

fn text_error_to_status(error: vllm_text::Error) -> Status {
    let message = error.to_report_string();
    if error.is_request_validation_error() {
        Status::invalid_argument(message)
    } else {
        Status::internal(message)
    }
}
