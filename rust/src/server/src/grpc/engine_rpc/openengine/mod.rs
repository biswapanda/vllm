mod convert;

use std::time::Duration;

use futures::StreamExt as _;
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;

use super::pb as private;
use super::{EngineServiceImpl, PreparedGenerate, ResponseStream, build_kv_event_sources};

pub mod pb {
    tonic::include_proto!("openengine.v1");
}

pub use pb::open_engine_server::OpenEngineServer;

const SCHEMA_REVISION: u32 = 1;
const SCHEMA_RELEASE: &str = "openengine@7093bf087fcca367bd2b9b4fc233fb434d3c1c31";

#[tonic::async_trait]
impl pb::open_engine_server::OpenEngine for EngineServiceImpl {
    type GenerateStream = ResponseStream<pb::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (request, extensions) = convert::to_private_request(request.into_inner())?;
        let mut prepared = self.prepare_generate(request).await?;
        convert::apply_extensions(&mut prepared.text_request, extensions);
        let PreparedGenerate {
            guard,
            text_request,
            request_id,
            role,
            kv_connector,
            handoff_dp_rank,
            lora_lease,
        } = prepared;

        info!(%request_id, "openengine generate");
        let stream = self
            .state
            .chat
            .text()
            .generate(text_request)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        let stream = crate::lora::hold_lora_lease(stream, lora_lease);

        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            let _guard = guard;
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                let responses = match event {
                    Ok(event) => convert::event_to_responses(
                        event,
                        &request_id,
                        role,
                        kv_connector.as_deref(),
                        handoff_dp_rank,
                    ),
                    Err(error) => {
                        let response =
                            convert::error_response(&request_id, error.to_report_string());
                        let _ = tx.send(Ok(response)).await;
                        break;
                    }
                };
                for response in responses {
                    if tx.send(Ok(response)).await.is_err() {
                        return;
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn get_engine_info(
        &self,
        _request: Request<pb::GetEngineInfoRequest>,
    ) -> Result<Response<pb::EngineInfo>, Status> {
        let ready = self.ready();
        let parallelism = self.parallelism_info();
        Ok(Response::new(pb::EngineInfo {
            engine_name: "vllm".to_string(),
            engine_version: ready.vllm_version.clone(),
            role: canonical_role(private_role(self)) as i32,
            instance_id: ready.kv_engine_id.clone().unwrap_or_default(),
            supported_models: self.state.served_model_names().to_vec(),
            parallelism: Some(pb::ParallelismInfo {
                tensor_parallel_size: Some(parallelism.tensor_parallel_size),
                pipeline_parallel_size: Some(parallelism.pipeline_parallel_size),
                data_parallel_size: Some(parallelism.data_parallel_size),
                data_parallel_rank: Some(parallelism.data_parallel_rank),
                data_parallel_start_rank: Some(parallelism.data_parallel_start_rank),
            }),
            kv_connector: Some(canonical_kv_connector(self.kv_connector_info())),
            schema_revision: SCHEMA_REVISION,
            minimum_client_revision: SCHEMA_REVISION,
            schema_release: SCHEMA_RELEASE.to_string(),
        }))
    }

    async fn get_model_info(
        &self,
        request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let request = request.into_inner();
        if !request.model.is_empty()
            && !self.state.served_model_names().iter().any(|name| name == &request.model)
        {
            return Err(Status::not_found(format!(
                "model `{}` not found",
                request.model
            )));
        }
        let client = self.state.engine_core_client();
        let ready = self.ready();
        let served = self.state.served_model_names();
        let max_logprobs = self.state.chat.text().max_logprobs();
        Ok(Response::new(pb::ModelInfo {
            model_id: self.state.chat.text().model_id().to_string(),
            served_model_name: self.state.primary_model_name().to_string(),
            served_model_aliases: served.iter().skip(1).cloned().collect(),
            max_context_length: Some(client.max_model_len()),
            max_output_tokens: None,
            kv_block_size: Some(ready.kv_event_block_size.min(u64::from(u32::MAX)) as u32),
            total_kv_blocks: Some(self.per_rank_kv_blocks()),
            max_running_requests: Some(ready.max_num_seqs),
            max_batched_tokens: Some(ready.max_num_batched_tokens),
            tokenizer_modes: Vec::new(),
            supports_text_input: Some(true),
            supports_token_ids_input: Some(true),
            generation: Some(pb::GenerationCapabilities {
                prompt_logprobs: Some(prompt_logprob_capabilities(max_logprobs)),
                output_logprobs: Some(output_logprob_capabilities(max_logprobs)),
                guided_decoding: Some(pb::GuidedDecodingCapabilities {
                    supported: Some(true),
                    modes: vec![
                        pb::GuidedDecodingMode::JsonSchema as i32,
                        pb::GuidedDecodingMode::Regex as i32,
                        pb::GuidedDecodingMode::EbnfGrammar as i32,
                        pb::GuidedDecodingMode::StructuralTag as i32,
                        pb::GuidedDecodingMode::Choice as i32,
                        pb::GuidedDecodingMode::JsonObject as i32,
                    ],
                }),
                max_num_sequences: Some(1),
                supports_priority: Some(true),
                supports_stop_in_output: Some(true),
                supports_cache_salt: Some(true),
                supports_prefix_cache_bypass: Some(true),
            }),
            supports_lora: Some(ready.supports_lora),
            supports_multimodal: Some(self.state.chat.supports_multimodal()),
            reasoning_parser: self
                .state
                .chat
                .reasoning_parser_name()
                .unwrap_or_default()
                .to_string(),
            tool_call_parser: self
                .state
                .chat
                .tool_call_parser_name()
                .unwrap_or_default()
                .to_string(),
        }))
    }

    async fn get_load(
        &self,
        _request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        Ok(Response::new(pb::LoadInfo {
            instance_id: self.ready().kv_engine_id.clone().unwrap_or_default(),
            timestamp_unix_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .and_then(|duration| u64::try_from(duration.as_nanos()).ok()),
            running_requests: Some(self.in_flight().min(u64::from(u32::MAX)) as u32),
            queued_requests: None,
            active_kv_sessions: None,
            used_kv_blocks: None,
            total_kv_blocks: Some(self.per_rank_kv_blocks()),
            running_tokens: None,
            waiting_tokens: None,
            prefill_batch_size: None,
            decode_batch_size: None,
            ranks: Vec::new(),
            attributes: Default::default(),
        }))
    }

    async fn health(
        &self,
        request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        if self.is_rl_indeterminate() {
            return Ok(Response::new(pb::HealthResponse {
                state: pb::HealthState::NotReady as i32,
                checks: vec![pb::HealthCheck {
                    name: "prime_rl_weight_state".to_string(),
                    state: pb::HealthState::NotReady as i32,
                    message: "engine weight state is indeterminate and requires restart"
                        .to_string(),
                }],
            }));
        }
        let request = request.into_inner();
        let private_response = <EngineServiceImpl as private::engine_server::Engine>::health(
            self,
            Request::new(private::HealthRequest {
                include_inference_probe: request.include_inference_probe,
                model: request.model,
            }),
        )
        .await?
        .into_inner();
        let mut response = pb::HealthResponse {
            state: private_response.state,
            checks: private_response
                .checks
                .into_iter()
                .map(|check| pb::HealthCheck {
                    name: check.name,
                    state: check.state,
                    message: check.message,
                })
                .collect(),
        };
        if request.role != pb::EngineRole::Unspecified as i32
            && request.role != canonical_role(private_role(self)) as i32
        {
            response.state = pb::HealthState::NotReady as i32;
            response.checks.push(pb::HealthCheck {
                name: "role".to_string(),
                state: pb::HealthState::NotReady as i32,
                message: "engine role does not match requested role".to_string(),
            });
        }
        Ok(Response::new(response))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let target = request
            .into_inner()
            .target
            .ok_or_else(|| Status::invalid_argument("abort target is required"))?;
        let request_id = match target {
            pb::abort_request::Target::RequestId(id) => id,
            pb::abort_request::Target::KvSession(session) => session.session_id,
            pb::abort_request::Target::AllRequests(_) => {
                return Err(Status::unimplemented("abort all is not supported"));
            }
        };
        let response = <EngineServiceImpl as private::engine_server::Engine>::abort(
            self,
            Request::new(private::AbortRequest {
                request_id,
                abort_all: false,
            }),
        )
        .await?
        .into_inner();
        Ok(Response::new(pb::AbortResponse {
            status: match private::AbortStatus::try_from(response.status)
                .unwrap_or(private::AbortStatus::Unspecified)
            {
                private::AbortStatus::AlreadyFinished | private::AbortStatus::NotFound => {
                    pb::AbortStatus::AlreadyFinished as i32
                }
                _ => pb::AbortStatus::Aborted as i32,
            },
            message: response.message,
        }))
    }

    type DrainStream = ResponseStream<pb::DrainResponse>;

    async fn drain(
        &self,
        request: Request<pb::DrainRequest>,
    ) -> Result<Response<Self::DrainStream>, Status> {
        let request = request.into_inner();
        if request.stop_accepting_new_requests {
            self.begin_drain();
        }
        let admission = self.admission.clone();
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let started = pb::DrainResponse {
                event: Some(pb::drain_response::Event::State(
                    pb::DrainState::Started as i32,
                )),
                in_flight_requests: Some(
                    admission
                        .in_flight
                        .load(std::sync::atomic::Ordering::SeqCst)
                        .min(u64::from(u32::MAX)) as u32,
                ),
                open_kv_sessions: None,
                message: String::new(),
            };
            if tx.send(Ok(started)).await.is_err() {
                return;
            }
            let deadline = request.deadline_ms.map(|millis| {
                tokio::time::Instant::now() + Duration::from_millis(u64::from(millis))
            });
            loop {
                let in_flight = admission.in_flight.load(std::sync::atomic::Ordering::SeqCst);
                if in_flight == 0 {
                    let _ = tx
                        .send(Ok(pb::DrainResponse {
                            event: Some(pb::drain_response::Event::State(
                                pb::DrainState::Complete as i32,
                            )),
                            in_flight_requests: Some(0),
                            open_kv_sessions: None,
                            message: String::new(),
                        }))
                        .await;
                    return;
                }
                if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
                    let _ = tx
                        .send(Ok(pb::DrainResponse {
                            event: Some(pb::drain_response::Event::Error(pb::EngineError {
                                code: pb::ErrorCode::Overloaded as i32,
                                message: if request.abort_after_deadline {
                                    "drain deadline elapsed; abort-all is not supported"
                                } else {
                                    "drain deadline elapsed"
                                }
                                .to_string(),
                                retryable: true,
                                retry_after_ms: None,
                                details: None,
                            })),
                            in_flight_requests: Some(in_flight.min(u64::from(u32::MAX)) as u32),
                            open_kv_sessions: None,
                            message: String::new(),
                        }))
                        .await;
                    return;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        let request = request.into_inner();
        let response = <EngineServiceImpl as private::engine_server::Engine>::load_lora(
            self,
            Request::new(private::LoadLoraRequest {
                adapter: request.adapter.map(private_lora),
            }),
        )
        .await?
        .into_inner();
        Ok(Response::new(pb::LoadLoraResponse {
            adapter: response.adapter.map(canonical_lora),
            already_loaded: response.already_loaded,
        }))
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        let response = <EngineServiceImpl as private::engine_server::Engine>::unload_lora(
            self,
            Request::new(private::UnloadLoraRequest {
                lora_name: request.into_inner().lora_name,
            }),
        )
        .await?
        .into_inner();
        Ok(Response::new(pb::UnloadLoraResponse {
            adapter: response.adapter.map(canonical_lora),
        }))
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        let response = <EngineServiceImpl as private::engine_server::Engine>::list_loras(
            self,
            Request::new(private::ListLorasRequest {}),
        )
        .await?
        .into_inner();
        Ok(Response::new(pb::ListLorasResponse {
            adapters: response.adapters.into_iter().map(canonical_lora).collect(),
        }))
    }

    async fn get_kv_connector_info(
        &self,
        _request: Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<Response<pb::KvConnectorInfo>, Status> {
        Ok(Response::new(canonical_kv_connector(
            self.kv_connector_info(),
        )))
    }

    async fn get_kv_event_sources(
        &self,
        request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let requested = request.into_inner().data_parallel_ranks;
        let responses = self.state.engine_core_client().ready_responses();
        let ranked = responses
            .into_iter()
            .map(|response| (response.data_parallel_rank, response))
            .collect::<Vec<_>>();
        let sources = build_kv_event_sources(&ranked)
            .into_iter()
            .filter(|source| requested.is_empty() || requested.contains(&source.data_parallel_rank))
            .map(|source| pb::KvEventSource {
                transport: source.transport,
                endpoint_addr: source.endpoint_addr.map(|endpoint| pb::KvEndpoint {
                    host: endpoint.host,
                    port: endpoint.port,
                    protocol: endpoint.protocol,
                }),
                topic: source.topic,
                replay_endpoint: source.replay_endpoint,
                data_parallel_rank: Some(source.data_parallel_rank),
                encoding: source.encoding,
                schema_version: Some(source.schema_version),
                buffer_steps: Some(source.buffer_steps),
                hwm: Some(source.hwm),
                max_queue_size: Some(source.max_queue_size),
            })
            .collect();
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }

    type SubscribeKvEventsStream = ResponseStream<pb::SubscribeKvEventsResponse>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented(
            "vLLM publishes KV events through the advertised ZMQ sources",
        ))
    }

    type SubscribeRuntimeEventsStream = ResponseStream<pb::SubscribeRuntimeEventsResponse>;

    async fn subscribe_runtime_events(
        &self,
        _request: Request<pb::SubscribeRuntimeEventsRequest>,
    ) -> Result<Response<Self::SubscribeRuntimeEventsStream>, Status> {
        Err(Status::unimplemented(
            "runtime event streaming is not implemented",
        ))
    }
}

fn private_role(service: &EngineServiceImpl) -> private::EngineRole {
    super::convert::role_from_kv_role(service.ready().kv_role.as_deref())
}

fn canonical_role(role: private::EngineRole) -> pb::EngineRole {
    match role {
        private::EngineRole::Prefill => pb::EngineRole::Prefill,
        private::EngineRole::Decode => pb::EngineRole::Decode,
        private::EngineRole::Aggregated => pb::EngineRole::Aggregated,
        private::EngineRole::Unspecified => pb::EngineRole::Unspecified,
    }
}

fn output_logprob_capabilities(max_logprobs: i32) -> pb::LogprobCapabilities {
    let mut modes = vec![
        pb::CandidateTokenSelectionMode::TopN as i32,
        pb::CandidateTokenSelectionMode::TokenIds as i32,
    ];
    if max_logprobs == -1 {
        modes.push(pb::CandidateTokenSelectionMode::All as i32);
    }
    pb::LogprobCapabilities {
        supported: Some(true),
        candidate_selection_modes: modes,
        max_top_n: u32::try_from(max_logprobs).ok(),
    }
}

fn prompt_logprob_capabilities(max_logprobs: i32) -> pb::LogprobCapabilities {
    let mut modes = vec![pb::CandidateTokenSelectionMode::TopN as i32];
    if max_logprobs == -1 {
        modes.push(pb::CandidateTokenSelectionMode::All as i32);
    }
    pb::LogprobCapabilities {
        supported: Some(true),
        candidate_selection_modes: modes,
        max_top_n: u32::try_from(max_logprobs).ok(),
    }
}

fn canonical_kv_connector(connector: private::KvConnectorInfo) -> pb::KvConnectorInfo {
    pb::KvConnectorInfo {
        enabled: Some(connector.enabled),
        transfer_backend: connector.transfer_backend,
        local_endpoints: connector
            .local_endpoints
            .into_iter()
            .map(|endpoint| pb::KvEndpoint {
                host: endpoint.host,
                port: endpoint.port,
                protocol: endpoint.protocol,
            })
            .collect(),
        supported_protocols: connector.supported_protocols,
        supports_remote_prefill: None,
        supports_decode_pull: None,
        supports_abort_cleanup: Some(true),
        supports_drain: Some(true),
        schema_version: Some(connector.schema_version),
    }
}

fn private_lora(adapter: pb::LoraAdapter) -> private::LoraAdapter {
    private::LoraAdapter {
        lora_id: adapter.lora_id,
        lora_name: adapter.lora_name,
        source_path: adapter.source_path,
    }
}

fn canonical_lora(adapter: private::LoraAdapter) -> pb::LoraAdapter {
    pb::LoraAdapter {
        lora_id: adapter.lora_id,
        lora_name: adapter.lora_name,
        source_path: adapter.source_path,
    }
}
