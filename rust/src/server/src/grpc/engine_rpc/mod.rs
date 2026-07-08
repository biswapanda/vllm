mod convert;
mod lora;

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;
use uuid::Uuid;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextOutputStreamExt as _, TextRequest};

use crate::state::AppState;

pub mod pb {
    tonic::include_proto!("vllm.engine.v1");
}

pub use pb::engine_server::EngineServer;

const ENGINE_RPC_API_VERSION: &str = "vllm.engine.v1";
const INFERENCE_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[derive(Default)]
struct AdmissionState {
    draining: AtomicBool,
    in_flight: AtomicU64,
}

pub struct EngineServiceImpl {
    state: Arc<AppState>,
    admission: Arc<AdmissionState>,
}

impl EngineServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            admission: Arc::new(AdmissionState::default()),
        }
    }

    fn ready(&self) -> &EngineCoreReadyResponse {
        self.state.engine_core_client().ready_response()
    }

    fn per_rank_kv_blocks(&self) -> u64 {
        per_rank_kv_blocks(&self.state.engine_core_client().ready_responses())
    }

    fn parallelism_info(&self) -> pb::ParallelismInfo {
        let ready = self.ready();
        pb::ParallelismInfo {
            tensor_parallel_size: ready.tensor_parallel_size,
            pipeline_parallel_size: ready.pipeline_parallel_size,
            data_parallel_size: ready.data_parallel_size.min(u64::from(u32::MAX)) as u32,
            data_parallel_rank: ready.data_parallel_rank,
            data_parallel_start_rank: ready.data_parallel_rank,
        }
    }

    fn kv_connector_info(&self) -> pb::KvConnectorInfo {
        let ready = self.ready();
        pb::KvConnectorInfo {
            enabled: ready.kv_connector.is_some(),
            transfer_backend: ready.kv_connector.clone().unwrap_or_default(),
            local_endpoints: Vec::new(),
            supported_protocols: Vec::new(),
            schema_version: 1,
        }
    }

    fn begin_drain(&self) {
        self.admission.draining.store(true, Ordering::SeqCst);
    }

    fn is_draining(&self) -> bool {
        self.admission.draining.load(Ordering::SeqCst)
    }

    fn in_flight(&self) -> u64 {
        self.admission.in_flight.load(Ordering::SeqCst)
    }

    fn try_admit(&self) -> Option<AdmissionGuard> {
        if self.is_draining() {
            return None;
        }
        self.admission.in_flight.fetch_add(1, Ordering::SeqCst);
        if self.is_draining() {
            self.admission.in_flight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(AdmissionGuard(self.admission.clone()))
    }
}

fn per_rank_kv_blocks(ready: &[&EngineCoreReadyResponse]) -> u64 {
    ready.iter().map(|response| response.num_gpu_blocks).min().unwrap_or(0)
}

struct AdmissionGuard(Arc<AdmissionState>);

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

fn health_check(name: &str, state: pb::HealthState, message: Option<String>) -> pb::HealthCheck {
    pb::HealthCheck {
        name: name.to_string(),
        state: state as i32,
        message: message.unwrap_or_default(),
    }
}

#[tonic::async_trait]
impl pb::engine_server::Engine for EngineServiceImpl {
    type GenerateStream = ResponseStream<pb::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let guard =
            self.try_admit().ok_or_else(|| Status::unavailable("engine RPC is draining"))?;

        let proto_request = request.into_inner();
        let lora_name = proto_request.lora_name.clone();
        let role = convert::role_from_kv_role(self.ready().kv_role.as_deref());
        let handoff_dp_rank = convert::validate_disaggregated_request(
            &proto_request,
            role,
            self.ready().kv_connector.as_deref(),
            self.ready().data_parallel_size,
            self.ready().data_parallel_rank,
        )?;
        let media_parts = convert::media_parts_from_request(&proto_request.media)?;
        let mut text_request =
            convert::to_text_request(proto_request, self.state.served_model_names())?;

        let mut lora_lease = None;
        if !lora_name.is_empty() {
            if !self.ready().supports_lora {
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

        if !media_parts.is_empty() {
            let Prompt::TokenIds(mut token_ids) = text_request.prompt else {
                return Err(Status::invalid_argument(
                    "multimodal engine RPC requests must provide token_ids input; placeholder markers are expanded engine-side",
                ));
            };
            let mm_features = self
                .state
                .chat
                .prepare_media(media_parts, &mut token_ids)
                .await
                .map_err(|error| Status::internal(error.to_report_string()))?;
            text_request.prompt = Prompt::TokenIds(token_ids);
            text_request.mm_features = mm_features;
        }

        let request_id = text_request.request_id.clone();
        info!(%request_id, "engine_rpc generate");
        let kv_connector = self.ready().kv_connector.clone();
        if role == pb::EngineRole::Prefill {
            convert::mark_prefill_request(&mut text_request);
        }
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
        Ok(Response::new(pb::EngineInfo {
            engine_name: "vllm".to_string(),
            engine_version: ready.vllm_version.clone(),
            api_version: ENGINE_RPC_API_VERSION.to_string(),
            role: convert::role_from_kv_role(ready.kv_role.as_deref()) as i32,
            instance_id: ready.kv_engine_id.clone().unwrap_or_default(),
            supported_models: self.state.served_model_names().to_vec(),
            parallelism: Some(self.parallelism_info()),
            kv_connector: Some(self.kv_connector_info()),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let client = self.state.engine_core_client();
        let ready = self.ready();
        let served = self.state.served_model_names();
        Ok(Response::new(pb::ModelInfo {
            model_id: self.state.chat.text().model_id().to_string(),
            served_model_name: self.state.primary_model_name().to_string(),
            served_model_aliases: served.iter().skip(1).cloned().collect(),
            max_context_length: client.max_model_len(),
            max_output_tokens: 0,
            kv_block_size: ready.kv_event_block_size.min(u64::from(u32::MAX)) as u32,
            total_kv_blocks: self.per_rank_kv_blocks(),
            max_running_requests: ready.max_num_seqs,
            max_batched_tokens: ready.max_num_batched_tokens,
            tokenizer_modes: Vec::new(),
            max_loras: ready.max_loras,
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_lora: ready.supports_lora,
            supports_multimodal: self.state.chat.supports_multimodal(),
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

    async fn health(
        &self,
        request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        let request = request.into_inner();
        let client = self.state.engine_core_client();
        let engine_state = if !client.is_healthy() {
            pb::HealthState::NotReady
        } else if self.is_draining() {
            pb::HealthState::Draining
        } else {
            pb::HealthState::Ready
        };
        let mut checks = vec![health_check(
            "engine",
            engine_state,
            client.health_error().map(|error| error.to_report_string()),
        )];
        let mut overall = engine_state;
        if self.ready().supports_lora && !self.state.lora_state_is_consistent() {
            checks.push(health_check(
                "lora",
                pb::HealthState::NotReady,
                Some("adapter state differs across engine ranks; restart required".to_string()),
            ));
            overall = pb::HealthState::NotReady;
        }
        if request.include_inference_probe && overall == pb::HealthState::Ready {
            if let Some(_guard) = self.try_admit() {
                let (probe_state, message) = self.run_inference_probe(&request.model).await;
                if probe_state != pb::HealthState::Ready {
                    overall = pb::HealthState::Degraded;
                }
                checks.push(health_check("inference_probe", probe_state, message));
            } else {
                overall = pb::HealthState::Draining;
            }
        }
        Ok(Response::new(pb::HealthResponse {
            state: overall as i32,
            checks,
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request = request.into_inner();
        if request.abort_all {
            return Ok(Response::new(pb::AbortResponse {
                status: pb::AbortStatus::Unsupported as i32,
                message: "abort_all is not supported".to_string(),
            }));
        }
        if request.request_id.is_empty() {
            return Err(Status::invalid_argument("request_id is required"));
        }
        self.state
            .engine_core_client()
            .abort(std::slice::from_ref(&request.request_id))
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(pb::AbortResponse {
            status: pb::AbortStatus::Aborted as i32,
            message: String::new(),
        }))
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<pb::DrainResponse>, Status> {
        self.begin_drain();
        let in_flight = self.in_flight().min(u64::from(u32::MAX)) as u32;
        let state = if in_flight == 0 {
            pb::DrainState::Complete
        } else {
            pb::DrainState::InProgress
        };
        Ok(Response::new(pb::DrainResponse {
            state: state as i32,
            in_flight_requests: in_flight,
            message: String::new(),
        }))
    }

    async fn load_lora(
        &self,
        request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        let _guard =
            self.try_admit().ok_or_else(|| Status::unavailable("engine RPC is draining"))?;
        lora::load_lora(&self.state, request).await
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        let _guard =
            self.try_admit().ok_or_else(|| Status::unavailable("engine RPC is draining"))?;
        lora::unload_lora(&self.state, request).await
    }

    async fn list_loras(
        &self,
        request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        lora::list_loras(&self.state, request).await
    }

    async fn get_kv_connector_info(
        &self,
        _request: Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<Response<pb::KvConnectorInfo>, Status> {
        Ok(Response::new(self.kv_connector_info()))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let responses = self.state.engine_core_client().ready_responses();
        let ranked = responses
            .into_iter()
            .map(|response| (response.data_parallel_rank, response))
            .collect::<Vec<_>>();
        Ok(Response::new(pb::GetKvEventSourcesResponse {
            sources: build_kv_event_sources(&ranked),
        }))
    }
}

impl EngineServiceImpl {
    async fn run_inference_probe(&self, model: &str) -> (pb::HealthState, Option<String>) {
        if !model.is_empty() && !self.state.served_model_names().iter().any(|name| name == model) {
            return (
                pb::HealthState::Degraded,
                Some(format!("model `{model}` not found")),
            );
        }

        let request = TextRequest {
            request_id: format!("engine_rpc-health-{}", Uuid::new_v4()),
            prompt: Prompt::Text("hi".to_string()),
            mm_features: None,
            sampling_params: SamplingParams {
                temperature: Some(0.0),
                max_tokens: Some(1),
                ..SamplingParams::default()
            },
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            priority: 0,
            cache_salt: None,
            add_special_tokens: true,
            data_parallel_rank: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
            arrival_time: None,
        };

        let probe = async {
            let stream = self.state.chat.text().generate(request).await?;
            stream.collect_output().await
        };
        match tokio::time::timeout(INFERENCE_PROBE_TIMEOUT, probe).await {
            Ok(Ok(_)) => (pb::HealthState::Ready, None),
            Ok(Err(error)) => (pb::HealthState::Degraded, Some(error.to_report_string())),
            Err(_) => (
                pb::HealthState::Degraded,
                Some(format!(
                    "inference probe timed out after {}s",
                    INFERENCE_PROBE_TIMEOUT.as_secs()
                )),
            ),
        }
    }
}

fn offset_endpoint_port(endpoint: &str, data_parallel_rank: u32) -> String {
    if data_parallel_rank == 0 || endpoint.is_empty() {
        return endpoint.to_string();
    }
    if endpoint.contains("inproc") {
        return format!("{endpoint}_dp{data_parallel_rank}");
    }
    if endpoint.contains("tcp")
        && let Some((base_addr, port)) = endpoint.rsplit_once(':')
        && let Ok(base_port) = port.parse::<u32>()
    {
        return format!("{base_addr}:{}", base_port + data_parallel_rank);
    }
    endpoint.to_string()
}

fn build_kv_event_sources(
    ready_responses: &[(u32, &EngineCoreReadyResponse)],
) -> Vec<pb::KvEventSource> {
    ready_responses
        .iter()
        .filter(|(_, response)| response.kv_events_publisher.as_deref() == Some("zmq"))
        .filter_map(|(rank, response)| {
            let base = response.kv_events_endpoint.as_ref()?;
            let endpoint = offset_endpoint_port(base, *rank);
            let endpoint_addr = kv_endpoint_from_zmq(&endpoint)?;
            Some(pb::KvEventSource {
                transport: "zmq".to_string(),
                endpoint_addr: Some(endpoint_addr),
                topic: response.kv_events_topic.clone().unwrap_or_default(),
                replay_endpoint: String::new(),
                data_parallel_rank: *rank,
                encoding: "msgpack".to_string(),
                schema_version: 1,
                buffer_steps: 0,
                hwm: 0,
                max_queue_size: 0,
            })
        })
        .collect()
}

fn kv_endpoint_from_zmq(endpoint: &str) -> Option<pb::KvEndpoint> {
    let rest = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let (host, port) = rest.rsplit_once(':')?;
    let port: u32 = port.parse().ok()?;
    let host = match host.trim_matches(|character| character == '[' || character == ']') {
        "*" | "0.0.0.0" | "::" | "" => advertise_host(),
        concrete => concrete.to_string(),
    };
    Some(pb::KvEndpoint {
        host,
        port,
        protocol: "tcp".to_string(),
    })
}

fn advertise_host() -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|socket| {
            socket.connect("10.255.255.255:1")?;
            Ok(socket.local_addr()?.ip().to_string())
        })
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}

#[cfg(test)]
mod tests;
