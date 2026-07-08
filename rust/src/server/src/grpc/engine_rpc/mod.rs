use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use futures::Stream;
use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

use crate::state::AppState;

pub mod pb {
    tonic::include_proto!("vllm.engine.v1");
}

pub use pb::engine_server::EngineServer;

const ENGINE_RPC_API_VERSION: &str = "vllm.engine.v1";

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

fn unimplemented<T>(method: &str) -> Result<Response<T>, Status> {
    Err(Status::unimplemented(format!(
        "engine RPC method `{method}` is not implemented"
    )))
}

fn role_from_kv_role(kv_role: Option<&str>) -> pb::EngineRole {
    match kv_role {
        Some("kv_producer") => pb::EngineRole::Prefill,
        Some("kv_consumer") => pb::EngineRole::Decode,
        _ => pb::EngineRole::Aggregated,
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
        _request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let _guard =
            self.try_admit().ok_or_else(|| Status::unavailable("engine RPC is draining"))?;
        unimplemented("Generate")
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
            role: role_from_kv_role(ready.kv_role.as_deref()) as i32,
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
            kv_block_size: ready.block_size.min(u64::from(u32::MAX)) as u32,
            total_kv_blocks: self.per_rank_kv_blocks(),
            max_running_requests: ready.max_num_seqs,
            max_batched_tokens: ready.max_num_batched_tokens,
            tokenizer_modes: Vec::new(),
            max_loras: 0,
            supports_text_input: true,
            supports_token_ids_input: true,
            supports_lora: false,
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
        if request.into_inner().include_inference_probe {
            return unimplemented("Health inference probe");
        }
        let client = self.state.engine_core_client();
        let state = if !client.is_healthy() {
            pb::HealthState::NotReady
        } else if self.is_draining() {
            pb::HealthState::Draining
        } else {
            pb::HealthState::Ready
        };
        Ok(Response::new(pb::HealthResponse {
            state: state as i32,
            checks: vec![health_check(
                "engine",
                state,
                client.health_error().map(|error| error.to_report_string()),
            )],
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
        _request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        unimplemented("LoadLora")
    }

    async fn unload_lora(
        &self,
        _request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        unimplemented("UnloadLora")
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        unimplemented("ListLoras")
    }

    async fn get_kv_connector_info(
        &self,
        _request: Request<pb::GetKvConnectorInfoRequest>,
    ) -> Result<Response<pb::KvConnectorInfo>, Status> {
        unimplemented("GetKvConnectorInfo")
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        unimplemented("GetKvEventSources")
    }
}

#[cfg(test)]
mod tests;
