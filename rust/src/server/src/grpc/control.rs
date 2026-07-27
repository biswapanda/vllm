// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::PathBuf;
use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use tonic_health::server::HealthReporter;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

use super::{AdmissionGuard, AdmissionState, ControlServer, lora_rpc, pb};
use crate::lora_path::runtime_lora_allowed_path_prefixes;
use crate::state::AppState;

pub(crate) type ControlGrpcService = ControlServer<ControlServiceImpl>;

const GRPC_API_VERSION: &str = "vllm";
const GRPC_CAPABILITIES: &[&str] = &[
    "generate.sampling.v2",
    "generate.preprocessed_mm.v1",
    "generate.routed_experts.v1",
];

/// gRPC control service backed by the shared application state.
pub struct ControlServiceImpl {
    state: Arc<AppState>,
    admission: Arc<AdmissionState>,
    health_reporter: Option<HealthReporter>,
    lora_allowed_path_prefixes: Option<Arc<[PathBuf]>>,
    runtime_lora_updating_enabled: bool,
}

impl ControlServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self::with_admission(state, Arc::new(AdmissionState::default()), None)
    }

    pub(crate) fn with_admission(
        state: Arc<AppState>,
        admission: Arc<AdmissionState>,
        health_reporter: Option<HealthReporter>,
    ) -> Self {
        Self {
            state,
            admission,
            health_reporter,
            lora_allowed_path_prefixes: runtime_lora_allowed_path_prefixes().map(Arc::from),
            runtime_lora_updating_enabled: crate::routes::runtime_lora_updating_enabled(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_lora_allowed_path_prefixes(mut self, prefixes: Vec<PathBuf>) -> Self {
        self.lora_allowed_path_prefixes = Some(prefixes.into());
        self
    }

    #[cfg(test)]
    pub(crate) fn with_runtime_lora_updating(mut self, enabled: bool) -> Self {
        self.runtime_lora_updating_enabled = enabled;
        self
    }

    fn ready(&self) -> &EngineCoreReadyResponse {
        self.state.engine_core_client().ready_response()
    }

    fn parallelism_info(&self) -> pb::ParallelismInfo {
        let ready = self.ready();
        let (data_parallel_start_rank, managed_data_parallel_size) =
            self.state.engine_core_client().managed_data_parallel_span();
        pb::ParallelismInfo {
            tensor_parallel_size: ready.tensor_parallel_size,
            pipeline_parallel_size: ready.pipeline_parallel_size,
            data_parallel_size: ready.data_parallel_size.min(u64::from(u32::MAX)) as u32,
            data_parallel_rank: ready.data_parallel_rank,
            decode_context_parallel_size: ready.decode_context_parallel_size,
            data_parallel_start_rank,
            managed_data_parallel_size,
        }
    }

    async fn report_not_serving(&self) {
        if let Some(reporter) = &self.health_reporter {
            crate::set_generate_not_serving(reporter).await;
        }
    }

    fn try_admit(&self) -> Option<AdmissionGuard> {
        self.admission.try_admit()
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for ControlServiceImpl {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        let ready = self.ready();
        Ok(Response::new(pb::ServerInfo {
            engine_version: ready.vllm_version.clone(),
            api_version: GRPC_API_VERSION.to_string(),
            instance_id: ready.instance_id.clone(),
            parallelism: Some(self.parallelism_info()),
            max_model_len: self.state.engine_core_client().max_model_len(),
            kv_block_size: ready.block_size.min(u64::from(u32::MAX)) as u32,
            total_kv_blocks: self.state.engine_core_client().total_num_gpu_blocks(),
            max_running_requests: ready.max_num_seqs,
            max_batched_tokens: ready.max_num_batched_tokens,
            max_loras: ready.max_loras,
            capabilities: GRPC_CAPABILITIES
                .iter()
                .map(|capability| (*capability).to_string())
                .collect(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let ready = self.ready();
        let served = self.state.served_model_names();
        Ok(Response::new(pb::ModelInfo {
            model_id: self.state.chat.text().model_id().to_string(),
            served_model_name: self.state.primary_model_name().to_string(),
            served_model_aliases: served.iter().skip(1).cloned().collect(),
            tokenizer_modes: Vec::new(),
            // GenerateRequest accepts both prompt representations.
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

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request_ids = request.into_inner().request_ids;
        if request_ids.is_empty() {
            return Ok(Response::new(pb::AbortResponse {}));
        }
        self.state
            .chat
            .abort(&request_ids)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(pb::AbortResponse {}))
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<pb::DrainResponse>, Status> {
        self.admission.begin_drain();
        self.report_not_serving().await;
        let in_flight = self.admission.in_flight().min(u64::from(u32::MAX)) as u32;
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
        if !self.runtime_lora_updating_enabled {
            return Err(Status::failed_precondition(
                "runtime LoRA updating is disabled",
            ));
        }
        let _guard = self
            .try_admit()
            .ok_or_else(|| Status::unavailable("gRPC service is draining"))?;
        lora_rpc::load_lora(
            &self.state,
            self.lora_allowed_path_prefixes.as_deref(),
            request,
        )
        .await
    }

    async fn unload_lora(
        &self,
        request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        if !self.runtime_lora_updating_enabled {
            return Err(Status::failed_precondition(
                "runtime LoRA updating is disabled",
            ));
        }
        let _guard = self
            .try_admit()
            .ok_or_else(|| Status::unavailable("gRPC service is draining"))?;
        lora_rpc::unload_lora(&self.state, request).await
    }

    async fn list_loras(
        &self,
        request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        lora_rpc::list_loras(&self.state, request).await
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        let sources = self
            .state
            .engine_core_client()
            .ready_responses()
            .into_iter()
            .filter_map(kv_event_source)
            .collect();
        Ok(Response::new(pb::GetKvEventSourcesResponse { sources }))
    }
}

fn kv_event_source(response: &EngineCoreReadyResponse) -> Option<pb::KvEventSource> {
    if response.kv_events_publisher.as_deref() != Some("zmq") {
        return None;
    }
    let endpoint = offset_endpoint_port(
        response.kv_events_endpoint.as_ref()?,
        response.data_parallel_rank,
    );
    Some(pb::KvEventSource {
        transport: "zmq".to_string(),
        endpoint_addr: Some(kv_endpoint_from_zmq(&endpoint)?),
        topic: response.kv_events_topic.clone().unwrap_or_default(),
        replay_endpoint: String::new(),
        data_parallel_rank: Some(response.data_parallel_rank),
        encoding: "msgpack".to_string(),
        schema_version: 1,
        buffer_steps: 0,
        hwm: 0,
        max_queue_size: 0,
    })
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

fn kv_endpoint_from_zmq(endpoint: &str) -> Option<pb::KvEventEndpoint> {
    let rest = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let (host, port) = rest.rsplit_once(':')?;
    let port = port.parse().ok()?;
    let host = match host.trim_matches(|character| character == '[' || character == ']') {
        "*" | "0.0.0.0" | "::" | "" => advertise_host(),
        concrete => concrete.to_string(),
    };
    Some(pb::KvEventEndpoint {
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
