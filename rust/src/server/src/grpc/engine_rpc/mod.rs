use std::pin::Pin;

use futures::Stream;
use tonic::{Request, Response, Status};

pub mod pb {
    tonic::include_proto!("vllm.engine.v1");
}

pub use pb::engine_server::EngineServer;

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[derive(Default)]
pub struct EngineServiceImpl;

impl EngineServiceImpl {
    pub fn new() -> Self {
        Self
    }
}

fn unimplemented<T>(method: &str) -> Result<Response<T>, Status> {
    Err(Status::unimplemented(format!(
        "engine RPC method `{method}` is not implemented"
    )))
}

#[tonic::async_trait]
impl pb::engine_server::Engine for EngineServiceImpl {
    type GenerateStream = ResponseStream<pb::GenerateResponse>;

    async fn generate(
        &self,
        _request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        unimplemented("Generate")
    }

    async fn get_engine_info(
        &self,
        _request: Request<pb::GetEngineInfoRequest>,
    ) -> Result<Response<pb::EngineInfo>, Status> {
        unimplemented("GetEngineInfo")
    }

    async fn get_model_info(
        &self,
        _request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        unimplemented("GetModelInfo")
    }

    async fn health(
        &self,
        _request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        unimplemented("Health")
    }

    async fn abort(
        &self,
        _request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        unimplemented("Abort")
    }

    async fn drain(
        &self,
        _request: Request<pb::DrainRequest>,
    ) -> Result<Response<pb::DrainResponse>, Status> {
        unimplemented("Drain")
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
