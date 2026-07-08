use tonic::transport::Server as TonicServer;
use tonic::{Code, Request};

use super::pb::engine_server::Engine as _;
use super::{EngineServiceImpl, pb};
use crate::listener::{Listener, MaybeTlsListener};

fn assert_unimplemented<T>(result: Result<tonic::Response<T>, tonic::Status>) {
    match result {
        Ok(_) => panic!("stub method unexpectedly succeeded"),
        Err(status) => assert_eq!(status.code(), Code::Unimplemented),
    }
}

#[tokio::test]
async fn every_engine_rpc_method_is_unimplemented() {
    let service = EngineServiceImpl::new();

    assert_unimplemented(service.generate(Request::new(Default::default())).await);
    assert_unimplemented(service.get_engine_info(Request::new(Default::default())).await);
    assert_unimplemented(service.get_model_info(Request::new(Default::default())).await);
    assert_unimplemented(service.health(Request::new(Default::default())).await);
    assert_unimplemented(service.abort(Request::new(Default::default())).await);
    assert_unimplemented(service.drain(Request::new(Default::default())).await);
    assert_unimplemented(service.load_lora(Request::new(Default::default())).await);
    assert_unimplemented(service.unload_lora(Request::new(Default::default())).await);
    assert_unimplemented(service.list_loras(Request::new(Default::default())).await);
    assert_unimplemented(service.get_kv_connector_info(Request::new(Default::default())).await);
    assert_unimplemented(service.get_kv_event_sources(Request::new(Default::default())).await);
}

#[tokio::test]
async fn stub_server_binds_and_shuts_down() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind engine RPC listener");
    let address = listener.local_addr().expect("listener address");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        TonicServer::builder()
            .add_service(super::EngineServer::new(EngineServiceImpl::new()))
            .serve_with_incoming_shutdown(
                MaybeTlsListener::plain(Listener::Tcp(listener)),
                async move {
                    let _ = shutdown_rx.await;
                },
            )
            .await
            .expect("serve engine RPC");
    });

    let mut client = pb::engine_client::EngineClient::connect(format!("http://{address}"))
        .await
        .expect("connect engine RPC client");
    let status = client
        .get_engine_info(pb::GetEngineInfoRequest {})
        .await
        .expect_err("stub RPC must fail");
    assert_eq!(status.code(), Code::Unimplemented);

    let _ = shutdown_tx.send(());
    server.await.expect("join engine RPC server");
}
