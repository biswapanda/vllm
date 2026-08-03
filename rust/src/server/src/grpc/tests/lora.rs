use std::path::PathBuf;

use vllm_engine_core_client::mock_engine::default_ready_response;
use vllm_engine_core_client::protocol::decode_value;
use vllm_engine_core_client::protocol::output::{EngineCoreOutputs, UtilityCallOutput};
use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};

use super::*;

async fn grpc_lora_test_server<F>(
    engine_id: impl Into<EngineId>,
    allowed_path_prefixes: Vec<PathBuf>,
    runtime_updates_enabled: bool,
    run: F,
) -> (
    ControlClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
)
where
    F: for<'a> FnOnce(&'a mut DealerSocket, &'a mut PushSocket) -> TestFuture<'a> + Send + 'static,
{
    let mut ready = default_ready_response();
    ready.supports_lora = true;
    ready.max_loras = 1;
    let (state, engine_health, engine_task) =
        setup_state_with_ready_and_engine(engine_id, ready, run).await;
    let control_service = ControlServer::new(
        ControlServiceImpl::new(state.clone())
            .with_lora_allowed_path_prefixes(allowed_path_prefixes)
            .with_runtime_lora_updating(runtime_updates_enabled),
    );
    let inference_service = InferenceServer::new(InferenceServiceImpl::new(state));
    let (channel, server_task) = start_grpc_test_server(
        inference_service,
        control_service,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    (ControlClient::new(channel), server_task, engine_task)
}

async fn send_utility_result(push: &mut PushSocket, call_id: u64, result: bool) {
    let output: EngineCoreOutputs = UtilityCallOutput {
        engine_index: 0,
        timestamp: 0.0,
        output: UtilityOutput {
            call_id: call_id.into(),
            failure_message: None,
            result: Some(UtilityResultEnvelope::without_type_info(rmpv::Value::from(
                result,
            ))),
        },
    }
    .into();
    send_outputs(push, output).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_accepts_directory_under_injected_prefix() {
    let temp = tempfile::tempdir().expect("create temporary LoRA root");
    let allowed = temp.path().join("allowed");
    let adapter = allowed.join("adapter-a");
    tokio::fs::create_dir_all(&adapter).await.expect("create adapter directory");
    let canonical_adapter = tokio::fs::canonicalize(&adapter).await.expect("canonical adapter");
    let expected_path = canonical_adapter.to_string_lossy().into_owned();
    let engine_expected_path = expected_path.clone();

    let (mut client, server_task, engine_task) = grpc_lora_test_server(
        b"engine-grpc-lora-allowed",
        vec![allowed],
        true,
        move |dealer, push| {
            boxed_test_future(async move {
                let utility = recv_engine_message(dealer).await;
                let payload = decode_value(&utility[1]).expect("decode utility payload");
                let array = payload.as_array().expect("utility payload array");
                assert_eq!(array[2], rmpv::Value::from("add_lora"));
                let lora = array[3].as_array().expect("utility args")[0]
                    .as_array()
                    .expect("LoRA request tuple");
                assert_eq!(lora[0], rmpv::Value::from("adapter-a"));
                assert_eq!(lora[1], rmpv::Value::from(1));
                assert_eq!(lora[2], rmpv::Value::from(engine_expected_path));
                send_utility_result(push, array[1].as_u64().expect("call id"), true).await;
            })
        },
    )
    .await;

    let response = client
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 1,
                lora_name: "adapter-a".to_string(),
                source_path: adapter.to_string_lossy().into_owned(),
            }),
            load_inplace: false,
        })
        .await
        .expect("load adapter under configured prefix")
        .into_inner();
    assert_eq!(
        response.adapter.expect("loaded adapter").source_path,
        expected_path
    );

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_rejects_directory_outside_injected_prefix() {
    let temp = tempfile::tempdir().expect("create temporary LoRA root");
    let allowed = temp.path().join("allowed");
    let outside = temp.path().join("outside").join("adapter-a");
    tokio::fs::create_dir_all(&allowed).await.expect("create allowed directory");
    tokio::fs::create_dir_all(&outside).await.expect("create outside adapter");

    let (mut client, server_task, engine_task) = grpc_lora_test_server(
        b"engine-grpc-lora-outside",
        vec![allowed],
        true,
        |_dealer, _push| boxed_test_future(async {}),
    )
    .await;
    let error = client
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 1,
                lora_name: "adapter-a".to_string(),
                source_path: outside.to_string_lossy().into_owned(),
            }),
            load_inplace: false,
        })
        .await
        .expect_err("reject adapter outside configured prefix");
    assert_eq!(error.code(), tonic::Code::InvalidArgument);

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_does_not_expose_unavailable_configured_prefix() {
    let temp = tempfile::tempdir().expect("create temporary LoRA root");
    let adapter = temp.path().join("adapter-a");
    let missing_prefix = temp.path().join("secret-configured-prefix");
    tokio::fs::create_dir_all(&adapter).await.expect("create adapter directory");

    let (mut client, server_task, engine_task) = grpc_lora_test_server(
        b"engine-grpc-lora-missing-prefix",
        vec![missing_prefix.clone()],
        true,
        |_dealer, _push| boxed_test_future(async {}),
    )
    .await;
    let error = client
        .load_lora(pb::LoadLoraRequest {
            adapter: Some(pb::LoraAdapter {
                lora_id: 1,
                lora_name: "adapter-a".to_string(),
                source_path: adapter.to_string_lossy().into_owned(),
            }),
            load_inplace: false,
        })
        .await
        .expect_err("unavailable configured prefix is a server error");

    assert_eq!(error.code(), tonic::Code::Internal);
    assert!(!error.message().contains(&missing_prefix.to_string_lossy().into_owned()));
    assert_eq!(
        error.message(),
        "Runtime LoRA path policy is unavailable; check the server configuration."
    );

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_requires_runtime_lora_updating() {
    let (mut client, server_task, engine_task) = grpc_lora_test_server(
        b"engine-grpc-lora-disabled",
        Vec::new(),
        false,
        |_dealer, _push| boxed_test_future(async {}),
    )
    .await;

    let error = client
        .load_lora(pb::LoadLoraRequest::default())
        .await
        .expect_err("runtime LoRA updating is disabled");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);

    engine_task.await.expect("mock engine task");
    server_task.abort();
}
