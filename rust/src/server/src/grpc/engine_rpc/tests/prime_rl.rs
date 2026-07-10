use super::*;
use crate::grpc::engine_rpc::PrimeRlEngineServer;
use crate::grpc::engine_rpc::openengine::pb as openengine_pb;
use crate::grpc::engine_rpc::openengine::pb::open_engine_server::OpenEngine;
use crate::grpc::engine_rpc::prime_rl::pb;
use crate::grpc::engine_rpc::prime_rl::pb::prime_rl_engine_client::PrimeRlEngineClient;

async fn send_utility_value(push: &mut PushSocket, call_id: u64, result: rmpv::Value) {
    send_outputs(
        push,
        UtilityCallOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: UtilityOutput {
                call_id: call_id.into(),
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(result)),
            },
        }
        .into(),
    )
    .await;
}

async fn prime_rl_test_server() -> (
    PrimeRlEngineClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
    Arc<EngineServiceImpl>,
) {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        vec![0x00, 0x00],
        move |dealer, push| {
            boxed_test_future(async move {
                let liveness = recv_engine_message(dealer).await;
                let liveness: rmpv::Value = rmp_serde::from_slice(&liveness[1]).unwrap();
                let liveness = liveness.as_array().unwrap();
                assert_eq!(liveness[2], rmpv::Value::from("collective_rpc"));
                let args = liveness[3].as_array().unwrap();
                assert_eq!(args[0], rmpv::Value::from("liveness_probe"));
                send_utility_value(
                    push,
                    liveness[1].as_u64().unwrap(),
                    rmpv::Value::Array(Vec::new()),
                )
                .await;

                let pause = recv_engine_message(dealer).await;
                let pause: rmpv::Value = rmp_serde::from_slice(&pause[1]).unwrap();
                let pause = pause.as_array().unwrap();
                assert_eq!(pause[2], rmpv::Value::from("pause_scheduler"));
                assert_eq!(
                    pause[3],
                    rmpv::Value::Array(vec![rmpv::Value::from("wait"), rmpv::Value::from(false)])
                );
                send_utility_value(push, pause[1].as_u64().unwrap(), rmpv::Value::Nil).await;

                let verify = recv_engine_message(dealer).await;
                let verify: rmpv::Value = rmp_serde::from_slice(&verify[1]).unwrap();
                let verify = verify.as_array().unwrap();
                assert_eq!(verify[2], rmpv::Value::from("is_scheduler_paused"));
                send_utility_value(push, verify[1].as_u64().unwrap(), rmpv::Value::from(true))
                    .await;

                let update = recv_engine_message(dealer).await;
                let update: rmpv::Value = rmp_serde::from_slice(&update[1]).unwrap();
                let update = update.as_array().unwrap();
                assert_eq!(update[2], rmpv::Value::from("collective_rpc"));
                let args = update[3].as_array().unwrap();
                assert_eq!(args[0], rmpv::Value::from("reload_weights"));
                send_utility_value(
                    push,
                    update[1].as_u64().unwrap(),
                    rmpv::Value::Array(Vec::new()),
                )
                .await;

                let reset = recv_engine_message(dealer).await;
                let reset: rmpv::Value = rmp_serde::from_slice(&reset[1]).unwrap();
                let reset = reset.as_array().unwrap();
                assert_eq!(reset[2], rmpv::Value::from("reset_prefix_cache"));
                assert_eq!(
                    reset[3],
                    rmpv::Value::Array(vec![rmpv::Value::from(false), rmpv::Value::from(true)])
                );
                send_utility_value(push, reset[1].as_u64().unwrap(), rmpv::Value::from(true)).await;

                let resume = recv_engine_message(dealer).await;
                let resume: rmpv::Value = rmp_serde::from_slice(&resume[1]).unwrap();
                let resume = resume.as_array().unwrap();
                assert_eq!(resume[2], rmpv::Value::from("resume_scheduler"));
                send_utility_value(push, resume[1].as_u64().unwrap(), rmpv::Value::Nil).await;
            })
        },
    ));

    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(
        test_llm(client),
        Arc::new(FakeTextBackend) as Arc<dyn ChatTextBackend>,
    );
    let state = Arc::new(AppState::new(vec!["test-model".to_string()], chat));
    let service = Arc::new(EngineServiceImpl::new(state));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind Prime RL listener");
    let addr = listener.local_addr().expect("Prime RL address");
    let server_service = service.clone();
    let server_task = tokio::spawn(async move {
        TonicServer::builder()
            .add_service(PrimeRlEngineServer::from_arc(server_service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .expect("Prime RL server");
    });
    let client = PrimeRlEngineClient::connect(format!("http://{addr}"))
        .await
        .expect("connect Prime RL client");
    (client, server_task, engine_task, service)
}

#[tokio::test]
async fn prime_rl_pause_update_version_and_resume_are_serialized() {
    let (mut client, server_task, engine_task, _service) = prime_rl_test_server().await;

    let liveness = client.liveness_probe(pb::LivenessProbeRequest {}).await.unwrap().into_inner();
    assert_eq!(liveness.status, "ok");

    client
        .pause_generation(pb::PauseGenerationRequest {
            mode: "wait".to_string(),
            clear_cache: false,
        })
        .await
        .unwrap();
    let update = client
        .update_weights_from_disk(pb::UpdateWeightsFromDiskRequest {
            model_path: "/models/checkpoint-7".to_string(),
            weight_version: "7".to_string(),
            engine_rpc: "reload_weights".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(update.status, "ok");
    assert_eq!(update.weight_version, "7");

    let version = client
        .get_weight_version(pb::GetWeightVersionRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(version.weight_version, "7");

    client.resume_generation(pb::ResumeGenerationRequest {}).await.unwrap();

    server_task.abort();
    engine_task.await.unwrap();
}

#[tokio::test]
async fn indeterminate_weight_state_is_not_ready_and_not_routable() {
    let (_client, server_task, engine_task, service) = prime_rl_test_server().await;
    service.mark_rl_indeterminate();

    let response = <EngineServiceImpl as OpenEngine>::health(
        service.as_ref(),
        tonic::Request::new(openengine_pb::HealthRequest::default()),
    )
    .await
    .unwrap()
    .into_inner();
    assert_eq!(
        openengine_pb::HealthState::try_from(response.state).unwrap(),
        openengine_pb::HealthState::NotReady
    );
    assert!(response.checks[0].message.contains("indeterminate"));
    assert!(service.is_draining());

    server_task.abort();
    drop(engine_task);
}
