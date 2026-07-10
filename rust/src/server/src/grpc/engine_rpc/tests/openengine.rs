use super::*;
use crate::grpc::engine_rpc::openengine::OpenEngineServer;
use crate::grpc::engine_rpc::openengine::pb;
use crate::grpc::engine_rpc::openengine::pb::open_engine_client::OpenEngineClient;

async fn openengine_test_server() -> (
    OpenEngineClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
) {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        vec![0x00, 0x00],
        move |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                assert_eq!(request.sampling_params.as_ref().unwrap().logprobs, Some(2));
                assert_eq!(request.cache_salt.as_deref(), Some("tenant-a"));
                assert_eq!(request.priority, 7);
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
                )
                .await;
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
        .expect("bind OpenEngine listener");
    let addr = listener.local_addr().expect("OpenEngine address");
    let server_task = tokio::spawn(async move {
        TonicServer::builder()
            .add_service(OpenEngineServer::from_arc(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .expect("OpenEngine server");
    });
    let client = OpenEngineClient::connect(format!("http://{addr}"))
        .await
        .expect("connect OpenEngine client");
    (client, server_task, engine_task)
}

#[tokio::test]
async fn canonical_openengine_discovers_and_generates() {
    let (mut client, server_task, engine_task) = openengine_test_server().await;

    let engine = client.get_engine_info(pb::GetEngineInfoRequest {}).await.unwrap().into_inner();
    assert_eq!(engine.schema_revision, 1);
    assert_eq!(engine.minimum_client_revision, 1);
    assert!(!engine.schema_release.is_empty());
    assert_eq!(engine.supported_models, vec!["test-model"]);

    let model = client
        .get_model_info(pb::GetModelInfoRequest {
            model: "test-model".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(model.max_context_length.is_some_and(|length| length > 0));
    let generation = model.generation.unwrap();
    assert!(generation.output_logprobs.as_ref().unwrap().supported.unwrap());
    let output_logprobs = generation.output_logprobs.unwrap();
    assert_eq!(output_logprobs.max_top_n, Some(20));
    assert!(
        output_logprobs
            .candidate_selection_modes
            .contains(&(pb::CandidateTokenSelectionMode::TokenIds as i32))
    );
    assert!(
        !output_logprobs
            .candidate_selection_modes
            .contains(&(pb::CandidateTokenSelectionMode::All as i32))
    );
    let prompt_logprobs = generation.prompt_logprobs.unwrap();
    assert_eq!(prompt_logprobs.max_top_n, Some(20));
    assert!(
        !prompt_logprobs
            .candidate_selection_modes
            .contains(&(pb::CandidateTokenSelectionMode::TokenIds as i32))
    );

    let response = pb::ResponseOptions {
        return_output_logprobs: Some(true),
        output_candidates: Some(pb::CandidateTokenSelection {
            selection: Some(pb::candidate_token_selection::Selection::TopN(2)),
        }),
        ..Default::default()
    };
    let request = pb::GenerateRequest {
        request_id: "openengine-req".to_string(),
        model: "test-model".to_string(),
        input: Some(pb::generate_request::Input::Prompt("hello".to_string())),
        stopping: Some(pb::StoppingOptions {
            max_tokens: Some(3),
            ..Default::default()
        }),
        response: Some(response),
        kv: Some(pb::KvOptions {
            cache_salt: Some("tenant-a".to_string()),
            ..Default::default()
        }),
        priority: Some(7),
        ..Default::default()
    };
    let mut stream = client.generate(request).await.unwrap().into_inner();
    let mut token_ids = Vec::new();
    let mut finished = false;
    while let Some(response) = stream.message().await.unwrap() {
        match response.event {
            Some(pb::generate_response::Event::Token(output)) => {
                token_ids.extend(output.tokens.into_iter().map(|token| token.token_id));
            }
            Some(pb::generate_response::Event::Finished(_)) => finished = true,
            Some(pb::generate_response::Event::Error(error)) => {
                panic!("unexpected OpenEngine error: {}", error.message)
            }
            _ => {}
        }
    }
    assert_eq!(token_ids, vec![b'h' as u32, b'i' as u32, b'!' as u32]);
    assert!(finished);

    server_task.abort();
    engine_task.await.unwrap();
}
