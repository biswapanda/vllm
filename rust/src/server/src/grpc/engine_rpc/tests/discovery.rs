use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn engine_info_reports_aggregated_role_and_topology() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let info = client
        .get_engine_info(pb::GetEngineInfoRequest {})
        .await
        .expect("get_engine_info")
        .into_inner();
    assert_eq!(info.engine_name, "vllm");
    assert_eq!(info.engine_version, "test-vllm-version");
    assert_eq!(info.api_version, "vllm.engine.v1");
    assert_eq!(info.role, pb::EngineRole::Aggregated as i32);
    assert_eq!(info.supported_models, vec!["test-model".to_string()]);
    let parallelism = info.parallelism.expect("parallelism present");
    assert_eq!(parallelism.tensor_parallel_size, 1);
    assert_eq!(parallelism.pipeline_parallel_size, 1);
    assert_eq!(parallelism.data_parallel_size, 1);
    assert!(!info.kv_connector.expect("kv connector").enabled);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn model_info_reports_caps_from_handshake() {
    let (mut client, server_task, _engine_task) =
        engine_rpc_test_server(&[0x00, 0x00], default_stream_output_specs()).await;
    let info = client
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .expect("get_model_info")
        .into_inner();
    assert_eq!(info.model_id, "test-model");
    assert_eq!(info.served_model_name, "test-model");
    assert!(info.served_model_aliases.is_empty());
    assert_eq!(info.kv_block_size, 16);
    assert_eq!(info.max_running_requests, 256);
    assert_eq!(info.max_batched_tokens, 8192);
    assert!(info.supports_text_input);
    assert!(info.supports_token_ids_input);
    assert!(!info.supports_multimodal);
    assert!(!info.supports_lora);
    assert_eq!(info.max_loras, 0);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn model_info_reports_effective_parser_names() {
    let (mut client, server_task, _engine_task) = engine_rpc_test_server_with_parsers(
        &[0x00, 0x00],
        default_stream_output_specs(),
        vllm_chat::ParserSelection::Explicit("hermes".to_string()),
        vllm_chat::ParserSelection::Explicit("deepseek_r1".to_string()),
    )
    .await;
    let info = client
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .expect("get_model_info")
        .into_inner();
    assert_eq!(info.tool_call_parser, "hermes");
    assert_eq!(info.reasoning_parser, "deepseek_r1");
    server_task.abort();
}
