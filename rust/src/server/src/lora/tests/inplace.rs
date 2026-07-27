use super::*;

struct ExactLoadFixture {
    client: EngineCoreClient,
    manager: LoraManager,
    shutdown: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl ExactLoadFixture {
    async fn start(results: impl IntoIterator<Item = bool>) -> Self {
        let results = results.into_iter().collect::<Vec<_>>();
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (shutdown, task) = spawn_mock_engine_task_with_ready(
            handshake.clone(),
            vec![0x00, 0x00],
            vllm_engine_core_client::mock_engine::default_ready_response(),
            move |dealer, push| {
                Box::pin(async move {
                    for result in results {
                        let load = recv_utility_call_id(dealer, "add_lora").await;
                        reply_utility(push, load, result).await;
                    }
                })
            },
        );
        let config = EngineCoreClientConfig::new_single(handshake)
            .with_model_name("test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            );
        Self {
            client: EngineCoreClient::connect(config).await.unwrap(),
            manager: LoraManager::new(),
            shutdown,
            task,
        }
    }

    fn adapter(path: &str) -> LoraRequest {
        LoraRequest::new("adapter-a".to_string(), 17, path.to_string(), false, false)
    }

    async fn load_exact(&self, path: &str, load_inplace: bool) -> Result<bool, LoadExactLoraError> {
        self.manager
            .load_lora_exact(
                &self.client,
                &["test-model".to_string()],
                Self::adapter(path),
                load_inplace,
            )
            .await
            .map(|(_, already_loaded)| already_loaded)
    }

    async fn finish(self) {
        let _ = self.shutdown.send(());
        self.task.await.unwrap();
        self.client.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_load_inplace_reloads_same_or_different_path_without_persisting_flag() {
    for (first_path, replacement_path) in [
        ("/adapters/step-1", "/adapters/step-2"),
        ("/adapters/stable", "/adapters/stable"),
    ] {
        let fixture = ExactLoadFixture::start([true, true]).await;
        fixture.load_exact(first_path, false).await.unwrap();

        let already_loaded = fixture.load_exact(replacement_path, true).await.unwrap();
        let loaded = fixture.manager.served_lora_requests().await;

        assert!(!already_loaded, "in-place load must reach the engine");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].lora_name, "adapter-a");
        assert_eq!(loaded[0].lora_int_id, 17);
        assert_eq!(loaded[0].lora_path, replacement_path);
        assert!(!loaded[0].load_inplace);
        fixture.finish().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_same_path_inplace_load_fails_closed() {
    let fixture = ExactLoadFixture::start([true, false]).await;
    fixture.load_exact("/adapters/stable", false).await.unwrap();

    let error = fixture.load_exact("/adapters/stable", true).await.unwrap_err();

    assert!(matches!(error, LoadExactLoraError::NotLoaded { .. }));
    assert!(!fixture.manager.is_consistent());
    fixture.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn named_load_inplace_does_not_persist_the_mutation_flag() {
    let fixture = ExactLoadFixture::start([true, true]).await;
    fixture
        .manager
        .load_lora(
            &fixture.client,
            &["test-model".to_string()],
            "adapter-a".to_string(),
            "/adapters/step-1".to_string(),
            false,
            false,
        )
        .await
        .unwrap();
    fixture
        .manager
        .load_lora(
            &fixture.client,
            &["test-model".to_string()],
            "adapter-a".to_string(),
            "/adapters/step-2".to_string(),
            true,
            false,
        )
        .await
        .unwrap();

    assert!(!fixture.manager.served_lora_requests().await[0].load_inplace);
    fixture.finish().await;
}
