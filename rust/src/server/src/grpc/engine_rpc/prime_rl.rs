use std::collections::BTreeMap;
use std::future::Future;
use std::str::FromStr as _;
use std::time::Duration;

use serde_json::Value;
use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::protocol::utility::PauseMode;

use super::EngineServiceImpl;

pub mod pb {
    tonic::include_proto!("prime_rl.engine.v1");
}

pub use pb::prime_rl_engine_server::PrimeRlEngineServer;

const SCHEDULER_OPERATION_TIMEOUT: Duration = Duration::from_secs(290);
const DEFAULT_GROUP_OPERATION_TIMEOUT: Duration = Duration::from_secs(290);
const GROUP_TIMEOUT_MARGIN_SECS: u64 = 10;
const MAX_GROUP_TIMEOUT_SECS: u64 = 86_400;
const PAUSE_CONSENSUS_TIMEOUT: Duration = Duration::from_secs(10);
const WEIGHT_OPERATION_TIMEOUT: Duration = Duration::from_secs(640);
const CACHE_RESET_TIMEOUT: Duration = Duration::from_secs(60);

pub(super) struct RlAdminState {
    paused: bool,
    drained: bool,
    indeterminate: Option<String>,
    update_group: Option<UpdateGroupConfig>,
    group_epoch: u64,
    committed_updates: BTreeMap<(u64, String), WeightUpdateIdentity>,
    weight_version: String,
}

#[derive(Clone)]
struct UpdateGroupConfig {
    operation_timeout: Duration,
}

#[derive(Clone, PartialEq)]
struct WeightUpdateIdentity {
    method: String,
    kwargs: BTreeMap<String, Value>,
    version: String,
    distributed: bool,
    controller_epoch: u64,
}

impl Default for RlAdminState {
    fn default() -> Self {
        Self {
            paused: false,
            drained: false,
            indeterminate: None,
            update_group: None,
            group_epoch: 0,
            committed_updates: BTreeMap::new(),
            weight_version: "initial".to_string(),
        }
    }
}

#[tonic::async_trait]
impl pb::prime_rl_engine_server::PrimeRlEngine for EngineServiceImpl {
    async fn liveness_probe(
        &self,
        _request: Request<pb::LivenessProbeRequest>,
    ) -> Result<Response<pb::AdminResponse>, Status> {
        ensure_determinate_flag(self)?;
        if !self.state.engine_core_client().is_healthy() {
            return Err(Status::unavailable(
                self.state
                    .engine_core_client()
                    .health_error()
                    .map(|error| error.to_report_string())
                    .unwrap_or_else(|| "engine is not healthy".to_string()),
            ));
        }
        Ok(Response::new(admin_ok("alive")))
    }

    async fn pause_generation(
        &self,
        request: Request<pb::PauseGenerationRequest>,
    ) -> Result<Response<pb::AdminResponse>, Status> {
        let request = request.into_inner();
        let mode = if request.mode.is_empty() {
            PauseMode::Wait
        } else {
            PauseMode::from_str(&request.mode).map_err(Status::invalid_argument)?
        };
        let mut state = self.rl_admin.lock().await;
        ensure_determinate(self, &state)?;
        if let Err(error) = uncertain_engine_call(
            self,
            "pause_generation",
            SCHEDULER_OPERATION_TIMEOUT,
            self.state.engine_core_client().pause_scheduler(mode, request.clear_cache),
        )
        .await
        {
            latch_indeterminate(self, &mut state, &error);
            return Err(error);
        }
        state.paused = true;
        state.drained = mode != PauseMode::Keep;
        Ok(Response::new(admin_ok("paused")))
    }

    async fn resume_generation(
        &self,
        _request: Request<pb::ResumeGenerationRequest>,
    ) -> Result<Response<pb::AdminResponse>, Status> {
        let mut state = self.rl_admin.lock().await;
        ensure_determinate(self, &state)?;
        if let Err(error) = uncertain_engine_call(
            self,
            "resume_generation",
            SCHEDULER_OPERATION_TIMEOUT,
            self.state.engine_core_client().resume_scheduler(),
        )
        .await
        {
            latch_indeterminate(self, &mut state, &error);
            return Err(error);
        }
        state.paused = false;
        state.drained = false;
        Ok(Response::new(admin_ok("resumed")))
    }

    async fn flush_cache(
        &self,
        request: Request<pb::FlushCacheRequest>,
    ) -> Result<Response<pb::AdminResponse>, Status> {
        let request = request.into_inner();
        let mut state = self.rl_admin.lock().await;
        ensure_determinate(self, &state)?;
        let reset = match uncertain_engine_call(
            self,
            "flush_cache",
            SCHEDULER_OPERATION_TIMEOUT,
            self.state
                .engine_core_client()
                .reset_prefix_cache(request.reset_running_requests, request.reset_connector),
        )
        .await
        {
            Ok(reset) => reset,
            Err(error) => {
                latch_indeterminate(self, &mut state, &error);
                return Err(error);
            }
        };
        if !reset {
            return Err(Status::failed_precondition(
                "prefix/KV/connector cache reset did not complete",
            ));
        }
        Ok(Response::new(admin_ok("cache flushed")))
    }

    async fn abort_request(
        &self,
        request: Request<pb::AbortRequestRequest>,
    ) -> Result<Response<pb::AdminResponse>, Status> {
        let request_id = request.into_inner().request_id;
        if request_id.trim().is_empty() {
            return Err(Status::invalid_argument("request_id is required"));
        }
        self.state
            .chat
            .abort(std::slice::from_ref(&request_id))
            .await
            .map_err(internal("abort_request"))?;
        Ok(Response::new(admin_ok("aborted")))
    }

    async fn init_weights_update_group(
        &self,
        request: Request<pb::InitWeightsUpdateGroupRequest>,
    ) -> Result<Response<pb::AdminResponse>, Status> {
        let request = request.into_inner();
        if request.host.trim().is_empty() {
            return Err(Status::invalid_argument("host is required"));
        }
        if request.port == 0 || request.inference_world_size == 0 {
            return Err(Status::invalid_argument(
                "port and inference_world_size must be positive",
            ));
        }
        let method = allowed_method(
            &request.engine_rpc,
            "init_broadcaster",
            &["init_broadcaster"],
        )?;
        let kwargs = BTreeMap::from([
            ("host".to_string(), Value::String(request.host)),
            ("port".to_string(), Value::from(request.port)),
            ("rank_offset".to_string(), Value::from(request.rank_offset)),
            (
                "inference_world_size".to_string(),
                Value::from(request.inference_world_size),
            ),
            ("timeout".to_string(), Value::from(request.timeout)),
            (
                "quantize_in_weight_transfer".to_string(),
                Value::Bool(request.quantize_in_weight_transfer),
            ),
        ]);
        let operation_timeout = group_operation_timeout(request.timeout)?;
        let mut state = self.rl_admin.lock().await;
        ensure_determinate(self, &state)?;
        if state.update_group.is_some() {
            return Err(Status::failed_precondition(
                "a weight update group is already initialized; destroy it before creating a new controller epoch",
            ));
        }
        let next_group_epoch = state
            .group_epoch
            .checked_add(1)
            .ok_or_else(|| Status::failed_precondition("weight update group epoch exhausted"))?;
        if let Err(error) = collective(self, method, kwargs, operation_timeout).await {
            latch_indeterminate(self, &mut state, &error);
            return Err(error);
        }
        state.group_epoch = next_group_epoch;
        state.update_group = Some(UpdateGroupConfig { operation_timeout });
        Ok(Response::new(admin_ok("weight update group initialized")))
    }

    async fn destroy_weights_update_group(
        &self,
        request: Request<pb::DestroyWeightsUpdateGroupRequest>,
    ) -> Result<Response<pb::AdminResponse>, Status> {
        let request = request.into_inner();
        let method = allowed_method(
            &request.engine_rpc,
            "destroy_broadcaster",
            &["destroy_broadcaster"],
        )?;
        let mut state = self.rl_admin.lock().await;
        ensure_determinate(self, &state)?;
        if state.update_group.is_none() {
            return Ok(Response::new(admin_ok(
                "weight update group already destroyed",
            )));
        }
        let operation_timeout = state
            .update_group
            .as_ref()
            .map(|config| config.operation_timeout)
            .unwrap_or(DEFAULT_GROUP_OPERATION_TIMEOUT);
        if let Err(error) = collective(self, method, BTreeMap::new(), operation_timeout).await {
            latch_indeterminate(self, &mut state, &error);
            return Err(error);
        }
        state.update_group = None;
        Ok(Response::new(admin_ok("weight update group destroyed")))
    }

    async fn update_weights_from_disk(
        &self,
        request: Request<pb::UpdateWeightsFromDiskRequest>,
    ) -> Result<Response<pb::WeightUpdateResponse>, Status> {
        let request = request.into_inner();
        if request.model_path.trim().is_empty() {
            return Err(Status::invalid_argument("model_path is required"));
        }
        let method = allowed_method(
            &request.engine_rpc,
            "reload_weights",
            &["reload_weights", "update_weights_from_path"],
        )?;
        let key = if method == "reload_weights" {
            "weights_path"
        } else {
            "weight_path"
        };
        let kwargs = BTreeMap::from([(key.to_string(), Value::String(request.model_path))]);
        update_weights(self, method, kwargs, request.weight_version, false).await
    }

    async fn update_weights_from_distributed(
        &self,
        request: Request<pb::UpdateWeightsFromDistributedRequest>,
    ) -> Result<Response<pb::WeightUpdateResponse>, Status> {
        let request = request.into_inner();
        if request.allow_unpaused {
            return Err(Status::invalid_argument(
                "unpaused weight updates are unsafe and are not supported",
            ));
        }
        if !request.reset_prefix_cache {
            return Err(Status::invalid_argument(
                "weight updates require prefix/KV/connector cache reset",
            ));
        }
        let method = allowed_method(
            &request.engine_rpc,
            "update_weights_from_path",
            &["update_weights_from_path"],
        )?;
        let kwargs = if request.weight_dir.is_empty() {
            BTreeMap::new()
        } else {
            BTreeMap::from([("weight_dir".to_string(), Value::String(request.weight_dir))])
        };
        update_weights(self, method, kwargs, request.weight_version, true).await
    }

    async fn get_weight_version(
        &self,
        _request: Request<pb::GetWeightVersionRequest>,
    ) -> Result<Response<pb::GetWeightVersionResponse>, Status> {
        let state = self.rl_admin.lock().await;
        ensure_determinate(self, &state)?;
        Ok(Response::new(pb::GetWeightVersionResponse {
            status: "ok".to_string(),
            weight_version: state.weight_version.clone(),
        }))
    }
}

async fn update_weights(
    service: &EngineServiceImpl,
    method: &str,
    kwargs: BTreeMap<String, Value>,
    version: String,
    requires_update_group: bool,
) -> Result<Response<pb::WeightUpdateResponse>, Status> {
    if version.trim().is_empty() || version == "unknown" {
        return Err(Status::invalid_argument(
            "weight_version must be a stable, non-empty identifier",
        ));
    }
    let mut state = service.rl_admin.lock().await;
    ensure_determinate(service, &state)?;
    if !state.paused {
        return Err(Status::failed_precondition(
            "pause_generation must succeed before updating weights",
        ));
    }
    if !state.drained {
        return Err(Status::failed_precondition(
            "weight updates require pause mode `wait` or `abort`",
        ));
    }
    if requires_update_group && state.update_group.is_none() {
        return Err(Status::failed_precondition(
            "init_weights_update_group must succeed before a distributed weight update",
        ));
    }

    let scheduler_paused = match bounded_engine_call(
        "verify_scheduler_paused",
        PAUSE_CONSENSUS_TIMEOUT,
        service.state.engine_core_client().is_scheduler_paused(),
    )
    .await
    {
        Ok(paused) => paused,
        Err(error) => {
            latch_indeterminate(service, &mut state, &error);
            return Err(error);
        }
    };
    if !scheduler_paused {
        state.paused = false;
        state.drained = false;
        return Err(Status::failed_precondition(
            "scheduler is no longer paused on every engine rank",
        ));
    }

    let update_identity = WeightUpdateIdentity {
        method: method.to_string(),
        kwargs: kwargs.clone(),
        version: version.clone(),
        distributed: requires_update_group,
        controller_epoch: if requires_update_group {
            state.group_epoch
        } else {
            0
        },
    };
    let replay_key = (update_identity.controller_epoch, version.clone());
    if let Some(previous) = state.committed_updates.get(&replay_key) {
        if previous == &update_identity {
            return Err(Status::failed_precondition(format!(
                "weight_version `{version}` was already committed in controller epoch {}",
                update_identity.controller_epoch
            )));
        }
        return Err(Status::failed_precondition(format!(
            "weight_version `{version}` was already used for a different update in controller epoch {}",
            update_identity.controller_epoch
        )));
    }

    if let Err(error) = collective(service, method, kwargs, WEIGHT_OPERATION_TIMEOUT).await {
        latch_indeterminate(service, &mut state, &error);
        return Err(error);
    }
    match uncertain_engine_call(
        service,
        "reset_prefix_cache_after_weight_update",
        CACHE_RESET_TIMEOUT,
        service.state.engine_core_client().reset_prefix_cache(false, true),
    )
    .await
    {
        Ok(true) => {}
        Ok(false) => {
            let message = "prefix/KV/connector cache reset failed after weight update";
            let error = Status::failed_precondition(message);
            latch_indeterminate(service, &mut state, &error);
            return Err(error);
        }
        Err(error) => {
            latch_indeterminate(service, &mut state, &error);
            return Err(error);
        }
    }

    state.weight_version = version.clone();
    state.committed_updates.insert(replay_key, update_identity);
    Ok(Response::new(pb::WeightUpdateResponse {
        status: "ok".to_string(),
        message: "weights updated".to_string(),
        weight_version: state.weight_version.clone(),
    }))
}

async fn collective(
    service: &EngineServiceImpl,
    method: &str,
    kwargs: BTreeMap<String, Value>,
    timeout: Duration,
) -> Result<(), Status> {
    uncertain_engine_call(
        service,
        method,
        timeout,
        service.state.engine_core_client().collective_rpc(
            method,
            None,
            Vec::<Value>::new(),
            kwargs,
        ),
    )
    .await?;
    Ok(())
}

async fn uncertain_engine_call<T, E, F>(
    service: &EngineServiceImpl,
    operation: &str,
    timeout: Duration,
    future: F,
) -> Result<T, Status>
where
    E: thiserror_ext::AsReport,
    F: Future<Output = Result<T, E>>,
{
    let mut guard = UncertaintyGuard::new(service);
    let result = bounded_engine_call(operation, timeout, future).await;
    if result.is_ok() {
        guard.disarm();
    }
    result
}

async fn bounded_engine_call<T, E, F>(
    operation: &str,
    timeout: Duration,
    future: F,
) -> Result<T, Status>
where
    E: thiserror_ext::AsReport,
    F: Future<Output = Result<T, E>>,
{
    match tokio::time::timeout(timeout, future).await {
        Ok(result) => result.map_err(internal(operation)),
        Err(_) => Err(Status::deadline_exceeded(format!(
            "{operation} exceeded the server-side timeout of {timeout:?}"
        ))),
    }
}

fn latch_indeterminate(service: &EngineServiceImpl, state: &mut RlAdminState, error: &Status) {
    state.indeterminate = Some(error.message().to_string());
    service.mark_rl_indeterminate();
}

fn ensure_determinate_flag(service: &EngineServiceImpl) -> Result<(), Status> {
    if service.is_rl_indeterminate() {
        Err(Status::failed_precondition(
            "engine weight state is indeterminate and requires restart",
        ))
    } else {
        Ok(())
    }
}

fn ensure_determinate(service: &EngineServiceImpl, state: &RlAdminState) -> Result<(), Status> {
    ensure_determinate_flag(service)?;
    match state.indeterminate.as_ref() {
        Some(reason) => Err(Status::failed_precondition(format!(
            "engine weight state is indeterminate and requires restart: {reason}"
        ))),
        None => Ok(()),
    }
}

struct UncertaintyGuard<'a> {
    service: &'a EngineServiceImpl,
    armed: bool,
}

impl<'a> UncertaintyGuard<'a> {
    fn new(service: &'a EngineServiceImpl) -> Self {
        Self {
            service,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for UncertaintyGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.service.mark_rl_indeterminate();
        }
    }
}

fn default_method<'a>(value: &'a str, default: &'a str) -> &'a str {
    if value.trim().is_empty() {
        default
    } else {
        value
    }
}

fn allowed_method<'a>(
    value: &'a str,
    default: &'a str,
    allowed: &[&str],
) -> Result<&'a str, Status> {
    let method = default_method(value, default);
    if allowed.contains(&method) {
        Ok(method)
    } else {
        Err(Status::invalid_argument(format!(
            "unsupported engine_rpc `{method}` for this operation"
        )))
    }
}

fn group_operation_timeout(engine_timeout_secs: u64) -> Result<Duration, Status> {
    if engine_timeout_secs > MAX_GROUP_TIMEOUT_SECS {
        return Err(Status::invalid_argument(format!(
            "timeout exceeds the maximum of {MAX_GROUP_TIMEOUT_SECS} seconds"
        )));
    }
    if engine_timeout_secs == 0 {
        return Ok(DEFAULT_GROUP_OPERATION_TIMEOUT);
    }
    Ok(Duration::from_secs(
        engine_timeout_secs.saturating_add(GROUP_TIMEOUT_MARGIN_SECS),
    ))
}

fn admin_ok(message: &str) -> pb::AdminResponse {
    pb::AdminResponse {
        status: "ok".to_string(),
        message: message.to_string(),
    }
}

fn internal<E: thiserror_ext::AsReport>(operation: impl Into<String>) -> impl FnOnce(E) -> Status {
    let operation = operation.into();
    move |error| Status::internal(format!("{operation}: {}", error.to_report_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collective_rpc_method_is_restricted_per_operation() {
        assert_eq!(
            allowed_method("", "reload_weights", &["reload_weights"]).unwrap(),
            "reload_weights"
        );
        assert!(
            allowed_method(
                "run_arbitrary_collective",
                "reload_weights",
                &["reload_weights"]
            )
            .is_err()
        );
    }

    #[test]
    fn group_timeout_wraps_engine_deadline_with_margin() {
        assert_eq!(
            group_operation_timeout(1_200).unwrap(),
            Duration::from_secs(1_210)
        );
        assert_eq!(
            group_operation_timeout(0).unwrap(),
            DEFAULT_GROUP_OPERATION_TIMEOUT
        );
        assert!(group_operation_timeout(MAX_GROUP_TIMEOUT_SECS + 1).is_err());
    }

    #[test]
    fn distributed_replay_identity_is_scoped_to_group_epoch() {
        let mut state = RlAdminState::default();
        let identity = WeightUpdateIdentity {
            method: "update_weights_from_path".to_string(),
            kwargs: BTreeMap::new(),
            version: "1".to_string(),
            distributed: true,
            controller_epoch: 1,
        };
        state.committed_updates.insert((1, "1".to_string()), identity);

        assert!(state.committed_updates.contains_key(&(1, "1".to_string())));
        assert!(!state.committed_updates.contains_key(&(2, "1".to_string())));
    }
}
