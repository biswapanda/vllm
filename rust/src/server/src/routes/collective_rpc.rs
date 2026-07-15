use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use rmpv::Value as MsgpackValue;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize)]
pub(crate) struct CollectiveRpcRequest {
    method: Option<String>,
    #[serde(default)]
    timeout: Option<f64>,
    #[serde(default)]
    args: Vec<JsonValue>,
    #[serde(default)]
    kwargs: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CollectiveRpcResponse {
    results: Vec<MsgpackValue>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UpdateWeightsRequest {
    weight_dir: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct UpdateWeightsResponse {
    status: &'static str,
}

/// Execute a development-only collective RPC on the connected engine(s).
pub async fn collective_rpc(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CollectiveRpcRequest>, JsonRejection>,
) -> Result<Json<CollectiveRpcResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let method = body.method.ok_or_else(|| {
        ApiError::invalid_request(
            "Missing 'method' in request body".to_string(),
            Some("method"),
        )
    })?;

    let results = state
        .engine_core_client()
        .collective_rpc(&method, body.timeout, body.args, body.kwargs)
        .await
        .map_err(|error| utility_call_error("collective_rpc", error))?;

    Ok(Json(CollectiveRpcResponse { results }))
}

/// Prime-RL compatible filesystem weight-update wrapper over vLLM's existing
/// collective worker-extension RPC.
pub async fn update_weights(
    State(state): State<Arc<AppState>>,
    body: Result<Json<UpdateWeightsRequest>, JsonRejection>,
) -> Result<Json<UpdateWeightsResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let weight_dir = body.weight_dir.map_or(JsonValue::Null, JsonValue::String);
    state
        .engine_core_client()
        .collective_rpc(
            "update_weights_from_path",
            None,
            vec![weight_dir],
            BTreeMap::<String, JsonValue>::new(),
        )
        .await
        .map_err(|error| utility_call_error("update_weights", error))?;
    Ok(Json(UpdateWeightsResponse { status: "ok" }))
}
