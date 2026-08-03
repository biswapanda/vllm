// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::extract::State;
use serde::Deserialize;
use thiserror_ext::AsReport;
use validator::Validate;

use crate::error::ApiError;
use crate::lora::{LoadLoraError, UnloadLoraError};
use crate::lora_path::{runtime_lora_allowed_path_prefixes, validate_lora_path_access};
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct LoadLoraAdapterRequest {
    lora_name: String,
    lora_path: String,
    #[serde(default)]
    load_inplace: bool,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

impl Normalizable for LoadLoraAdapterRequest {}

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct UnloadLoraAdapterRequest {
    lora_name: String,
    #[serde(default)]
    lora_int_id: Option<u64>,
}

impl Normalizable for UnloadLoraAdapterRequest {}

/// Dynamically load one LoRA adapter and expose it as an OpenAI model id.
pub async fn load_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<LoadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() || request.lora_path.is_empty() {
        return Err(ApiError::invalid_request(
            "Both 'lora_name' and 'lora_path' must be provided.".to_string(),
            None,
        ));
    }
    let allowed_prefixes = runtime_lora_allowed_path_prefixes();
    let lora_path = validate_lora_path_access(&request.lora_path, allowed_prefixes.as_deref())
        .await
        .map_err(|error| {
            if error.is_client_error() {
                ApiError::invalid_request(error.public_message().to_string(), Some("lora_path"))
            } else {
                tracing::error!(error = %error, "runtime LoRA path policy validation failed");
                ApiError::server_error(error.public_message().to_string())
            }
        })?
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or(request.lora_path);

    let lora_name = request.lora_name;
    state
        .load_lora(
            lora_name.clone(),
            lora_path,
            request.load_inplace,
            request.is_3d_lora_weight,
        )
        .await
        .map_err(|error| match error {
            LoadLoraError::Inconsistent => ApiError::server_error(
                "LoRA state differs across engine ranks; restart the engine".to_string(),
            ),
            LoadLoraError::AlreadyLoaded { lora_name } => ApiError::invalid_request(
                format!(
                    "The lora adapter '{lora_name}' has already been loaded. If you want to load the adapter in place, set 'load_inplace' to true."
                ),
                Some("lora_name"),
            ),
            LoadLoraError::BaseModelName { lora_name } => ApiError::invalid_request(
                format!("The lora adapter name '{lora_name}' conflicts with a served base model."),
                Some("lora_name"),
            ),
            LoadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': {}",
                error.to_report_string()
            )),
            LoadLoraError::NotLoaded { lora_name } => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': engine rejected the adapter"
            )),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{lora_name}' added successfully."
    ))
}

/// Remove one LoRA adapter from the engine and frontend registry.
pub async fn unload_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<UnloadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() {
        return Err(ApiError::invalid_request(
            "'lora_name' needs to be provided to unload a LoRA adapter.".to_string(),
            Some("lora_name"),
        ));
    }

    let lora_request = state
        .unload_lora(&request.lora_name, request.lora_int_id)
        .await
        .map_err(|error| match error {
            UnloadLoraError::Inconsistent => ApiError::server_error(
                "LoRA state differs across engine ranks; restart the engine".to_string(),
            ),
            UnloadLoraError::NotFound { lora_name } => ApiError::model_not_found(lora_name),
            UnloadLoraError::IntIdMismatch {
                lora_name,
                expected,
                actual,
            } => ApiError::invalid_request(
                format!(
                    "The requested lora_int_id {actual} does not match loaded adapter '{lora_name}' with id {expected}."
                ),
                Some("lora_int_id"),
            ),
            UnloadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{}': {}",
                request.lora_name,
                error.to_report_string()
            )),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{}' removed successfully.",
        lora_request.lora_name
    ))
}
