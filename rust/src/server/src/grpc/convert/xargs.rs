use std::collections::HashMap;

use tonic::Status;

const MAX_VLLM_XARGS_BYTES: usize = 64 * 1024;
const MAX_VLLM_XARGS_KEYS: usize = 64;
const MAX_VLLM_XARGS_DEPTH: usize = 16;
const MAX_VLLM_XARGS_NODES: usize = 1024;
const MAX_KEY_BYTES: usize = 256;

pub(super) fn parse_vllm_xargs_json(
    raw: &[u8],
) -> Result<HashMap<String, serde_json::Value>, Status> {
    if raw.len() > MAX_VLLM_XARGS_BYTES {
        return Err(Status::resource_exhausted("vllm_xargs_json exceeds 64 KiB"));
    }
    let value: serde_json::Value = serde_json::from_slice(raw).map_err(|error| {
        Status::invalid_argument(format!("vllm_xargs_json must be a JSON object: {error}"))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| Status::invalid_argument("vllm_xargs_json must be a JSON object"))?;
    if object.len() > MAX_VLLM_XARGS_KEYS {
        return Err(Status::resource_exhausted(
            "vllm_xargs_json contains too many keys",
        ));
    }
    if object.contains_key("kv_transfer_params") {
        return Err(Status::invalid_argument(
            "vllm_xargs_json key kv_transfer_params is reserved for typed KV parameters",
        ));
    }
    let mut nodes = 0usize;
    validate_json_budget(&value, 0, &mut nodes)?;
    serde_json::from_value(value).map_err(|error| {
        Status::invalid_argument(format!("vllm_xargs_json must be a JSON object: {error}"))
    })
}

fn validate_json_budget(
    value: &serde_json::Value,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), Status> {
    if depth > MAX_VLLM_XARGS_DEPTH {
        return Err(Status::resource_exhausted(
            "vllm_xargs_json nesting exceeds 16 levels",
        ));
    }
    *nodes = nodes
        .checked_add(1)
        .ok_or_else(|| Status::resource_exhausted("vllm_xargs_json is too complex"))?;
    if *nodes > MAX_VLLM_XARGS_NODES {
        return Err(Status::resource_exhausted(
            "vllm_xargs_json contains too many values",
        ));
    }
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                validate_json_budget(value, depth + 1, nodes)?;
            }
        }
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                if key.len() > MAX_KEY_BYTES {
                    return Err(Status::resource_exhausted(
                        "vllm_xargs_json key exceeds 256 bytes",
                    ));
                }
                validate_json_budget(value, depth + 1, nodes)?;
            }
        }
        _ => {}
    }
    Ok(())
}
