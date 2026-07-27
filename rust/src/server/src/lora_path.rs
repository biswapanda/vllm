//! Transport-neutral policy for resolving runtime LoRA adapter sources.

use std::fmt;
use std::path::{Component, Path, PathBuf};

pub(crate) const RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV: &str =
    "VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES";

#[derive(Debug)]
pub(crate) enum LoraPathError {
    LocalPathsDisabled,
    RelativeLocalPath,
    PathUnavailable,
    AllowedPrefixUnavailable(PathBuf),
    OutsideAllowedPrefixes,
}

impl LoraPathError {
    /// Misconfigured allowed prefixes are server errors; all other failures
    /// describe an invalid adapter source supplied by the caller.
    pub(crate) fn is_client_error(&self) -> bool {
        !matches!(self, Self::AllowedPrefixUnavailable(_))
    }

    /// Stable caller-facing error text. Configuration paths remain available
    /// through `Display` for server logs, but never cross the API boundary.
    pub(crate) fn public_message(&self) -> &'static str {
        match self {
            Self::LocalPathsDisabled => {
                "Local LoRA adapter paths require VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES to be configured."
            }
            Self::RelativeLocalPath => {
                "Local LoRA adapter paths must be absolute and under one of the prefixes configured by VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES."
            }
            Self::PathUnavailable => "Local LoRA adapter path must exist and be accessible.",
            Self::AllowedPrefixUnavailable(_) => {
                "Runtime LoRA path policy is unavailable; check the server configuration."
            }
            Self::OutsideAllowedPrefixes => {
                "Local LoRA adapter path is outside the configured allowed prefixes."
            }
        }
    }
}

impl fmt::Display for LoraPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalPathsDisabled => write!(
                formatter,
                "Local LoRA adapter paths require {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} to be configured."
            ),
            Self::RelativeLocalPath => write!(
                formatter,
                "Local LoRA adapter paths must be absolute and under one of the prefixes configured by {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV}."
            ),
            Self::PathUnavailable => {
                formatter.write_str("Local LoRA adapter path must exist and be accessible.")
            }
            Self::AllowedPrefixUnavailable(prefix) => write!(
                formatter,
                "configured {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} path prefix `{}` must exist and be accessible",
                prefix.display()
            ),
            Self::OutsideAllowedPrefixes => formatter
                .write_str("Local LoRA adapter path is outside the configured allowed prefixes."),
        }
    }
}

pub(crate) fn runtime_lora_allowed_path_prefixes() -> Option<Vec<PathBuf>> {
    let prefixes = std::env::var_os(RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV)?;
    let prefixes: Vec<_> = std::env::split_paths(&prefixes)
        .filter(|path| !path.as_os_str().is_empty())
        .collect();
    (!prefixes.is_empty()).then_some(prefixes)
}

fn looks_like_local_lora_path(lora_path: &str) -> bool {
    let path = Path::new(lora_path);
    path.is_absolute()
        || lora_path.starts_with('~')
        || lora_path.starts_with('.')
        || path.components().any(|component| matches!(component, Component::ParentDir))
}

/// Resolve a local adapter path under the configured allowlist.
///
/// A `None` result means `lora_path` is a non-local identifier such as a
/// Hugging Face repository id and therefore does not need filesystem policy.
pub(crate) async fn validate_lora_path_access(
    lora_path: &str,
    allowed_prefixes: Option<&[PathBuf]>,
) -> Result<Option<PathBuf>, LoraPathError> {
    let path = Path::new(lora_path);
    if !looks_like_local_lora_path(lora_path) {
        match tokio::fs::try_exists(path).await {
            Ok(false) => return Ok(None),
            Ok(true) => {}
            Err(_) => return Err(LoraPathError::PathUnavailable),
        }
    }

    let Some(allowed_prefixes) = allowed_prefixes else {
        return Err(LoraPathError::LocalPathsDisabled);
    };

    if !path.is_absolute() {
        return Err(LoraPathError::RelativeLocalPath);
    }

    let canonical_path = tokio::fs::canonicalize(path)
        .await
        .map_err(|_| LoraPathError::PathUnavailable)?;
    let mut canonical_prefixes = Vec::with_capacity(allowed_prefixes.len());
    for prefix in allowed_prefixes {
        canonical_prefixes.push(
            tokio::fs::canonicalize(prefix)
                .await
                .map_err(|_| LoraPathError::AllowedPrefixUnavailable(prefix.clone()))?,
        );
    }

    canonical_prefixes
        .iter()
        .any(|prefix| canonical_path.starts_with(prefix))
        .then_some(Some(canonical_path))
        .ok_or(LoraPathError::OutsideAllowedPrefixes)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{LoraPathError, validate_lora_path_access};

    fn temp_lora_dir(test_name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vllm-lora-{test_name}-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp lora dir");
        path
    }

    #[tokio::test]
    async fn allows_hf_repo_ids_without_prefixes() {
        assert_eq!(
            validate_lora_path_access("org/adapter-a", None)
                .await
                .expect("hf repo id should be allowed"),
            None
        );
    }

    #[tokio::test]
    async fn rejects_local_paths_without_prefixes() {
        for path in [
            "/tmp/adapter-a",
            "./adapter-a",
            "~/adapter-a",
            "subdir/../../../etc/sensitive",
        ] {
            assert!(matches!(
                validate_lora_path_access(path, None).await,
                Err(LoraPathError::LocalPathsDisabled)
            ));
        }
    }

    #[tokio::test]
    async fn rejects_existing_bare_relative_paths_without_prefixes() {
        let root =
            PathBuf::from("target").join(format!("vllm-lora-relative-{}", std::process::id()));
        let adapter = root.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create relative adapter dir");

        assert!(matches!(
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), None).await,
            Err(LoraPathError::LocalPathsDisabled)
        ));

        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn allows_absolute_paths_under_configured_prefixes() {
        let root = temp_lora_dir("allowed-prefix");
        let allowed = root.join("allowed");
        let adapter = allowed.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create adapter dir");

        let prefixes = [allowed];
        let resolved =
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .await
                .expect("path under configured prefix should be allowed");
        assert_eq!(
            resolved,
            Some(adapter.canonicalize().expect("canonical adapter"))
        );

        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn rejects_parent_escape_from_configured_prefixes() {
        let root = temp_lora_dir("parent-escape");
        let allowed = root.join("allowed");
        let private_adapter = root.join("private").join("adapter-a");
        fs::create_dir_all(&allowed).expect("create allowed dir");
        fs::create_dir_all(&private_adapter).expect("create private adapter dir");

        let escaped = allowed.join("../private/adapter-a");
        let prefixes = [allowed];
        assert!(matches!(
            validate_lora_path_access(escaped.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .await,
            Err(LoraPathError::OutsideAllowedPrefixes)
        ));

        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn unavailable_allowed_prefix_is_a_server_error() {
        let root = temp_lora_dir("missing-prefix");
        let adapter = root.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create adapter dir");
        let prefixes = [root.join("missing")];

        let error =
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .await
                .expect_err("missing configured prefix should fail");
        assert!(!error.is_client_error());
        assert!(error.to_string().contains(prefixes[0].to_str().expect("utf-8 prefix")));
        assert!(!error.public_message().contains(prefixes[0].to_str().expect("utf-8 prefix")));

        fs::remove_dir_all(root).ok();
    }
}
