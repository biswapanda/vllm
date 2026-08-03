// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC services backed by the shared application state.

mod control;
mod convert;
mod health;
mod inference;
mod lora_rpc;
mod struct_json;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Generated protobuf/gRPC types for the `vllm` package.
pub mod pb {
    tonic::include_proto!("vllm");
}

pub(crate) use control::ControlGrpcService;
pub use control::ControlServiceImpl;
pub(crate) use health::monitor_health;
pub(crate) use inference::InferenceGrpcService;
pub use inference::InferenceServiceImpl;
pub use pb::control_server::ControlServer;
pub use pb::inference_server::InferenceServer;

/// Drain/admission state shared by the inference and control services.
///
/// `Drain` arrives on the control service but must stop admitting work on the
/// inference service, so both hold the same `Arc`.
#[derive(Default)]
pub(crate) struct AdmissionState {
    draining: AtomicBool,
    in_flight: AtomicU64,
}

impl AdmissionState {
    fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    pub(crate) fn begin_drain(&self) {
        self.draining.store(true, Ordering::SeqCst);
    }

    pub(crate) fn in_flight(&self) -> u64 {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// Reserve an in-flight slot, or return `None` once draining has begun.
    pub(crate) fn try_admit(self: &Arc<Self>) -> Option<AdmissionGuard> {
        if self.is_draining() {
            return None;
        }
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        // Re-check: `begin_drain` may land between the check and the increment.
        if self.is_draining() {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(AdmissionGuard(self.clone()))
    }
}

pub(crate) struct AdmissionGuard(Arc<AdmissionState>);

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests;
