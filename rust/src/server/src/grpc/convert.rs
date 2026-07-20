//! Conversion between gRPC protobuf types and internal `vllm-text`
//! request/response types.

mod multimodal;
mod sampling;
mod xargs;

use multimodal::convert_mm_features;
pub(crate) use multimodal::media_parts_from_request;
#[cfg(test)]
use multimodal::{mm_cache_identifier, preflight_msgpack};
use sampling::build_sampling_params;
use xargs::parse_vllm_xargs_json;

use tonic::Status;
use uuid::Uuid;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};
use vllm_text::{
    DecodedLogprobs, DecodedPromptLogprobs, FinishReason, Finished, Prompt, TextDecodeOptions,
    TextRequest,
};

use super::pb;
use super::struct_json::{json_to_prost_struct, prost_struct_to_json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvRole {
    Aggregated,
    Prefill,
    Decode,
}

pub fn role_from_kv_role(kv_role: Option<&str>) -> KvRole {
    match kv_role {
        Some("kv_producer") => KvRole::Prefill,
        Some("kv_consumer") => KvRole::Decode,
        _ => KvRole::Aggregated,
    }
}

pub fn validate_disaggregated_request(
    request: &pb::GenerateRequest,
    role: KvRole,
) -> Result<(), Status> {
    let has_transfer_params = request
        .kv
        .as_ref()
        .and_then(|kv| kv.kv_transfer_params.as_ref())
        .is_some_and(|params| !params.fields.is_empty());
    if role == KvRole::Decode && !has_transfer_params {
        return Err(Status::invalid_argument(
            "kv.kv_transfer_params is required for decode requests",
        ));
    }
    Ok(())
}

pub fn mark_prefill_request(request: &mut TextRequest) {
    let params = request
        .sampling_params
        .vllm_xargs
        .get_or_insert_with(Default::default)
        .entry("kv_transfer_params".to_string())
        .or_insert_with(|| serde_json::Value::Object(Default::default()));
    if let Some(params) = params.as_object_mut() {
        params.insert(
            "do_remote_decode".to_string(),
            serde_json::Value::Bool(true),
        );
    }
}

// ========================================================================================
// Request conversion
// ========================================================================================

/// Convert a gRPC `GenerateRequest` into the internal `TextRequest`.
///
/// If `req.model` is non-empty, it must match one of `served_model_names`;
/// otherwise the request is rejected with `NotFound`. An empty string is
/// treated as "unset" (proto3 default) and accepted.
pub fn to_text_request(
    req: pb::GenerateRequest,
    stream: bool,
    served_model_names: &[String],
) -> Result<TextRequest, Status> {
    if !req.model.is_empty() && !served_model_names.iter().any(|n| n == &req.model) {
        return Err(Status::not_found(format!(
            "model `{}` not found",
            req.model
        )));
    }

    if req.truncate_prompt_tokens != 0 {
        return Err(Status::invalid_argument(
            "truncate_prompt_tokens is not supported",
        ));
    }

    if !req.media.is_empty() && !req.mm_features.is_empty() {
        return Err(Status::invalid_argument(
            "media and mm_features are mutually exclusive",
        ));
    }
    if !req.lora_name.is_empty() && (!req.media.is_empty() || !req.mm_features.is_empty()) {
        return Err(Status::invalid_argument(
            "native gRPC does not yet advertise tower-LoRA multimodal cache semantics; multimodal requests with LoRA are unsupported",
        ));
    }

    let prompt = match req.prompt {
        Some(pb::generate_request::Prompt::Text(text)) => Prompt::Text(text),
        Some(pb::generate_request::Prompt::TokenIds(ids)) => Prompt::TokenIds(ids.ids),
        None => return Err(Status::invalid_argument("prompt is required")),
    };
    match &prompt {
        Prompt::TokenIds(ids) if req.routed_experts_prompt_start as usize >= ids.len() => {
            return Err(Status::invalid_argument(
                "routed_experts_prompt_start must be less than the prompt length",
            ));
        }
        Prompt::Text(_) if req.routed_experts_prompt_start != 0 => {
            return Err(Status::invalid_argument(
                "nonzero routed_experts_prompt_start requires a token-ID prompt",
            ));
        }
        _ => {}
    }

    let request_id = if req.request_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        req.request_id
    };

    let sampling = req.sampling.as_ref();
    let decoding = req.decoding.as_ref();
    let stopping = req.stopping.as_ref();
    let response = req.response.as_ref();
    let kv = req.kv.as_ref();

    let mut sampling_params =
        build_sampling_params(req.temperature, sampling, decoding, stopping, response)?;
    sampling_params.routed_experts_prompt_start = req.routed_experts_prompt_start;

    if let Some(raw) = req.vllm_xargs_json.as_deref() {
        let xargs = parse_vllm_xargs_json(raw)?;
        sampling_params.vllm_xargs = Some(xargs);
    }

    // Thread KVCacheParameters → SamplingParams fields.
    if let Some(kv) = kv {
        // Thread kv_transfer_params through vllm_xargs, matching the HTTP route
        // convention.
        if let Some(kv_struct) = kv.kv_transfer_params.as_ref() {
            let kv_json = prost_struct_to_json(kv_struct);
            let map = sampling_params.vllm_xargs.get_or_insert_with(Default::default);
            if map.contains_key("kv_transfer_params") {
                return Err(Status::invalid_argument(
                    "kv_transfer_params cannot be supplied in both vllm_xargs_json and kv",
                ));
            }
            map.insert("kv_transfer_params".to_string(), kv_json);
        }
        if kv.bypass_prefix_cache {
            sampling_params.skip_reading_prefix_cache = Some(true);
        }
    }

    let decode_options = TextDecodeOptions {
        skip_special_tokens: true,
        include_stop_str_in_output: stopping.is_some_and(|s| s.include_stop_strings),
        stop_strings: stopping.map(|s| &s.stop_strings).filter(|ss| !ss.is_empty()).cloned(),
        min_tokens: stopping.map_or(0, |s| s.min_new_tokens),
    };

    let mm_features = convert_mm_features(&req.mm_features, &prompt)?;

    Ok(TextRequest {
        request_id,
        prompt,
        mm_features,
        sampling_params,
        decode_options,
        intermediate: stream,
        priority: req.priority,
        cache_salt: kv.map(|k| &k.cache_salt).filter(|s| !s.is_empty()).cloned(),
        add_special_tokens: true,
        data_parallel_rank: None,
        reasoning_parser_kwargs: None,
        lora_request: None,
        arrival_time: None,
    })
}

// ========================================================================================
// Response conversion
// ========================================================================================

/// Convert a `DecodedTextEvent::Start` into the prompt info portion of a gRPC
/// response.
pub fn to_prompt_info(
    prompt_token_ids: &[u32],
    prompt_logprobs: Option<&DecodedPromptLogprobs>,
    opts: &ResponseOpts,
) -> pb::PromptInfo {
    let token_ids = if opts.prompt_token_ids {
        prompt_token_ids.to_vec()
    } else {
        vec![]
    };

    let (logprobs, ranks, candidate_tokens) = match prompt_logprobs {
        Some(plp) if opts.prompt_logprobs => prompt_logprobs_to_proto(plp),
        _ => (vec![], vec![], vec![]),
    };

    pb::PromptInfo {
        num_prompt_tokens: prompt_token_ids.len() as u32,
        token_ids,
        logprobs,
        ranks,
        candidate_tokens,
    }
}

/// Convert a `DecodedTextEvent::TextDelta` into a gRPC `SequenceOutput`.
pub fn to_sequence_output(
    delta: &str,
    token_ids: &[u32],
    logprobs: Option<&DecodedLogprobs>,
    finished: Option<&Finished>,
    opts: &ResponseOpts,
) -> pb::SequenceOutput {
    let (lp_values, rank_values, candidates) = match logprobs {
        Some(lp) if opts.output_logprobs => output_logprobs_to_proto(lp),
        _ => (vec![], vec![], vec![]),
    };

    pb::SequenceOutput {
        index: 0, // TODO: multi-sequence (n > 1) not supported
        text: if opts.output_text {
            delta.to_string()
        } else {
            String::new()
        },
        num_tokens: token_ids.len() as u32,
        token_ids: if opts.output_token_ids {
            token_ids.to_vec()
        } else {
            vec![]
        },
        logprobs: lp_values,
        ranks: rank_values,
        candidate_tokens: candidates,
        finish_info: finished.map(|f| to_finish_info(f, token_ids)),
        routed_experts: finished
            .and_then(|finished| finished.routed_experts.as_ref().map(routed_experts_to_proto)),
    }
}

fn routed_experts_to_proto(tensor: &WireTensor) -> pb::RoutedExpertsTensor {
    let data = match &tensor.data {
        WireArrayData::RawView(data) => data.clone(),
        WireArrayData::AuxIndex(_) => {
            unreachable!("engine-core output arrays are resolved before response conversion")
        }
    };
    pb::RoutedExpertsTensor {
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.iter().map(|&dim| dim as u64).collect(),
        data,
    }
}

fn to_finish_info(finished: &Finished, token_ids: &[u32]) -> pb::FinishInfo {
    use pb::finish_info::FinishReason as PbFinishReason;

    let (finish_reason, stop_reason) = match &finished.finish_reason {
        FinishReason::Stop(reason) => {
            let sr = match reason {
                Some(StopReason::TokenId(id)) => {
                    Some(pb::finish_info::StopReason::StopTokenId(*id))
                }
                Some(StopReason::Text(s)) => {
                    Some(pb::finish_info::StopReason::StopString(s.clone()))
                }
                // EOS-driven stop: engine-core matched the primary EOS token id but did not
                // echo it back as a `stop_reason`. The matched token is, by construction, the
                // last token of the terminal output batch (see vllm's `check_stop` in
                // vllm/v1/core/sched/utils.py), so we recover it from there.
                None => token_ids.last().copied().map(pb::finish_info::StopReason::EosTokenId),
            };
            (PbFinishReason::Stop as i32, sr)
        }
        FinishReason::Length => (PbFinishReason::Length as i32, None),
        FinishReason::Abort | FinishReason::Error | FinishReason::Repetition(_) => {
            (PbFinishReason::Aborted as i32, None)
        }
    };

    pb::FinishInfo {
        num_output_tokens: finished.usage.output_token_count as u32,
        finish_reason,
        stop_reason,
        kv_transfer_params: finished.kv_transfer_params.as_ref().and_then(json_to_prost_struct),
    }
}

// ========================================================================================
// Logprobs helpers
// ========================================================================================

/// Convert output logprobs to the flat proto representation.
///
/// Returns (logprob_values, ranks, candidate_tokens) — all parallel arrays
/// indexed by position.
fn output_logprobs_to_proto(
    lp: &DecodedLogprobs,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    positions_to_proto(&lp.positions)
}

/// Convert prompt logprobs to the flat proto representation.
fn prompt_logprobs_to_proto(
    plp: &DecodedPromptLogprobs,
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    // The proto PromptInfo has flat parallel arrays covering all prompt positions.
    // DecodedPromptLogprobs has first_token separately + scored_positions for the
    // rest. The first prompt position has no scores, so we emit zeros for it.
    let (mut logprobs, mut ranks, mut candidates) = positions_to_proto(&plp.scored_positions);
    logprobs.insert(0, 0.0);
    ranks.insert(0, 0);
    candidates.insert(0, pb::CandidateTokenInfo { tokens: vec![] });
    (logprobs, ranks, candidates)
}

/// Shared helper: convert a slice of decoded position logprobs to flat proto
/// arrays.
fn positions_to_proto(
    positions: &[vllm_text::DecodedPositionLogprobs],
) -> (Vec<f32>, Vec<u32>, Vec<pb::CandidateTokenInfo>) {
    let mut logprobs = Vec::with_capacity(positions.len());
    let mut ranks = Vec::with_capacity(positions.len());
    let mut candidates = Vec::with_capacity(positions.len());

    for pos in positions {
        // First entry is the sampled/scored token.
        if let Some(first) = pos.entries.first() {
            logprobs.push(first.logprob);
            ranks.push(first.rank);

            // Engine-core can include the sampled token again in its top-k
            // alternatives. The gRPC schema carries that token separately in
            // the parallel token_ids/logprobs/ranks fields, so do not repeat it
            // in CandidateTokenInfo (whose contract is alternatives only).
            candidates.push(pb::CandidateTokenInfo {
                tokens: pos
                    .entries
                    .iter()
                    .skip(1)
                    .filter(|entry| entry.token_id != first.token_id)
                    .map(|entry| pb::candidate_token_info::TokenInfo {
                        id: entry.token_id,
                        logprob: entry.logprob,
                        rank: entry.rank,
                    })
                    .collect(),
            });
        } else {
            candidates.push(pb::CandidateTokenInfo { tokens: vec![] });
        }
    }

    (logprobs, ranks, candidates)
}

// ========================================================================================
// Options extracted from the request for response building
// ========================================================================================

/// Response-shaping options extracted from the proto `ResponseOptions`.
#[derive(Default)]
pub struct ResponseOpts {
    pub prompt_token_ids: bool,
    pub prompt_logprobs: bool,
    pub output_text: bool,
    pub output_token_ids: bool,
    pub output_logprobs: bool,
}

impl ResponseOpts {
    pub fn from_proto(r: Option<&pb::ResponseOptions>) -> Self {
        match r {
            Some(r) => Self {
                prompt_token_ids: r.prompt_token_ids,
                prompt_logprobs: r.prompt_logprobs,
                output_text: r.output_text.unwrap_or(true),
                output_token_ids: r.output_token_ids,
                output_logprobs: r.output_logprobs,
            },
            None => Self {
                output_text: true,
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmKwargValue,
    };
    use vllm_engine_core_client::protocol::output::StopReason;
    use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputConstraint;
    use vllm_engine_core_client::protocol::tensor::WireTensor;
    use vllm_text::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob, FinishReason, Finished,
        Prompt,
    };

    use super::pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};
    use super::{
        KvRole, ResponseOpts, mark_prefill_request, media_parts_from_request, mm_cache_identifier,
        pb, preflight_msgpack, to_finish_info, to_sequence_output, to_text_request,
        validate_disaggregated_request,
    };

    fn base_request() -> pb::GenerateRequest {
        pb::GenerateRequest {
            request_id: "req".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            ..Default::default()
        }
    }

    #[test]
    fn media_requires_a_source() {
        let error = media_parts_from_request(&[pb::MediaItem {
            modality: pb::Modality::Image as i32,
            ..Default::default()
        }])
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn media_rejects_unsupported_modalities() {
        let error = media_parts_from_request(&[pb::MediaItem {
            modality: pb::Modality::Audio as i32,
            source: Some(pb::media_item::Source::Url(
                "https://example.test/a.wav".into(),
            )),
            ..Default::default()
        }])
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Unimplemented);
    }

    #[test]
    fn decode_requires_kv_transfer_params() {
        let error = validate_disaggregated_request(&base_request(), KvRole::Decode).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn prefill_marks_remote_decode() {
        let mut text = to_text_request(base_request(), false, &["test-model".to_string()]).unwrap();
        mark_prefill_request(&mut text);
        assert_eq!(
            text.sampling_params.vllm_xargs.as_ref().unwrap()["kv_transfer_params"]["do_remote_decode"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn temperature_propagates_from_top_level_request_field() {
        let req = pb::GenerateRequest {
            temperature: Some(0.7),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.temperature, Some(0.7));
    }

    #[test]
    fn unset_temperature_defaults_to_greedy() {
        let text = to_text_request(base_request(), false, &["test-model".to_string()])
            .expect("convert ok");
        // The gRPC API defaults to greedy (0.0) when temperature is not specified.
        assert_eq!(text.sampling_params.temperature, Some(0.0));
    }

    #[test]
    fn absent_seed_is_none() {
        let req = pb::GenerateRequest {
            sampling: Some(pb::RandomSampling {
                seed: None,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.seed, None);
    }

    #[test]
    fn routed_experts_prompt_start_reaches_text_sampling_params() {
        let req = pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![1, 2],
            })),
            routed_experts_prompt_start: 1,
            ..base_request()
        };

        let text = to_text_request(req, false, &["test-model".to_string()]).unwrap();

        assert_eq!(text.sampling_params.routed_experts_prompt_start, 1);
    }

    #[test]
    fn routed_experts_prompt_start_rejects_unverifiable_or_out_of_range_values() {
        let text_prompt = pb::GenerateRequest {
            routed_experts_prompt_start: 1,
            ..base_request()
        };
        assert!(to_text_request(text_prompt, false, &["test-model".to_string()]).is_err());

        let token_prompt = pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![1, 2],
            })),
            routed_experts_prompt_start: 2,
            ..base_request()
        };
        assert!(to_text_request(token_prompt, false, &["test-model".to_string()]).is_err());
    }

    #[test]
    fn zero_seed_is_valid() {
        let req = pb::GenerateRequest {
            sampling: Some(pb::RandomSampling {
                seed: Some(0),
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.seed, Some(0));
    }

    #[test]
    fn bypass_prefix_cache_maps_to_skip_reading_prefix_cache() {
        let req = pb::GenerateRequest {
            kv: Some(pb::KvCacheParameters {
                bypass_prefix_cache: true,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.skip_reading_prefix_cache, Some(true));
    }

    #[test]
    fn bypass_prefix_cache_false_leaves_field_unset() {
        let req = pb::GenerateRequest {
            kv: Some(pb::KvCacheParameters {
                bypass_prefix_cache: false,
                ..Default::default()
            }),
            ..base_request()
        };
        let text = to_text_request(req, false, &["test-model".to_string()]).expect("convert ok");
        assert_eq!(text.sampling_params.skip_reading_prefix_cache, None);
        // Prompt conversion still succeeds and reaches the expected variant.
        assert!(matches!(text.prompt, Prompt::Text(s) if s == "hi"));
    }

    #[test]
    fn extended_sampling_fields_reach_text_request_losslessly() {
        let req = pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            sampling: Some(pb::RandomSampling {
                top_k: Some(-1),
                top_p: Some(0.0),
                min_p: Some(0.0),
                ..Default::default()
            }),
            decoding: Some(pb::DecodingParameters {
                presence_penalty: Some(0.0),
                frequency_penalty: Some(0.0),
                repetition_penalty: Some(0.0),
                logit_bias: [(7, -1.25)].into_iter().collect(),
                allowed_token_ids: vec![7, 8],
                bad_words: vec!["blocked".to_string()],
                structured_output: Some(pb::decoding_parameters::StructuredOutput::Regex(
                    "[a-z]+".to_string(),
                )),
                structured_output_disable_any_whitespace: true,
                structured_output_disable_additional_properties: true,
                structured_output_whitespace_pattern: Some("\\s*".to_string()),
            }),
            stopping: Some(pb::StoppingCriteria {
                thinking_token_budget: Some(64),
                ..Default::default()
            }),
            response: Some(pb::ResponseOptions {
                output_logprobs: true,
                output_candidates: Some(pb::CandidateTokens {
                    select: Some(pb::candidate_tokens::Select::TokenIds(pb::TokenIds {
                        ids: vec![7, 8],
                    })),
                }),
                ..Default::default()
            }),
            vllm_xargs_json: Some(br#"{"custom_integer":9007199254740993}"#.to_vec()),
            ..base_request()
        };

        let text = to_text_request(req, false, &["test-model".to_string()]).unwrap();

        assert_eq!(text.sampling_params.top_k, Some(0));
        assert_eq!(text.sampling_params.top_p, Some(0.0));
        assert_eq!(text.sampling_params.min_p, Some(0.0));
        assert_eq!(text.sampling_params.presence_penalty, Some(0.0));
        assert_eq!(text.sampling_params.frequency_penalty, Some(0.0));
        assert_eq!(text.sampling_params.repetition_penalty, Some(0.0));
        assert_eq!(text.sampling_params.thinking_token_budget, Some(64));
        assert_eq!(text.sampling_params.logit_bias.as_ref().unwrap()[&7], -1.25);
        assert_eq!(text.sampling_params.allowed_token_ids, Some(vec![7, 8]));
        assert_eq!(
            text.sampling_params.bad_words,
            Some(vec!["blocked".to_string()])
        );
        assert_eq!(text.sampling_params.logprobs, Some(2));
        assert_eq!(text.sampling_params.logprob_token_ids, Some(vec![7, 8]));
        let structured = text.sampling_params.structured_outputs.unwrap();
        assert_eq!(
            structured.constraint,
            StructuredOutputConstraint::Regex("[a-z]+".to_string())
        );
        assert!(structured.options.disable_any_whitespace);
        assert!(structured.options.disable_additional_properties);
        assert_eq!(
            structured.options.whitespace_pattern.as_deref(),
            Some("\\s*")
        );
        assert_eq!(
            text.sampling_params.vllm_xargs.as_ref().unwrap()["custom_integer"],
            serde_json::json!(9_007_199_254_740_993_u64)
        );
    }

    #[test]
    fn preprocessed_multimodal_features_reach_text_request() {
        let kwargs = BTreeMap::from([(
            "num_tiles".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Int(2)),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
            },
        )]);
        let kwargs_msgpack = rmp_serde::to_vec_named(&kwargs).unwrap();
        let identifier = mm_cache_identifier("image", &kwargs_msgpack);
        let req = pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![10, 99, 99, 20],
            })),
            mm_features: vec![pb::PreprocessedMultimodalFeature {
                modality: "image".to_string(),
                mm_hash: identifier.clone(),
                position: Some(pb::MultimodalPlaceholder {
                    offset: 1,
                    length: 2,
                    is_embed: vec![true, false],
                }),
                cache_identifier: identifier.clone(),
                kwargs_msgpack: Some(kwargs_msgpack),
            }],
            ..base_request()
        };

        let text = to_text_request(req, false, &["test-model".to_string()]).unwrap();
        let features = text.mm_features.unwrap();

        assert_eq!(features.len(), 1);
        assert_eq!(features[0].modality, "image");
        assert!(features[0].identifier.starts_with("grpc-mm:"));
        assert_eq!(features[0].mm_hash.as_deref(), Some(identifier.as_str()));
        assert_eq!(features[0].data.as_ref(), Some(&kwargs));
        assert_eq!(features[0].mm_position.offset, 1);
        assert_eq!(features[0].mm_position.length, 2);
        assert!(features[0].mm_position.is_embed.is_some());
    }

    #[test]
    fn multimodal_lora_is_rejected_until_tower_cache_semantics_are_advertised() {
        for request in [
            pb::GenerateRequest {
                lora_name: "adapter-a".to_string(),
                media: vec![pb::MediaItem::default()],
                ..base_request()
            },
            pb::GenerateRequest {
                lora_name: "adapter-a".to_string(),
                mm_features: vec![pb::PreprocessedMultimodalFeature::default()],
                ..base_request()
            },
        ] {
            let error = to_text_request(request, false, &["test-model".to_string()]).unwrap_err();
            assert_eq!(error.code(), tonic::Code::InvalidArgument);
            assert!(error.message().contains("tower-LoRA"));
        }
    }

    #[test]
    fn preprocessed_multimodal_cache_hit_without_data_is_rejected() {
        let req = pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![10, 99, 20],
            })),
            mm_features: vec![pb::PreprocessedMultimodalFeature {
                modality: "image".to_string(),
                mm_hash: "image-hash".to_string(),
                position: Some(pb::MultimodalPlaceholder {
                    offset: 1,
                    length: 1,
                    is_embed: Vec::new(),
                }),
                kwargs_msgpack: None,
                cache_identifier: String::new(),
            }],
            ..base_request()
        };

        let error = to_text_request(req, false, &["test-model".to_string()]).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn multimodal_cache_identity_is_data_bound_without_language_lora_scoping() {
        fn request(value: i64) -> pb::GenerateRequest {
            let kwargs = BTreeMap::from([(
                "value".to_string(),
                MmFieldElem {
                    data: Some(MmKwargValue::Int(value)),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
                },
            )]);
            let kwargs_msgpack = rmp_serde::to_vec_named(&kwargs).unwrap();
            let identifier = mm_cache_identifier("image", &kwargs_msgpack);
            pb::GenerateRequest {
                prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                    ids: vec![10, 99, 20],
                })),
                mm_features: vec![pb::PreprocessedMultimodalFeature {
                    modality: "image".to_string(),
                    mm_hash: identifier.clone(),
                    position: Some(pb::MultimodalPlaceholder {
                        offset: 1,
                        length: 1,
                        is_embed: Vec::new(),
                    }),
                    cache_identifier: identifier,
                    kwargs_msgpack: Some(kwargs_msgpack),
                }],
                ..base_request()
            }
        }

        let first = to_text_request(request(1), false, &["test-model".to_string()]).unwrap();
        let second = to_text_request(request(2), false, &["test-model".to_string()]).unwrap();
        assert_ne!(
            first.mm_features.as_ref().unwrap()[0].identifier,
            second.mm_features.as_ref().unwrap()[0].identifier
        );
        assert!(first.mm_features.as_ref().unwrap()[0].identifier.starts_with("grpc-mm:"));
    }

    #[test]
    fn multimodal_cache_identity_has_a_cross_language_fixed_vector() {
        assert_eq!(
            mm_cache_identifier("image", b"abc"),
            "grpc-mm:c2f2df4bb94911d850921fa6d577ee0713ad5884c276f85330a8e50137f6a59d"
        );
    }

    #[test]
    fn multimodal_cache_identity_mismatch_is_rejected() {
        let kwargs = BTreeMap::from([(
            "value".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Int(1)),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
            },
        )]);
        let kwargs_msgpack = rmp_serde::to_vec_named(&kwargs).unwrap();
        let req = pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![10, 99, 20],
            })),
            mm_features: vec![pb::PreprocessedMultimodalFeature {
                modality: "image".to_string(),
                mm_hash: "image-hash".to_string(),
                position: Some(pb::MultimodalPlaceholder {
                    offset: 1,
                    length: 1,
                    is_embed: Vec::new(),
                }),
                kwargs_msgpack: Some(kwargs_msgpack.clone()),
                cache_identifier: mm_cache_identifier("image", &kwargs_msgpack),
            }],
            ..base_request()
        };

        let error = to_text_request(req, false, &["test-model".to_string()]).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn vllm_xargs_reject_reserved_kv_control() {
        let req = pb::GenerateRequest {
            vllm_xargs_json: Some(br#"{"kv_transfer_params":{}}"#.to_vec()),
            ..base_request()
        };
        let error = to_text_request(req, false, &["test-model".to_string()]).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn msgpack_preflight_rejects_trailing_and_amplified_values() {
        let mut nodes = 0;
        assert!(preflight_msgpack(&[0xc0, 0xc0], &mut nodes).is_err());

        let mut nodes = 0;
        let oversized_array = [0xdd, 0x00, 0x01, 0x00, 0x01];
        assert!(preflight_msgpack(&oversized_array, &mut nodes).is_err());
    }

    #[test]
    fn malformed_preprocessed_multimodal_feature_is_rejected() {
        let req = pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![1, 2],
            })),
            mm_features: vec![pb::PreprocessedMultimodalFeature {
                modality: "image".to_string(),
                mm_hash: "image-hash".to_string(),
                position: Some(pb::MultimodalPlaceholder {
                    offset: 1,
                    length: 2,
                    is_embed: Vec::new(),
                }),
                kwargs_msgpack: Some(vec![0xc1]),
                cache_identifier: "invalid".to_string(),
            }],
            ..base_request()
        };

        let error = to_text_request(req, false, &["test-model".to_string()]).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    fn finished(reason: FinishReason) -> Finished {
        Finished {
            usage: vllm_llm::TokenUsage {
                prompt_token_count: 0,
                output_token_count: 0,
                cached_token_count: 0,
            },
            finish_reason: reason,
            kv_transfer_params: None,
            routed_experts: None,
        }
    }

    #[test]
    fn eos_stop_reports_last_output_token_as_eos_id() {
        let fin = finished(FinishReason::Stop(None));
        let token_ids = [1_u32, 2, 3, 151643];

        let info = to_finish_info(&fin, &token_ids);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(info.stop_reason, Some(PbStopReason::EosTokenId(151643)));
    }

    #[test]
    fn eos_stop_with_empty_token_ids_leaves_stop_reason_unset() {
        let fin = finished(FinishReason::Stop(None));

        let info = to_finish_info(&fin, &[]);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(info.stop_reason, None);
    }

    #[test]
    fn explicit_stop_token_id_is_preserved() {
        let fin = finished(FinishReason::Stop(Some(StopReason::TokenId(42))));
        // Terminal token list should be ignored when an explicit stop reason is
        // present.
        let info = to_finish_info(&fin, &[7, 42]);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(info.stop_reason, Some(PbStopReason::StopTokenId(42)));
    }

    #[test]
    fn explicit_stop_string_is_preserved() {
        let fin = finished(FinishReason::Stop(Some(StopReason::Text("</stop>".into()))));

        let info = to_finish_info(&fin, &[1, 2, 3]);

        assert_eq!(info.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(
            info.stop_reason,
            Some(PbStopReason::StopString("</stop>".into()))
        );
    }

    #[test]
    fn length_finish_has_no_stop_reason() {
        let fin = finished(FinishReason::Length);

        let info = to_finish_info(&fin, &[1, 2, 3]);

        assert_eq!(info.finish_reason, PbFinishReason::Length as i32);
        assert_eq!(info.stop_reason, None);
    }

    #[test]
    fn abort_finish_is_mapped_to_aborted() {
        let fin = finished(FinishReason::Abort);

        let info = to_finish_info(&fin, &[]);

        assert_eq!(info.finish_reason, PbFinishReason::Aborted as i32);
        assert_eq!(info.stop_reason, None);
    }

    #[test]
    fn to_sequence_output_threads_token_ids_into_eos_id() {
        let fin = finished(FinishReason::Stop(None));
        let opts = ResponseOpts {
            output_text: true,
            output_token_ids: true,
            ..Default::default()
        };

        let out = to_sequence_output("hello", &[10, 20, 30], None, Some(&fin), &opts);

        let finish = out.finish_info.expect("finish_info should be present");
        assert_eq!(finish.finish_reason, PbFinishReason::Stop as i32);
        assert_eq!(finish.stop_reason, Some(PbStopReason::EosTokenId(30)));
    }

    #[test]
    fn to_sequence_output_emits_typed_routed_experts() {
        let mut fin = finished(FinishReason::Length);
        fin.routed_experts = Some(WireTensor::from_raw("|u1", vec![1, 2, 2], vec![1, 2, 3, 4]));

        let output = to_sequence_output("", &[], None, Some(&fin), &ResponseOpts::default());
        let routed = output.routed_experts.unwrap();
        assert_eq!(routed.dtype, "|u1");
        assert_eq!(routed.shape, vec![1, 2, 2]);
        assert_eq!(routed.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn output_logprobs_do_not_repeat_the_sampled_token_as_a_candidate() {
        let logprobs = DecodedLogprobs {
            positions: vec![DecodedPositionLogprobs {
                entries: vec![
                    DecodedTokenLogprob {
                        token_id: 42,
                        token: "sampled".into(),
                        logprob: -0.1,
                        rank: 1,
                    },
                    DecodedTokenLogprob {
                        token_id: 42,
                        token: "sampled".into(),
                        logprob: -0.1,
                        rank: 1,
                    },
                    DecodedTokenLogprob {
                        token_id: 7,
                        token: "alternate".into(),
                        logprob: -1.2,
                        rank: 2,
                    },
                ],
            }],
        };
        let opts = ResponseOpts {
            output_token_ids: true,
            output_logprobs: true,
            ..Default::default()
        };

        let output = to_sequence_output("sampled", &[42], Some(&logprobs), None, &opts);

        assert_eq!(output.logprobs, vec![-0.1]);
        assert_eq!(output.ranks, vec![1]);
        assert_eq!(output.candidate_tokens.len(), 1);
        assert_eq!(
            output.candidate_tokens[0]
                .tokens
                .iter()
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>(),
            vec![7]
        );
    }
}
