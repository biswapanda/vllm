use tonic::Status;
use uuid::Uuid;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputBackend, StructuredOutputsParams,
};
use vllm_text::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTextEvent,
    FinishReason, Finished, TextRequest,
};

use super::super::pb as private;
use super::pb;
use crate::grpc::struct_json::json_to_prost_struct;

pub(super) struct RequestExtensions {
    pub(super) logprobs: Option<i32>,
    pub(super) prompt_logprobs: Option<i32>,
    pub(super) logprob_token_ids: Option<Vec<u32>>,
    pub(super) structured_outputs: Option<StructuredOutputsParams>,
    pub(super) include_stop_in_output: bool,
    pub(super) cache_salt: Option<String>,
    pub(super) bypass_prefix_cache: Option<bool>,
    pub(super) skip_special_tokens: Option<bool>,
}

type LogprobOptions = (Option<i32>, Option<i32>, Option<Vec<u32>>);

pub(super) fn to_private_request(
    request: pb::GenerateRequest,
) -> Result<(private::GenerateRequest, RequestExtensions), Status> {
    let pb::GenerateRequest {
        request_id,
        model,
        input,
        sampling,
        stopping,
        response,
        kv,
        guided,
        media,
        lora_name,
        priority,
        metadata: _,
    } = request;

    if sampling.as_ref().and_then(|params| params.num_sequences).unwrap_or(1) != 1 {
        return Err(Status::invalid_argument(
            "the vLLM Rust OpenEngine server currently supports num_sequences=1",
        ));
    }

    let seed = sampling
        .as_ref()
        .and_then(|params| params.seed)
        .map(|seed| {
            i64::try_from(seed)
                .map_err(|_| Status::invalid_argument("sampling.seed exceeds i64::MAX"))
        })
        .transpose()?;
    let stopping = stopping.unwrap_or_default();
    let needs_private_sampling = sampling.is_some()
        || stopping.max_tokens.is_some()
        || stopping.min_tokens.is_some()
        || stopping.ignore_eos.is_some();
    let private_sampling = needs_private_sampling.then(|| {
        let sampling = sampling.unwrap_or_default();
        private::SamplingParams {
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k,
            frequency_penalty: sampling.frequency_penalty,
            presence_penalty: sampling.presence_penalty,
            max_tokens: stopping.max_tokens,
            seed,
            ignore_eos: stopping.ignore_eos.unwrap_or(false),
            min_p: sampling.min_p,
            repetition_penalty: sampling.repetition_penalty,
            min_tokens: stopping.min_tokens,
        }
    });

    let input = input.map(|input| match input {
        pb::generate_request::Input::Prompt(prompt) => {
            private::generate_request::Input::Prompt(prompt)
        }
        pb::generate_request::Input::TokenIds(ids) => {
            private::generate_request::Input::TokenIds(private::TokenIds { ids: ids.ids })
        }
    });
    let stop = stopping
        .conditions
        .into_iter()
        .map(|condition| private::StopCondition {
            condition: condition.condition.map(|condition| match condition {
                pb::stop_condition::Condition::StopText(text) => {
                    private::stop_condition::Condition::StopText(text)
                }
                pb::stop_condition::Condition::StopTokenId(id) => {
                    private::stop_condition::Condition::StopTokenId(id)
                }
            }),
        })
        .collect();
    let media = media
        .into_iter()
        .map(|item| private::MediaItem {
            modality: item.modality,
            source: item.source.map(|source| match source {
                pb::media_item::Source::Url(url) => private::media_item::Source::Url(url),
                pb::media_item::Source::DataUri(data_uri) => {
                    private::media_item::Source::DataUri(data_uri)
                }
                pb::media_item::Source::RawBytes(bytes) => {
                    private::media_item::Source::RawBytes(bytes)
                }
            }),
            mime_type: item.mime_type,
            uuid: item.uuid,
        })
        .collect();

    let (kv_session, data_parallel_rank, cache_salt, bypass_prefix_cache) =
        kv.map_or((None, None, None, None), |kv| {
            let session = kv.session.map(|session| private::KvSessionRef {
                session_id: session.session_id,
                transfer_backend: session.transfer_backend,
                dp_rank: session.dp_rank,
                attributes_struct: session.attributes_struct,
            });
            (
                session,
                kv.data_parallel_rank,
                kv.cache_salt,
                kv.bypass_prefix_cache,
            )
        });

    let (logprobs, prompt_logprobs, logprob_token_ids) = response_options(response.as_ref())?;
    let skip_special_tokens = response.as_ref().and_then(|options| options.skip_special_tokens);
    let structured_outputs = guided.map(guided_to_structured_output).transpose()?;
    let extensions = RequestExtensions {
        logprobs,
        prompt_logprobs,
        logprob_token_ids,
        structured_outputs,
        include_stop_in_output: stopping.include_stop_in_output.unwrap_or(false),
        cache_salt,
        bypass_prefix_cache,
        skip_special_tokens,
    };

    Ok((
        private::GenerateRequest {
            request_id: if request_id.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                request_id
            },
            model,
            input,
            sampling: private_sampling,
            stop,
            stream: true,
            media,
            lora_name,
            kv_session,
            data_parallel_rank,
            priority,
        },
        extensions,
    ))
}

pub(super) fn apply_extensions(request: &mut TextRequest, extensions: RequestExtensions) {
    request.sampling_params.logprobs = extensions.logprobs;
    request.sampling_params.prompt_logprobs = extensions.prompt_logprobs;
    request.sampling_params.logprob_token_ids = extensions.logprob_token_ids;
    request.sampling_params.structured_outputs = extensions.structured_outputs;
    request.sampling_params.skip_reading_prefix_cache = extensions.bypass_prefix_cache;
    request.decode_options.include_stop_str_in_output = extensions.include_stop_in_output;
    if let Some(skip_special_tokens) = extensions.skip_special_tokens {
        request.decode_options.skip_special_tokens = skip_special_tokens;
    }
    request.cache_salt = extensions.cache_salt;
}

fn response_options(response: Option<&pb::ResponseOptions>) -> Result<LogprobOptions, Status> {
    let Some(response) = response else {
        return Ok((None, None, None));
    };
    if response.prompt_logprob_start.is_some_and(|start| start != 0) {
        return Err(Status::invalid_argument(
            "prompt_logprob_start values other than zero are not supported",
        ));
    }

    let (logprobs, token_ids) = candidate_selection(
        response.return_output_logprobs,
        response.output_candidates.as_ref(),
        true,
    )?;
    let (prompt_logprobs, prompt_token_ids) = candidate_selection(
        response.return_prompt_logprobs,
        response.prompt_candidates.as_ref(),
        false,
    )?;
    debug_assert!(prompt_token_ids.is_none());
    Ok((logprobs, prompt_logprobs, token_ids))
}

fn candidate_selection(
    enabled: Option<bool>,
    candidates: Option<&pb::CandidateTokenSelection>,
    allow_token_ids: bool,
) -> Result<(Option<i32>, Option<Vec<u32>>), Status> {
    if enabled == Some(false) {
        if candidates.and_then(|value| value.selection.as_ref()).is_some() {
            return Err(Status::invalid_argument(
                "candidate selection cannot be set when logprobs are disabled",
            ));
        }
        return Ok((None, None));
    }
    if enabled.is_none() && candidates.and_then(|value| value.selection.as_ref()).is_none() {
        return Ok((None, None));
    }

    match candidates.and_then(|value| value.selection.as_ref()) {
        None => Ok((Some(0), None)),
        Some(pb::candidate_token_selection::Selection::TopN(top_n)) => Ok((
            Some(
                i32::try_from(*top_n)
                    .map_err(|_| Status::invalid_argument("logprob top_n exceeds i32::MAX"))?,
            ),
            None,
        )),
        Some(pb::candidate_token_selection::Selection::All(_)) => Ok((Some(-1), None)),
        Some(pb::candidate_token_selection::Selection::TokenIds(ids)) if allow_token_ids => {
            Ok((None, Some(ids.ids.clone())))
        }
        Some(pb::candidate_token_selection::Selection::TokenIds(_)) => Err(
            Status::invalid_argument("prompt token-id candidate selection is not supported"),
        ),
    }
}

fn guided_to_structured_output(
    guided: pb::GuidedDecoding,
) -> Result<StructuredOutputsParams, Status> {
    let mut params = match guided
        .guide
        .ok_or_else(|| Status::invalid_argument("guided decoding must set exactly one guide"))?
    {
        pb::guided_decoding::Guide::JsonSchema(schema) => {
            let schema = serde_json::from_str(&schema).map_err(|error| {
                Status::invalid_argument(format!("guided JSON schema is invalid: {error}"))
            })?;
            StructuredOutputsParams::json(schema)
        }
        pb::guided_decoding::Guide::Regex(regex) => StructuredOutputsParams::regex(regex),
        pb::guided_decoding::Guide::EbnfGrammar(grammar) => {
            StructuredOutputsParams::grammar(grammar)
        }
        pb::guided_decoding::Guide::StructuralTag(tag) => {
            StructuredOutputsParams::structural_tag(tag)
        }
        pb::guided_decoding::Guide::Choice(choice) => {
            StructuredOutputsParams::choice(choice.choices)
        }
        pb::guided_decoding::Guide::JsonObject(_) => StructuredOutputsParams::json_object(),
    };
    params.backend = match guided.backend.as_str() {
        "" | "guidance" | "llguidance" => StructuredOutputBackend::Guidance,
        "xgrammar" => StructuredOutputBackend::Xgrammar,
        "outlines" => StructuredOutputBackend::Outlines,
        "lm-format-enforcer" => StructuredOutputBackend::LmFormatEnforcer,
        backend => {
            return Err(Status::invalid_argument(format!(
                "unsupported guided decoding backend `{backend}`"
            )));
        }
    };
    Ok(params)
}

pub(super) fn event_to_responses(
    event: DecodedTextEvent,
    request_id: &str,
    role: private::EngineRole,
    kv_connector: Option<&str>,
    handoff_dp_rank: u32,
) -> Vec<pb::GenerateResponse> {
    match event {
        DecodedTextEvent::Start {
            prompt_token_ids,
            prompt_logprobs,
        } => prompt_logprobs.map_or_else(Vec::new, |logprobs| {
            vec![pb::GenerateResponse {
                request_id: request_id.to_string(),
                event: Some(pb::generate_response::Event::Prompt(prompt_output(
                    &prompt_token_ids,
                    logprobs,
                ))),
                usage: None,
            }]
        }),
        DecodedTextEvent::TextDelta {
            delta,
            token_ids,
            logprobs,
            finished,
        } => {
            let mut responses = Vec::new();
            if !token_ids.is_empty() || !delta.is_empty() {
                let mut tokens = token_output(&token_ids, logprobs.as_ref());
                if tokens.len() == 1 && tokens[0].token.is_empty() && !delta.is_empty() {
                    tokens[0].token = delta.clone();
                }
                responses.push(pb::GenerateResponse {
                    request_id: request_id.to_string(),
                    event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                        output_index: Some(0),
                        tokens,
                        text: delta,
                    })),
                    usage: None,
                });
            }
            if let Some(finished) = finished {
                responses.push(terminal_response(
                    &finished,
                    request_id,
                    role,
                    kv_connector,
                    handoff_dp_rank,
                ));
            }
            responses
        }
    }
}

fn prompt_output(prompt_token_ids: &[u32], logprobs: DecodedPromptLogprobs) -> pb::PromptOutput {
    let mut tokens = Vec::with_capacity(prompt_token_ids.len());
    tokens.push(pb::TokenInfo {
        token_id: logprobs.first_token_id,
        token: logprobs.first_token,
        logprob: None,
        rank: None,
        candidates: Vec::new(),
    });
    tokens.extend(
        prompt_token_ids
            .iter()
            .copied()
            .skip(1)
            .zip(logprobs.scored_positions)
            .map(|(token_id, position)| token_info(token_id, position)),
    );
    pb::PromptOutput { tokens }
}

fn token_output(token_ids: &[u32], logprobs: Option<&DecodedLogprobs>) -> Vec<pb::TokenInfo> {
    token_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, token_id)| {
            logprobs.and_then(|value| value.positions.get(index).cloned()).map_or_else(
                || pb::TokenInfo {
                    token_id,
                    ..Default::default()
                },
                |position| token_info(token_id, position),
            )
        })
        .collect()
}

fn token_info(token_id: u32, position: DecodedPositionLogprobs) -> pb::TokenInfo {
    let selected = position.entries.iter().find(|entry| entry.token_id == token_id);
    pb::TokenInfo {
        token_id,
        token: selected.map(|entry| entry.token.clone()).unwrap_or_default(),
        logprob: selected.map(|entry| f64::from(entry.logprob)),
        rank: selected.map(|entry| entry.rank),
        candidates: position
            .entries
            .into_iter()
            .map(|entry| pb::LogProb {
                token_id: entry.token_id,
                logprob: f64::from(entry.logprob),
                token: entry.token,
                rank: Some(entry.rank),
            })
            .collect(),
    }
}

fn terminal_response(
    finished: &Finished,
    request_id: &str,
    role: private::EngineRole,
    kv_connector: Option<&str>,
    handoff_dp_rank: u32,
) -> pb::GenerateResponse {
    let usage = Some(to_usage(finished));
    if role == private::EngineRole::Prefill {
        if matches!(
            finished.finish_reason,
            FinishReason::Abort | FinishReason::Error
        ) {
            return error_response_with_usage(
                request_id,
                pb::ErrorCode::KvTransferFailed,
                "prefill did not complete successfully",
                usage,
            );
        }
        let Some(params) = finished.kv_transfer_params.as_ref() else {
            return error_response_with_usage(
                request_id,
                pb::ErrorCode::KvTransferFailed,
                "prefill completed without KV transfer metadata",
                usage,
            );
        };
        let attributes_struct = json_to_prost_struct(params);
        if attributes_struct.as_ref().is_none_or(|value| value.fields.is_empty()) {
            return error_response_with_usage(
                request_id,
                pb::ErrorCode::KvTransferFailed,
                "prefill returned invalid KV transfer metadata",
                usage,
            );
        }
        return pb::GenerateResponse {
            request_id: request_id.to_string(),
            event: Some(pb::generate_response::Event::PrefillReady(
                pb::PrefillReady {
                    kv_session: Some(pb::KvSessionRef {
                        session_id: request_id.to_string(),
                        transfer_backend: kv_connector.unwrap_or_default().to_string(),
                        endpoints: Vec::new(),
                        dp_rank: handoff_dp_rank,
                        attributes_struct,
                    }),
                },
            )),
            usage,
        };
    }

    if finished.finish_reason == FinishReason::Error {
        return error_response_with_usage(
            request_id,
            pb::ErrorCode::Internal,
            "engine reported an internal generation error",
            usage,
        );
    }
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: finish_reason(&finished.finish_reason) as i32,
                message: String::new(),
                stop_match: stop_match(&finished.finish_reason),
            },
        )),
        usage,
    }
}

pub(super) fn error_response(request_id: &str, message: String) -> pb::GenerateResponse {
    error_response_with_usage(request_id, pb::ErrorCode::Internal, &message, None)
}

fn error_response_with_usage(
    request_id: &str,
    code: pb::ErrorCode,
    message: &str,
    usage: Option<pb::Usage>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Error(pb::EngineError {
            code: code as i32,
            message: message.to_string(),
            retryable: code == pb::ErrorCode::Internal,
            retry_after_ms: None,
            details: None,
        })),
        usage,
    }
}

fn to_usage(finished: &Finished) -> pb::Usage {
    let prompt = finished.usage.prompt_token_count.min(u32::MAX as usize) as u32;
    let completion = finished.usage.output_token_count.min(u32::MAX as usize) as u32;
    pb::Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt.saturating_add(completion),
        cached_prompt_tokens: Some(finished.usage.cached_token_count.min(u32::MAX as usize) as u32),
        reasoning_tokens: None,
    }
}

fn finish_reason(reason: &FinishReason) -> pb::FinishReason {
    match reason {
        FinishReason::Stop(_) | FinishReason::Repetition(_) => pb::FinishReason::Stop,
        FinishReason::Length => pb::FinishReason::Length,
        FinishReason::Abort => pb::FinishReason::Cancelled,
        FinishReason::Error => pb::FinishReason::Unspecified,
    }
}

fn stop_match(reason: &FinishReason) -> Option<pb::StopMatch> {
    let stop = match reason {
        FinishReason::Stop(stop) | FinishReason::Repetition(stop) => stop.as_ref(),
        _ => None,
    }?;
    let r#match = match stop {
        StopReason::TokenId(id) => pb::stop_match::Match::StopTokenId(*id),
        StopReason::Text(text) => pb::stop_match::Match::StopText(text.clone()),
    };
    Some(pb::StopMatch {
        r#match: Some(r#match),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_text::DecodedTokenLogprob;

    use super::*;

    #[test]
    fn canonical_request_maps_nested_sampling_stopping_response_and_kv() {
        let request = pb::GenerateRequest {
            request_id: "req-canonical".to_string(),
            model: "test-model".to_string(),
            input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            sampling: Some(pb::SamplingParams {
                temperature: Some(0.7),
                top_p: Some(0.9),
                seed: Some(17),
                num_sequences: Some(1),
                ..Default::default()
            }),
            stopping: Some(pb::StoppingOptions {
                max_tokens: Some(32),
                min_tokens: Some(2),
                conditions: vec![pb::StopCondition {
                    condition: Some(pb::stop_condition::Condition::StopText("done".to_string())),
                }],
                ignore_eos: Some(true),
                include_stop_in_output: Some(true),
            }),
            response: Some(pb::ResponseOptions {
                return_output_logprobs: Some(true),
                output_candidates: Some(pb::CandidateTokenSelection {
                    selection: Some(pb::candidate_token_selection::Selection::TopN(3)),
                }),
                return_prompt_logprobs: Some(true),
                prompt_candidates: Some(pb::CandidateTokenSelection {
                    selection: Some(pb::candidate_token_selection::Selection::TopN(2)),
                }),
                prompt_logprob_start: Some(0),
                skip_special_tokens: Some(false),
            }),
            kv: Some(pb::KvOptions {
                session: None,
                data_parallel_rank: Some(3),
                bypass_prefix_cache: Some(true),
                cache_salt: Some("tenant-a".to_string()),
            }),
            guided: Some(pb::GuidedDecoding {
                guide: Some(pb::guided_decoding::Guide::Regex("[0-9]+".to_string())),
                backend: "xgrammar".to_string(),
            }),
            priority: Some(7),
            ..Default::default()
        };

        let (private, extensions) = to_private_request(request).unwrap();
        let sampling = private.sampling.as_ref().unwrap();
        assert_eq!(sampling.max_tokens, Some(32));
        assert_eq!(sampling.min_tokens, Some(2));
        assert!(sampling.ignore_eos);
        assert_eq!(private.stop.len(), 1);
        assert_eq!(private.data_parallel_rank, Some(3));
        assert_eq!(private.priority, Some(7));

        let mut text_request = TextRequest::for_test();
        apply_extensions(&mut text_request, extensions);
        assert_eq!(text_request.sampling_params.logprobs, Some(3));
        assert_eq!(text_request.sampling_params.prompt_logprobs, Some(2));
        assert_eq!(
            text_request.sampling_params.skip_reading_prefix_cache,
            Some(true)
        );
        assert!(text_request.sampling_params.structured_outputs.is_some());
        assert!(text_request.decode_options.include_stop_str_in_output);
        assert!(!text_request.decode_options.skip_special_tokens);
        assert_eq!(text_request.cache_salt.as_deref(), Some("tenant-a"));
    }

    #[test]
    fn canonical_stopping_options_apply_without_explicit_sampling() {
        let request = pb::GenerateRequest {
            stopping: Some(pb::StoppingOptions {
                max_tokens: Some(9),
                min_tokens: Some(2),
                ignore_eos: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };

        let (private, _) = to_private_request(request).unwrap();
        let sampling =
            private.sampling.expect("stopping fields must create private sampling params");
        assert_eq!(sampling.max_tokens, Some(9));
        assert_eq!(sampling.min_tokens, Some(2));
        assert!(sampling.ignore_eos);
    }

    #[test]
    fn canonical_token_event_preserves_selected_and_candidate_logprobs() {
        let event = DecodedTextEvent::TextDelta {
            delta: "x".to_string(),
            token_ids: vec![120],
            logprobs: Some(DecodedLogprobs {
                positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token_id: 120,
                            token: "x".to_string(),
                            logprob: -0.25,
                            rank: 1,
                        },
                        DecodedTokenLogprob {
                            token_id: 121,
                            token: "y".to_string(),
                            logprob: -1.5,
                            rank: 2,
                        },
                    ],
                }],
            }),
            finished: None,
        };

        let responses = event_to_responses(
            event,
            "req-logprobs",
            private::EngineRole::Aggregated,
            None,
            0,
        );
        let Some(pb::generate_response::Event::Token(output)) = responses[0].event.as_ref() else {
            panic!("expected token output");
        };
        assert_eq!(output.output_index, Some(0));
        assert_eq!(output.tokens.len(), 1);
        assert_eq!(output.tokens[0].token_id, 120);
        assert_eq!(output.tokens[0].logprob, Some(-0.25));
        assert_eq!(output.tokens[0].candidates.len(), 2);
    }

    #[test]
    fn canonical_single_token_without_logprobs_uses_decoded_delta() {
        let event = DecodedTextEvent::TextDelta {
            delta: "x".to_string(),
            token_ids: vec![120],
            logprobs: None,
            finished: None,
        };
        let responses = event_to_responses(
            event,
            "req-no-logprobs",
            private::EngineRole::Aggregated,
            None,
            0,
        );
        let Some(pb::generate_response::Event::Token(output)) = responses[0].event.as_ref() else {
            panic!("expected token output");
        };
        assert_eq!(output.tokens[0].token, "x");
    }

    #[test]
    fn canonical_prompt_event_marks_first_token_unscored() {
        let event = DecodedTextEvent::Start {
            prompt_token_ids: Arc::from([112, 120]),
            prompt_logprobs: Some(DecodedPromptLogprobs {
                first_token_id: 112,
                first_token: "p".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![DecodedTokenLogprob {
                        token_id: 120,
                        token: "x".to_string(),
                        logprob: -0.4,
                        rank: 1,
                    }],
                }],
            }),
        };

        let responses = event_to_responses(
            event,
            "req-prompt",
            private::EngineRole::Aggregated,
            None,
            0,
        );
        let Some(pb::generate_response::Event::Prompt(output)) = responses[0].event.as_ref() else {
            panic!("expected prompt output");
        };
        assert_eq!(output.tokens.len(), 2);
        assert_eq!(output.tokens[0].logprob, None);
        assert!(output.tokens[1].logprob.is_some_and(|value| (value - -0.4).abs() < 1e-6));
    }
}
