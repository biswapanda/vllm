use vllm_llm::TokenUsage;
use vllm_text::{DecodedTextEvent, FinishReason, Finished};

use super::super::pb;
use super::*;

fn base_request() -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: "req".to_string(),
        model: "test-model".to_string(),
        input: Some(pb::generate_request::Input::Prompt("hi".to_string())),
        ..Default::default()
    }
}

fn finished(reason: FinishReason) -> Finished {
    Finished {
        usage: TokenUsage {
            prompt_token_count: 3,
            output_token_count: 2,
            cached_token_count: 0,
        },
        finish_reason: reason,
        kv_transfer_params: None,
    }
}

#[test]
fn request_conversion_preserves_sampling_stop_and_priority() {
    let request = pb::GenerateRequest {
        sampling: Some(pb::SamplingParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(50),
            seed: Some(-7),
            min_tokens: Some(2),
            ..Default::default()
        }),
        stop: vec![
            pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopText("END".to_string())),
            },
            pb::StopCondition {
                condition: Some(pb::stop_condition::Condition::StopTokenId(7)),
            },
        ],
        priority: Some(-3),
        ..base_request()
    };
    let text = to_text_request(request, &["test-model".to_string()]).unwrap();
    assert_eq!(text.sampling_params.temperature, Some(0.7));
    assert_eq!(text.sampling_params.seed, Some(-7));
    assert_eq!(text.sampling_params.stop_token_ids, Some(vec![7]));
    assert_eq!(
        text.decode_options.stop_strings,
        Some(vec!["END".to_string()])
    );
    assert_eq!(text.decode_options.min_tokens, 2);
    assert_eq!(text.priority, -3);
    assert_eq!(text.data_parallel_rank, None);
    assert_eq!(text.arrival_time, None);
}

#[test]
fn request_conversion_rejects_missing_input_and_wrong_model() {
    let missing = pb::GenerateRequest {
        input: None,
        ..base_request()
    };
    assert_eq!(
        to_text_request(missing, &["test-model".to_string()]).unwrap_err().code(),
        tonic::Code::InvalidArgument
    );
    let wrong = pb::GenerateRequest {
        model: "other".to_string(),
        ..base_request()
    };
    assert_eq!(
        to_text_request(wrong, &["test-model".to_string()]).unwrap_err().code(),
        tonic::Code::NotFound
    );
}

#[test]
fn request_conversion_rejects_fields_owned_by_later_stack_layers() {
    for request in [
        pb::GenerateRequest {
            lora_name: "adapter".to_string(),
            ..base_request()
        },
        pb::GenerateRequest {
            kv_session: Some(pb::KvSessionRef::default()),
            ..base_request()
        },
        pb::GenerateRequest {
            data_parallel_rank: Some(0),
            ..base_request()
        },
    ] {
        assert_eq!(
            to_text_request(request, &["test-model".to_string()]).unwrap_err().code(),
            tonic::Code::Unimplemented
        );
    }
}

#[test]
fn terminal_event_maps_finish_reason_and_usage() {
    let responses = event_to_responses(
        DecodedTextEvent::TextDelta {
            delta: "done".to_string(),
            token_ids: vec![5],
            logprobs: None,
            finished: Some(finished(FinishReason::Length)),
        },
        "req",
    );
    assert_eq!(responses.len(), 2);
    let terminal = responses.last().unwrap();
    let Some(pb::generate_response::Event::Finished(event)) = &terminal.event else {
        panic!("expected finished event");
    };
    assert_eq!(event.reason, pb::FinishReason::Length as i32);
    let usage = terminal.usage.as_ref().unwrap();
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (3, 2, 5)
    );
}

#[test]
fn error_response_is_structured_and_terminal() {
    let response = error_response("req", "engine failed".to_string());
    let Some(pb::generate_response::Event::Error(error)) = response.event else {
        panic!("expected error event");
    };
    assert_eq!(error.code, pb::ErrorCode::Internal as i32);
    assert_eq!(error.message, "engine failed");
    assert!(response.usage.is_none());
}
