use vllm_text::{DecodedTextEvent, FinishReason, Finished};

use super::super::pb;

pub fn event_to_responses(event: DecodedTextEvent, request_id: &str) -> Vec<pb::GenerateResponse> {
    match event {
        DecodedTextEvent::Start { .. } => Vec::new(),
        DecodedTextEvent::TextDelta {
            delta,
            token_ids,
            logprobs: _,
            finished,
        } => {
            let mut responses = Vec::new();
            if !token_ids.is_empty() || !delta.is_empty() {
                responses.push(pb::GenerateResponse {
                    request_id: request_id.to_string(),
                    event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                        token_ids,
                        text: delta,
                    })),
                    usage: None,
                });
            }
            if let Some(finished) = finished {
                responses.push(terminal_response(&finished, request_id));
            }
            responses
        }
    }
}

fn terminal_response(finished: &Finished, request_id: &str) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                reason: finish_reason_to_proto(&finished.finish_reason) as i32,
                message: String::new(),
            },
        )),
        usage: Some(to_usage(finished)),
    }
}

pub fn error_response(request_id: &str, message: String) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(pb::generate_response::Event::Error(pb::EngineError {
            code: pb::ErrorCode::Internal as i32,
            message,
            retry_hint: String::new(),
        })),
        usage: None,
    }
}

fn to_usage(finished: &Finished) -> pb::Usage {
    let prompt = finished.usage.prompt_token_count as u32;
    let completion = finished.usage.output_token_count as u32;
    pb::Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

fn finish_reason_to_proto(reason: &FinishReason) -> pb::FinishReason {
    match reason {
        FinishReason::Stop(_) | FinishReason::Repetition(_) => pb::FinishReason::Stop,
        FinishReason::Length => pb::FinishReason::Length,
        FinishReason::Abort => pb::FinishReason::Cancelled,
        FinishReason::Error => pb::FinishReason::Error,
    }
}
