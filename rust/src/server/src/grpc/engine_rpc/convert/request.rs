use tonic::Status;
use uuid::Uuid;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextRequest};

use super::super::pb;

pub fn to_text_request(
    req: pb::GenerateRequest,
    served_model_names: &[String],
) -> Result<TextRequest, Status> {
    if !req.model.is_empty() && !served_model_names.iter().any(|name| name == &req.model) {
        return Err(Status::not_found(format!(
            "model `{}` not found",
            req.model
        )));
    }
    if !req.lora_name.is_empty() {
        return Err(Status::unimplemented(
            "LoRA request selection is not implemented",
        ));
    }
    if req.kv_session.is_some() {
        return Err(Status::unimplemented("KV sessions are not implemented"));
    }
    if req.data_parallel_rank.is_some() {
        return Err(Status::unimplemented(
            "forced data-parallel routing is not implemented",
        ));
    }

    let prompt = match req.input {
        Some(pb::generate_request::Input::Prompt(text)) => Prompt::Text(text),
        Some(pb::generate_request::Input::TokenIds(ids)) => Prompt::TokenIds(ids.ids),
        None => {
            return Err(Status::invalid_argument(
                "input (prompt or token_ids) is required",
            ));
        }
    };
    let request_id = if req.request_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        req.request_id
    };
    let min_tokens = req.sampling.as_ref().and_then(|sampling| sampling.min_tokens).unwrap_or(0);
    let mut sampling_params = build_sampling_params(req.sampling.as_ref())?;

    let mut stop_strings = Vec::new();
    let mut stop_token_ids = Vec::new();
    for condition in &req.stop {
        match condition.condition.as_ref() {
            Some(pb::stop_condition::Condition::StopText(text)) => {
                stop_strings.push(text.clone());
            }
            Some(pb::stop_condition::Condition::StopTokenId(id)) => stop_token_ids.push(*id),
            None => {
                return Err(Status::invalid_argument(
                    "stop condition must set stop_text or stop_token_id",
                ));
            }
        }
    }
    if !stop_token_ids.is_empty() {
        sampling_params.stop_token_ids = Some(stop_token_ids);
    }

    Ok(TextRequest {
        request_id,
        prompt,
        mm_features: None,
        sampling_params,
        decode_options: TextDecodeOptions {
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            stop_strings: (!stop_strings.is_empty()).then_some(stop_strings),
            min_tokens,
        },
        intermediate: req.stream,
        priority: req.priority.unwrap_or(0),
        cache_salt: None,
        add_special_tokens: true,
        data_parallel_rank: None,
        reasoning_parser_kwargs: None,
        lora_request: None,
        arrival_time: None,
    })
}

fn build_sampling_params(sampling: Option<&pb::SamplingParams>) -> Result<SamplingParams, Status> {
    let Some(sampling) = sampling else {
        return Ok(SamplingParams::default());
    };
    let top_k = match sampling.top_k {
        None => None,
        Some(-1 | 0) => Some(0),
        Some(value) if value > 0 => Some(value as u32),
        Some(value) => {
            return Err(Status::invalid_argument(format!(
                "top_k must be -1, 0, or a positive integer; got {value}"
            )));
        }
    };

    Ok(SamplingParams {
        temperature: sampling.temperature.map(|value| value as f32),
        top_p: sampling.top_p.map(|value| value as f32),
        top_k,
        seed: sampling.seed,
        max_tokens: sampling.max_tokens,
        min_tokens: sampling.min_tokens,
        min_p: sampling.min_p.map(|value| value as f32),
        frequency_penalty: sampling.frequency_penalty.map(|value| value as f32),
        presence_penalty: sampling.presence_penalty.map(|value| value as f32),
        repetition_penalty: sampling.repetition_penalty.map(|value| value as f32),
        ignore_eos: sampling.ignore_eos,
        ..SamplingParams::default()
    })
}
