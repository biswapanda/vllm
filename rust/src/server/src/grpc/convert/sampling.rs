use tonic::Status;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputOptions, StructuredOutputsParams,
};
use vllm_text::SamplingParams;

use super::pb;

pub(super) fn build_sampling_params(
    temperature: Option<f32>,
    sampling: Option<&pb::RandomSampling>,
    decoding: Option<&pb::DecodingParameters>,
    stopping: Option<&pb::StoppingCriteria>,
    response: Option<&pb::ResponseOptions>,
) -> Result<SamplingParams, Status> {
    // Temperature is a top-level GenerateRequest field. Default to greedy (0.0) for
    // the gRPC API when the caller does not specify a value. This differs from
    // the HTTP/OpenAI API (which defaults to 1.0) and matches the convention of
    // programmatic generation APIs.
    let temperature = temperature.or(Some(0.0));
    let mut params = SamplingParams {
        temperature,
        ..SamplingParams::default()
    };

    // Optional scalar presence preserves explicit zero and sentinel values;
    // omitted fields remain available for model defaults during lowering.
    if let Some(s) = sampling {
        // num_sequences (n > 1) is not supported yet by the TextLlm layer; the response
        // path also hardcodes SequenceOutput.index = 0, so accepting >1 would silently
        // truncate output cardinality. Reject explicitly.
        if s.num_sequences > 1 {
            return Err(Status::invalid_argument(
                "num_sequences > 1 is not supported",
            ));
        }
        params.top_k = s
            .top_k
            .map(|value| match value {
                -1 => Ok(0),
                0.. => u32::try_from(value)
                    .map_err(|_| Status::invalid_argument("top_k exceeds uint32")),
                _ => Err(Status::invalid_argument("top_k must be at least -1")),
            })
            .transpose()?;
        params.top_p = s.top_p;
        params.min_p = s.min_p;
        params.seed = s.seed;
    }

    // DecodingParameters
    if let Some(d) = decoding {
        params.presence_penalty = d.presence_penalty;
        params.frequency_penalty = d.frequency_penalty;
        params.repetition_penalty = d.repetition_penalty;
        if !d.logit_bias.is_empty() {
            params.logit_bias = Some(d.logit_bias.clone());
        }
        if !d.allowed_token_ids.is_empty() {
            params.allowed_token_ids = Some(d.allowed_token_ids.clone());
        }
        if !d.bad_words.is_empty() {
            params.bad_words = Some(d.bad_words.clone());
        }
        params.structured_outputs = convert_structured_output(d)?;
    }

    // StoppingCriteria
    if let Some(s) = stopping {
        if s.max_new_tokens != 0 {
            params.max_tokens = Some(s.max_new_tokens);
        }
        if s.min_new_tokens != 0 {
            params.min_tokens = Some(s.min_new_tokens);
        }
        if !s.stop_token_ids.is_empty() {
            params.stop_token_ids = Some(s.stop_token_ids.clone());
        }
        params.ignore_eos = s.ignore_eos;
        params.thinking_token_budget = s.thinking_token_budget;
    }

    // ResponseOptions → logprobs
    if let Some(r) = response {
        if r.output_logprobs {
            let (count, token_ids) = candidate_logprob_spec(r.output_candidates.as_ref());
            params.logprobs = Some(count);
            params.logprob_token_ids = token_ids;
        }
        if r.prompt_logprobs {
            // The engine-core protocol has only one shared `logprob_token_ids` field
            // for output and prompt logprobs, so a per-token-id selector for prompt
            // candidates can't be honored independently. Reject it instead of silently
            // dropping the list.
            if matches!(
                r.prompt_candidates.as_ref().and_then(|c| c.select.as_ref()),
                Some(pb::candidate_tokens::Select::TokenIds(_))
            ) {
                return Err(Status::invalid_argument(
                    "prompt_candidates token_ids selector is not supported",
                ));
            }
            let (count, _) = candidate_logprob_spec(r.prompt_candidates.as_ref());
            params.prompt_logprobs = Some(count);
        }
    }

    Ok(params)
}

/// Map the proto `CandidateTokens` selector to a `(logprobs_count,
/// logprob_token_ids)` pair.
///
/// - `top_n(k)` → `(k, None)` — return top-k candidates by probability
/// - `all` → `(-1, None)` — return the full vocabulary
/// - `token_ids(n)` → `(1, Some(vec of n token ids))` — return logprobs for specific tokens (the
///   count `n` is stored in the proto as the number of token IDs that follow, but the actual IDs
///   are carried via `logprob_token_ids` on `SamplingParams`)
/// - absent → `(1, None)` — just the sampled/scored token
fn candidate_logprob_spec(candidates: Option<&pb::CandidateTokens>) -> (i32, Option<Vec<u32>>) {
    match candidates.and_then(|c| c.select.as_ref()) {
        Some(pb::candidate_tokens::Select::TopN(n)) => (*n as i32, None),
        Some(pb::candidate_tokens::Select::All(true)) => (-1, None),
        Some(pb::candidate_tokens::Select::TokenIds(ids)) => (
            ids.ids.len().try_into().unwrap_or(i32::MAX),
            Some(ids.ids.clone()),
        ),
        _ => (1, None),
    }
}

fn convert_structured_output(
    d: &pb::DecodingParameters,
) -> Result<Option<StructuredOutputsParams>, Status> {
    let so = match d.structured_output.as_ref() {
        None => return Ok(None),
        Some(so) => so,
    };
    use pb::decoding_parameters::StructuredOutput;
    let mut params = match so {
        StructuredOutput::Json(schema) => {
            let json: serde_json::Value = serde_json::from_str(schema)
                .map_err(|e| Status::invalid_argument(format!("invalid json schema: {e}")))?;
            StructuredOutputsParams::json(json)
        }
        StructuredOutput::Regex(regex) => StructuredOutputsParams::regex(regex.clone()),
        StructuredOutput::Choice(choices) => {
            StructuredOutputsParams::choice(choices.choices.clone())
        }
        StructuredOutput::Grammar(grammar) => StructuredOutputsParams::grammar(grammar.clone()),
        StructuredOutput::JsonObject(true) => StructuredOutputsParams::json_object(),
        StructuredOutput::JsonObject(false) => return Ok(None),
        StructuredOutput::StructuralTag(tag) => {
            StructuredOutputsParams::structural_tag(tag.clone())
        }
    };
    params.options = StructuredOutputOptions {
        disable_any_whitespace: d.structured_output_disable_any_whitespace,
        disable_additional_properties: d.structured_output_disable_additional_properties,
        whitespace_pattern: d.structured_output_whitespace_pattern.clone(),
    };
    Ok(Some(params))
}
