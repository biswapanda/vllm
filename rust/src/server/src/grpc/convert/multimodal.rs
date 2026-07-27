use std::collections::HashMap;
use std::io::Cursor;

use serde::Deserialize as _;
use sha2::{Digest, Sha256};
use tonic::Status;
use vllm_chat::MediaContentPart;
use vllm_engine_core_client::protocol::multimodal::{
    MmFeatureSpec, MmField, MmKwargValue, MmKwargsItem, MmSlice, PlaceholderRange,
};
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};
use vllm_text::Prompt;

use super::pb;

const MAX_MM_FEATURE_BYTES: usize = 16 * 1024 * 1024;
const MAX_MM_FEATURES: usize = 64;
const MAX_MM_DEPTH: usize = 32;
const MAX_MM_NODES: usize = 65_536;
const MAX_MM_FIELDS_PER_ITEM: usize = 256;
const MAX_MM_KEY_BYTES: usize = 256;
const MAX_MM_HASH_BYTES: usize = 256;
const MAX_MM_TENSOR_RANK: usize = 32;

pub(crate) fn media_parts_from_request(
    media: &[pb::MediaItem],
) -> Result<Vec<MediaContentPart>, Status> {
    let mut parts = Vec::with_capacity(media.len());
    for item in media {
        let modality = pb::Modality::try_from(item.modality).map_err(|_| {
            Status::invalid_argument(format!("unknown media modality {}", item.modality))
        })?;
        match modality {
            pb::Modality::Image | pb::Modality::Unspecified => {}
            other => {
                return Err(Status::unimplemented(format!(
                    "media modality {other:?} is not supported by the gRPC service"
                )));
            }
        }
        let uuid = (!item.uuid.is_empty()).then(|| item.uuid.clone());
        let part = match item.source.as_ref() {
            Some(pb::media_item::Source::Url(url)) | Some(pb::media_item::Source::DataUri(url)) => {
                MediaContentPart::ImageUrl {
                    url: url.clone(),
                    detail: None,
                    uuid,
                }
            }
            Some(pb::media_item::Source::RawBytes(bytes)) => MediaContentPart::ImageData {
                data: bytes.clone(),
                mime_type: (!item.mime_type.is_empty()).then(|| item.mime_type.clone()),
                uuid,
                detail: None,
            },
            None => return Err(Status::invalid_argument("media item has no source")),
        };
        parts.push(part);
    }
    Ok(parts)
}

pub(super) fn convert_mm_features(
    features: &[pb::PreprocessedMultimodalFeature],
    prompt: &Prompt,
) -> Result<Option<Vec<MmFeatureSpec>>, Status> {
    if features.is_empty() {
        return Ok(None);
    }
    let Prompt::TokenIds(token_ids) = prompt else {
        return Err(Status::invalid_argument(
            "preprocessed multimodal features require token_ids input",
        ));
    };
    if features.len() > MAX_MM_FEATURES || features.len() > token_ids.len() {
        return Err(Status::resource_exhausted(
            "too many preprocessed multimodal features",
        ));
    }

    let mut encoded_bytes = 0usize;
    let mut wire_nodes = 0usize;
    let mut converted = Vec::with_capacity(features.len());
    for feature in features {
        if feature.modality.is_empty() || feature.modality.len() > 64 {
            return Err(Status::invalid_argument(
                "multimodal feature modality must contain between 1 and 64 bytes",
            ));
        }
        if feature.mm_hash.is_empty() || feature.mm_hash.len() > MAX_MM_HASH_BYTES {
            return Err(Status::invalid_argument(
                "multimodal feature mm_hash must contain between 1 and 256 bytes",
            ));
        }
        let position = feature
            .position
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("multimodal feature position is required"))?;
        let offset = usize::try_from(position.offset)
            .map_err(|_| Status::invalid_argument("multimodal feature offset is too large"))?;
        let length = usize::try_from(position.length)
            .map_err(|_| Status::invalid_argument("multimodal feature length is too large"))?;
        if length == 0 {
            return Err(Status::invalid_argument(
                "multimodal feature length must be positive",
            ));
        }
        let end = offset
            .checked_add(length)
            .ok_or_else(|| Status::invalid_argument("multimodal feature range overflows"))?;
        if end > token_ids.len() {
            return Err(Status::invalid_argument(
                "multimodal feature range exceeds token_ids",
            ));
        }
        if !position.is_embed.is_empty() && position.is_embed.len() != length {
            return Err(Status::invalid_argument(
                "multimodal feature is_embed length must match position length",
            ));
        }
        let is_embed = if position.is_embed.is_empty() {
            None
        } else {
            Some(
                WireTensor::from_bool(vec![length], position.is_embed.clone())
                    .map_err(Status::invalid_argument)?,
            )
        };

        let raw = feature.kwargs_msgpack.as_deref().ok_or_else(|| {
            Status::invalid_argument(
                "multimodal feature kwargs_msgpack is required; unverified cache hits are unsupported",
            )
        })?;
        encoded_bytes = encoded_bytes
            .checked_add(raw.len())
            .ok_or_else(|| Status::resource_exhausted("multimodal feature payload is too large"))?;
        if encoded_bytes > MAX_MM_FEATURE_BYTES {
            return Err(Status::resource_exhausted(
                "multimodal feature payload exceeds 16 MiB",
            ));
        }
        let item = decode_mm_kwargs(raw, &mut wire_nodes)?;
        validate_mm_kwargs_item(&item)?;
        let identifier = mm_cache_identifier(&feature.modality, raw);
        if feature.cache_identifier != identifier {
            return Err(Status::invalid_argument(
                "multimodal feature cache_identifier does not match its canonical payload identity",
            ));
        }
        if feature.mm_hash != identifier {
            return Err(Status::invalid_argument(
                "multimodal feature mm_hash does not match its canonical payload identity",
            ));
        }

        converted.push(MmFeatureSpec {
            data: Some(item),
            modality: feature.modality.clone(),
            identifier: identifier.clone(),
            mm_position: PlaceholderRange {
                offset,
                length,
                is_embed,
            },
            mm_hash: Some(identifier),
        });
    }
    converted.sort_by_key(|feature| feature.mm_position.offset);
    for pair in converted.windows(2) {
        let previous_end = pair[0]
            .mm_position
            .offset
            .checked_add(pair[0].mm_position.length)
            .ok_or_else(|| Status::invalid_argument("multimodal feature range overflows"))?;
        if previous_end > pair[1].mm_position.offset {
            return Err(Status::invalid_argument(
                "multimodal feature ranges must not overlap",
            ));
        }
    }
    validate_mm_field_metadata(&converted)?;
    Ok(Some(converted))
}

fn validate_mm_kwargs_item(item: &MmKwargsItem) -> Result<(), Status> {
    if item.is_empty() || item.len() > MAX_MM_FIELDS_PER_ITEM {
        return Err(Status::invalid_argument(
            "multimodal kwargs item must contain between 1 and 256 fields",
        ));
    }
    for (key, element) in item {
        if key.is_empty() || key.len() > MAX_MM_KEY_BYTES {
            return Err(Status::invalid_argument(
                "multimodal kwargs keys must contain between 1 and 256 bytes",
            ));
        }
        let value = element.data.as_ref().ok_or_else(|| {
            Status::invalid_argument("multimodal kwargs fields must carry inline data")
        })?;
        validate_mm_kwarg_value(value, 0)?;
    }
    Ok(())
}

fn validate_mm_kwarg_value(value: &MmKwargValue, depth: usize) -> Result<(), Status> {
    if depth > 32 {
        return Err(Status::invalid_argument(
            "multimodal kwargs nesting exceeds 32 levels",
        ));
    }
    match value {
        MmKwargValue::Tensor(tensor) => validate_wire_tensor(tensor),
        MmKwargValue::List(values) => {
            for value in values {
                validate_mm_kwarg_value(value, depth + 1)?;
            }
            Ok(())
        }
        MmKwargValue::Int(_) | MmKwargValue::Float(_) => Ok(()),
    }
}

fn validate_wire_tensor(tensor: &WireTensor) -> Result<(), Status> {
    if tensor.shape.len() > MAX_MM_TENSOR_RANK {
        return Err(Status::invalid_argument(
            "multimodal tensor rank exceeds 32",
        ));
    }
    let width = match tensor.dtype.as_str() {
        "bool" | "uint8" | "int8" => 1,
        "float16" | "bfloat16" | "uint16" | "int16" => 2,
        "float32" | "uint32" | "int32" => 4,
        "float64" | "uint64" | "int64" => 8,
        dtype => {
            return Err(Status::invalid_argument(format!(
                "unsupported multimodal tensor dtype {dtype:?}"
            )));
        }
    };
    let numel = tensor
        .shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| Status::invalid_argument("multimodal tensor shape overflows"))?;
    let expected = numel
        .checked_mul(width)
        .ok_or_else(|| Status::invalid_argument("multimodal tensor byte length overflows"))?;
    match &tensor.data {
        WireArrayData::RawView(bytes) if bytes.len() == expected => Ok(()),
        WireArrayData::RawView(bytes) => Err(Status::invalid_argument(format!(
            "multimodal tensor byte length {} does not match expected {expected}",
            bytes.len()
        ))),
        WireArrayData::AuxIndex(_) => Err(Status::invalid_argument(
            "multimodal kwargs must encode tensors inline",
        )),
    }
}

pub(super) fn mm_cache_identifier(modality: &str, raw: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vllm.grpc.preprocessed-mm.v1");
    hasher.update((modality.len() as u64).to_be_bytes());
    hasher.update(modality.as_bytes());
    hasher.update((raw.len() as u64).to_be_bytes());
    hasher.update(raw);
    format!("grpc-mm:{:x}", hasher.finalize())
}

fn decode_mm_kwargs(raw: &[u8], nodes: &mut usize) -> Result<MmKwargsItem, Status> {
    preflight_msgpack(raw, nodes)?;
    let mut deserializer = rmp_serde::Deserializer::new(Cursor::new(raw));
    deserializer.set_max_depth(MAX_MM_DEPTH);
    let item = MmKwargsItem::deserialize(&mut deserializer).map_err(|error| {
        Status::invalid_argument(format!("invalid multimodal kwargs msgpack: {error}"))
    })?;
    if deserializer.position() != raw.len() as u64 {
        return Err(Status::invalid_argument(
            "multimodal kwargs msgpack contains trailing data",
        ));
    }
    Ok(item)
}

pub(super) fn preflight_msgpack(raw: &[u8], nodes: &mut usize) -> Result<(), Status> {
    let mut cursor = 0usize;
    scan_msgpack_value(raw, &mut cursor, 0, nodes)?;
    if cursor != raw.len() {
        return Err(Status::invalid_argument(
            "multimodal kwargs msgpack contains trailing data",
        ));
    }
    Ok(())
}

fn scan_msgpack_value(
    raw: &[u8],
    cursor: &mut usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), Status> {
    if depth > MAX_MM_DEPTH {
        return Err(Status::resource_exhausted(
            "multimodal kwargs nesting exceeds 32 levels",
        ));
    }
    *nodes = nodes
        .checked_add(1)
        .ok_or_else(|| Status::resource_exhausted("multimodal kwargs are too complex"))?;
    if *nodes > MAX_MM_NODES {
        return Err(Status::resource_exhausted(
            "multimodal kwargs contain too many values",
        ));
    }
    let marker = take_msgpack(raw, cursor, 1)?[0];
    match marker {
        0x00..=0x7f | 0xc0 | 0xc2 | 0xc3 | 0xe0..=0xff => Ok(()),
        0x80..=0x8f => {
            scan_msgpack_children(raw, cursor, depth, nodes, (marker & 0x0f) as usize * 2)
        }
        0x90..=0x9f => scan_msgpack_children(raw, cursor, depth, nodes, (marker & 0x0f) as usize),
        0xa0..=0xbf => skip_msgpack(raw, cursor, (marker & 0x1f) as usize),
        0xc1 => Err(Status::invalid_argument("reserved MessagePack marker")),
        0xc4 => {
            let len = read_msgpack_u8(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xc5 => {
            let len = read_msgpack_u16(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xc6 => {
            let len = read_msgpack_u32(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xc7 => {
            let len = read_msgpack_u8(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len + 1)
        }
        0xc8 => {
            let len = read_msgpack_u16(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len + 1)
        }
        0xc9 => {
            let len = read_msgpack_u32(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len + 1)
        }
        0xca => skip_msgpack(raw, cursor, 4),
        0xcb => skip_msgpack(raw, cursor, 8),
        0xcc | 0xd0 => skip_msgpack(raw, cursor, 1),
        0xcd | 0xd1 => skip_msgpack(raw, cursor, 2),
        0xce | 0xd2 => skip_msgpack(raw, cursor, 4),
        0xcf | 0xd3 => skip_msgpack(raw, cursor, 8),
        0xd4 => skip_msgpack(raw, cursor, 2),
        0xd5 => skip_msgpack(raw, cursor, 3),
        0xd6 => skip_msgpack(raw, cursor, 5),
        0xd7 => skip_msgpack(raw, cursor, 9),
        0xd8 => skip_msgpack(raw, cursor, 17),
        0xd9 => {
            let len = read_msgpack_u8(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xda => {
            let len = read_msgpack_u16(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xdb => {
            let len = read_msgpack_u32(raw, cursor)? as usize;
            skip_msgpack(raw, cursor, len)
        }
        0xdc => {
            let count = read_msgpack_u16(raw, cursor)? as usize;
            scan_msgpack_children(raw, cursor, depth, nodes, count)
        }
        0xdd => {
            let count = read_msgpack_u32(raw, cursor)? as usize;
            scan_msgpack_children(raw, cursor, depth, nodes, count)
        }
        0xde => {
            let count = (read_msgpack_u16(raw, cursor)? as usize)
                .checked_mul(2)
                .ok_or_else(|| Status::resource_exhausted("MessagePack map is too large"))?;
            scan_msgpack_children(raw, cursor, depth, nodes, count)
        }
        0xdf => {
            let count = (read_msgpack_u32(raw, cursor)? as usize)
                .checked_mul(2)
                .ok_or_else(|| Status::resource_exhausted("MessagePack map is too large"))?;
            scan_msgpack_children(raw, cursor, depth, nodes, count)
        }
    }
}

fn scan_msgpack_children(
    raw: &[u8],
    cursor: &mut usize,
    depth: usize,
    nodes: &mut usize,
    count: usize,
) -> Result<(), Status> {
    if count > MAX_MM_NODES.saturating_sub(*nodes) {
        return Err(Status::resource_exhausted(
            "multimodal kwargs contain too many values",
        ));
    }
    for _ in 0..count {
        scan_msgpack_value(raw, cursor, depth + 1, nodes)?;
    }
    Ok(())
}

fn take_msgpack<'a>(raw: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], Status> {
    let end = cursor
        .checked_add(len)
        .filter(|end| *end <= raw.len())
        .ok_or_else(|| Status::invalid_argument("truncated multimodal kwargs msgpack"))?;
    let bytes = &raw[*cursor..end];
    *cursor = end;
    Ok(bytes)
}

fn skip_msgpack(raw: &[u8], cursor: &mut usize, len: usize) -> Result<(), Status> {
    take_msgpack(raw, cursor, len).map(|_| ())
}

fn read_msgpack_u8(raw: &[u8], cursor: &mut usize) -> Result<u8, Status> {
    Ok(take_msgpack(raw, cursor, 1)?[0])
}

fn read_msgpack_u16(raw: &[u8], cursor: &mut usize) -> Result<u16, Status> {
    Ok(u16::from_be_bytes(
        take_msgpack(raw, cursor, 2)?.try_into().expect("fixed two-byte slice"),
    ))
}

fn read_msgpack_u32(raw: &[u8], cursor: &mut usize) -> Result<u32, Status> {
    Ok(u32::from_be_bytes(
        take_msgpack(raw, cursor, 4)?.try_into().expect("fixed four-byte slice"),
    ))
}

fn validate_mm_field_metadata(features: &[MmFeatureSpec]) -> Result<(), Status> {
    let mut occurrences: HashMap<(String, String), usize> = HashMap::new();
    let mut fields: HashMap<(String, String), MmField> = HashMap::new();
    for feature in features {
        let item = feature.data.as_ref().expect("inline multimodal data is required above");
        for (key, element) in item {
            let identity = (feature.modality.clone(), key.clone());
            *occurrences.entry(identity.clone()).or_default() += 1;
            if let Some(previous) = fields.get(&identity) {
                if previous != &element.field {
                    return Err(Status::invalid_argument(
                        "multimodal field configuration differs across items",
                    ));
                }
            } else {
                fields.insert(identity, element.field.clone());
            }
        }
    }
    for feature in features {
        let item = feature.data.as_ref().expect("inline multimodal data is required above");
        for (key, element) in item {
            let count = occurrences[&(feature.modality.clone(), key.clone())];
            validate_mm_field(
                &element.field,
                element.data.as_ref().expect("inline multimodal field data is required above"),
                count,
            )?;
        }
    }
    Ok(())
}

fn validate_mm_field(
    field: &MmField,
    data: &MmKwargValue,
    occurrences: usize,
) -> Result<(), Status> {
    match field {
        MmField::Batched(_) => Ok(()),
        MmField::Shared(shared) => {
            if shared.batch_size == 0 || shared.batch_size != occurrences {
                return Err(Status::invalid_argument(
                    "multimodal shared-field batch_size must match item count",
                ));
            }
            Ok(())
        }
        MmField::Flat(flat) => {
            if flat.slices.is_empty() || flat.slices.len() != occurrences {
                return Err(Status::invalid_argument(
                    "multimodal flat-field slices must match item count",
                ));
            }
            for slice in &flat.slices {
                match slice {
                    MmSlice::Slice(slice) => validate_slice_step(slice.step)?,
                    MmSlice::Slices(slices) => {
                        if slices.is_empty() || slices.len() > MAX_MM_TENSOR_RANK {
                            return Err(Status::invalid_argument(
                                "multimodal flat-field slice tuple must contain 1 to 32 slices",
                            ));
                        }
                        for slice in slices {
                            validate_slice_step(slice.step)?;
                        }
                    }
                }
            }
            match data {
                MmKwargValue::Tensor(tensor) => {
                    let rank = i32::try_from(tensor.shape.len()).unwrap_or(i32::MAX);
                    if rank == 0 || flat.dim < -rank || flat.dim >= rank {
                        return Err(Status::invalid_argument(
                            "multimodal flat-field dim is outside the tensor rank",
                        ));
                    }
                }
                _ if flat.dim != 0 => {
                    return Err(Status::invalid_argument(
                        "multimodal non-tensor flat fields require dim=0",
                    ));
                }
                _ => {}
            }
            Ok(())
        }
    }
}

fn validate_slice_step(step: Option<isize>) -> Result<(), Status> {
    if step == Some(0) {
        return Err(Status::invalid_argument(
            "multimodal slice step must not be zero",
        ));
    }
    Ok(())
}
