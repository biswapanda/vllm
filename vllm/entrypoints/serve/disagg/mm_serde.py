# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for multimodal serialization helpers."""

from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)

__all__ = ["decode_mm_kwargs_item", "encode_mm_kwargs_item"]
