"""Quantize Megatron expert weights to ModelOpt NVFP4 for SGLang rollout sync."""

from __future__ import annotations

import re

import torch

from .nvfp4_weight_quant import quantize_named_gated_pair_nvfp4, quantize_named_weight_nvfp4

GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}

# Streaming update_weight can deliver gate and up in consecutive calls.
_gated_pair_pending: dict[str, tuple[str, str, torch.Tensor]] = {}


def _ignored_module_names(quantization_config):
    names = []
    for key in ("modules_to_not_convert", "ignored_layers", "ignore"):
        value = quantization_config.get(key) or []
        names.extend(value)
    return names


def _should_skip_nvfp4_quant(converted_name: str, ignored_module_names: list[str]) -> bool:
    if not ignored_module_names:
        return False
    stem = converted_name[: -len(".weight")] if converted_name.endswith(".weight") else converted_name
    for rule in ignored_module_names:
        if not rule:
            continue
        if stem == rule or converted_name == rule:
            return True
        if stem.startswith(rule) or converted_name.startswith(rule):
            return True
    return False


def _split_gated_pair_name(name: str) -> tuple[str | None, str | None]:
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def _quantize_moe_named_params(
    converted_named_params: list[tuple[str, torch.Tensor]],
    ignored_module_names: list[str],
) -> list[tuple[str, torch.Tensor]]:
    quantize_named_params: list[tuple[str, torch.Tensor]] = []
    gated_candidates: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}

    for converted_name, param in converted_named_params:
        if converted_name.endswith("_scale") or converted_name.endswith("input_scale"):
            continue
        if _should_skip_nvfp4_quant(converted_name, ignored_module_names):
            quantize_named_params.append((converted_name, param))
            continue
        if not converted_name.endswith(".weight"):
            quantize_named_params.append((converted_name, param))
            continue

        base, role = _split_gated_pair_name(converted_name)
        if base is not None and role in ("gate", "up"):
            gated_candidates.setdefault(base, {})[role] = (converted_name, param)
            continue

        quantize_named_params.extend(quantize_named_weight_nvfp4(converted_name, param))

    for base, roles in gated_candidates.items():
        if "gate" in roles and "up" in roles:
            gate_name, gate_weight = roles["gate"]
            up_name, up_weight = roles["up"]
            quantize_named_params.extend(
                quantize_named_gated_pair_nvfp4(gate_name, gate_weight, up_name, up_weight)
            )
            _gated_pair_pending.pop(base, None)
            continue

        role, (converted_name, param) = next(iter(roles.items()))
        pending = _gated_pair_pending.get(base)
        if pending is None:
            _gated_pair_pending[base] = (role, converted_name, param)
            continue

        pending_role, pending_name, pending_weight = pending
        if pending_role == role:
            _gated_pair_pending[base] = (role, converted_name, param)
            continue
        _gated_pair_pending.pop(base, None)
        if pending_role == "gate":
            quantize_named_params.extend(
                quantize_named_gated_pair_nvfp4(pending_name, pending_weight, converted_name, param)
            )
        else:
            quantize_named_params.extend(
                quantize_named_gated_pair_nvfp4(converted_name, param, pending_name, pending_weight)
            )

    return quantize_named_params


def quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config):
    del args
    quant_algo = str(quantization_config.get("quant_algo", "")).upper()
    quant_method = quantization_config.get("quant_method")
    if quant_method not in {"modelopt", "modelopt_fp4"} and quant_algo not in {"NVFP4", "FP4"}:
        raise ValueError(f"Unexpected NVFP4 quantization config: {quantization_config}")

    ignored_module_names = _ignored_module_names(quantization_config)

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)
    if not match:
        return converted_named_params

    _layer_idx, rest = match.groups()

    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _expert_idx = match.groups()
        if rest in {"linear_fc1", "linear_fc2"}:
            return _quantize_moe_named_params(converted_named_params, ignored_module_names)

    return converted_named_params
