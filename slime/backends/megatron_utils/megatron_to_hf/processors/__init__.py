from .padding_remover import remove_padding

__all__ = ["quantize_params", "remove_padding"]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params

    if quantization_config["quant_method"] == "fp8":
        from .quantizer_fp8 import quantize_params_fp8

        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)

    if quantization_config["quant_method"] == "compressed-tensors":
        from .quantizer_compressed_tensors import quantize_params_compressed_tensors

        # only int4 at the moment.
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)

    quant_algo = str(quantization_config.get("quant_algo", "")).upper()
    if quantization_config.get("quant_method") in {"modelopt", "modelopt_fp4"} or quant_algo in {
        "NVFP4",
        "FP4",
    }:
        from .quantizer_nvfp4 import quantize_params_nvfp4

        return quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config)

    # Unknown quant method — pass through BF16 params as-is
    return converted_named_params
