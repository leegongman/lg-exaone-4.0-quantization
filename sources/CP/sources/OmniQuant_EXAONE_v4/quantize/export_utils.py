import importlib.util
import json
import struct
from pathlib import Path
from typing import Optional


SUPPORTED_VLLM_GPTQ_BITS = {2, 3, 4, 8}


def has_auto_gptq() -> bool:
    return importlib.util.find_spec("auto_gptq") is not None


def uses_symmetric_weight_export(args) -> bool:
    return bool(getattr(args, "symmetric", False) or getattr(args, "disable_zero_point", False))


def uses_marlin_checkpoint_format(args) -> bool:
    return bool(
        uses_symmetric_weight_export(args)
        and int(args.wbits) in {4, 8}
        and getattr(args, "group_size", None) not in (None, 0)
    )


def build_vllm_gptq_config(args) -> dict:
    config = {
        "quant_method": "gptq",
        "bits": int(args.wbits),
        "group_size": args.group_size if args.group_size is not None else -1,
        "desc_act": False,
        "sym": uses_symmetric_weight_export(args),
        "true_sequential": True,
    }
    if uses_marlin_checkpoint_format(args):
        config["checkpoint_format"] = "gptq_marlin"
    return config


def validate_export_args(args):
    if getattr(args, "save_dir", None) is None:
        return

    if args.save_format == "compressed-tensors":
        raise ValueError(
            "The current repository does not emit packed compressed-tensors weights that "
            "vLLM expects. Use '--save_format hf' for a dense export or "
            "'--save_format gptq --real_quant' for a vLLM-compatible quantized export."
        )

    if args.save_format == "omni-act":
        if "exaone" not in args.net.lower():
            raise ValueError("Activation-aware real-quant export is currently implemented only for EXAONE.")
        if not args.real_quant:
            raise ValueError(
                "Activation-aware real-quant export requires '--real_quant' because it writes a custom "
                "runtime quantized checkpoint instead of a fake-quant dense checkpoint."
            )
        if args.wbits >= 16 or args.abits >= 16:
            raise ValueError(
                "Activation-aware real-quant export is only meaningful when both wbits<16 and abits<16."
            )
        return

    if args.save_format != "gptq":
        return

    if getattr(args, "real_quant", False) and not has_auto_gptq():
        raise ValueError(
            "real_quant requires 'auto_gptq' to be installed in the active environment."
        )

    if args.wbits not in SUPPORTED_VLLM_GPTQ_BITS:
        raise ValueError(
            f"vLLM GPTQ export only supports weight bits {sorted(SUPPORTED_VLLM_GPTQ_BITS)}, "
            f"but got wbits={args.wbits}."
        )

    if args.abits < 16:
        raise ValueError(
            "vLLM GPTQ export in this repo only supports weight-only checkpoints. "
            "Set abits>=16 or export with '--save_format hf'."
        )

    if not args.real_quant:
        raise ValueError(
            "vLLM GPTQ export requires '--real_quant' so the checkpoint is actually packed. "
            "Without packing, the saved tensors are not a valid GPTQ checkpoint."
        )

def write_quantize_config(save_dir: str, config: dict):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "quantize_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_dense_export_config(model):
    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    model.config.dtype = "float16"
    model.config.torch_dtype = "float16"


def _validate_single_safetensors_file(path: Path):
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))

    max_end = 0
    for tensor_name, tensor_info in header.items():
        if tensor_name == "__metadata__":
            continue
        begin, end = tensor_info["data_offsets"]
        if begin > end:
            raise ValueError(
                f"Invalid safetensors offsets in {path.name}: {tensor_name} has begin>end."
            )
        max_end = max(max_end, end)

    expected_size = 8 + header_len + max_end
    actual_size = path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(
            f"Invalid safetensors file {path.name}: header expects {expected_size} bytes "
            f"but file size is {actual_size} bytes."
        )


def verify_export_artifacts(save_dir: str, expected_quant_config: Optional[dict] = None):
    save_path = Path(save_dir)
    config_path = save_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Missing exported config.json in {save_dir}.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if expected_quant_config is not None:
        quant_config = config.get("quantization_config")
        if quant_config != expected_quant_config:
            raise ValueError(
                "Exported config.json quantization_config does not match the requested GPTQ layout."
            )

    safetensor_files = sorted(save_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No .safetensors weights were exported in {save_dir}.")

    for safetensor_file in safetensor_files:
        _validate_single_safetensors_file(safetensor_file)
