import argparse
import json

from transformers import AutoConfig

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.models import registry as model_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--try_llm_load", action="store_true")
    args = parser.parse_args()

    hf_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=False)
    quant_cfg = hf_config.quantization_config
    config_cls = get_quantization_config(quant_cfg["quant_method"])
    config = config_cls.from_config(quant_cfg)
    config.maybe_update_config(args.model_dir)
    runtime_metadata = getattr(config, "module_runtime_metadata", {}) or {}
    checkpoint_summary = {
        "module_count": len(runtime_metadata),
        "sample_modules": sorted(runtime_metadata.keys())[:5],
    }

    architectures = getattr(hf_config, "architectures", [])
    registered_architectures = [
        arch for arch in architectures if arch in model_registry._TEXT_GENERATION_MODELS
    ]

    summary = {
        "architectures": architectures,
        "registered_architectures": registered_architectures,
        "registered_config_class": config_cls.__name__,
        "config_type": type(config).__name__,
        "weight_bits": getattr(config, "weight_bits", None),
        "activation_bits": getattr(config, "activation_bits", None),
        "group_size": getattr(config, "group_size", None),
        "runtime_format": getattr(config, "runtime_format", None),
        "runtime_module_count": len(runtime_metadata),
        "checkpoint_summary": checkpoint_summary,
    }

    if args.try_llm_load:
        try:
            from vllm import LLM

            llm = LLM(
                model=args.model_dir,
                tokenizer=args.model_dir,
                trust_remote_code=False,
                quantization="omni_activation_real",
                tensor_parallel_size=1,
                max_model_len=128,
                enforce_eager=True,
                disable_log_stats=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            summary["llm_load"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        else:  # pragma: no cover - unlikely on current scaffold
            summary["llm_load"] = {"ok": True, "type": type(llm).__name__}

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
