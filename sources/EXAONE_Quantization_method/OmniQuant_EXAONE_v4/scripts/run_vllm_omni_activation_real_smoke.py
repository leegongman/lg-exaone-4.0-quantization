import argparse
import json

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt", default="1+1=")
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=256)
    parser.add_argument("--attention_backend", default="TRITON_ATTN")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    attention_config = (
        {"backend": args.attention_backend} if args.attention_backend else None
    )
    llm = LLM(
        model=args.model_dir,
        tokenizer=args.model_dir,
        trust_remote_code=False,
        quantization="omni_activation_real",
        tensor_parallel_size=1,
        dtype="float16",
        enforce_eager=True,
        disable_log_stats=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_config=attention_config,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    summary = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "text": outputs[0].outputs[0].text,
        "token_ids": outputs[0].outputs[0].token_ids,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
