import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the exported checkpoint.")
    parser.add_argument("--prompt", default="Hello", help="Prompt for a tiny generation smoke test.")
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        print(f"Failed to import vllm: {exc}", file=sys.stderr)
        raise

    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    generations = [
        {
            "prompt": output.prompt,
            "text": output.outputs[0].text,
            "token_ids": output.outputs[0].token_ids,
        }
        for output in outputs
    ]
    print(json.dumps(generations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
