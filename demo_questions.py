import argparse
import json
from pathlib import Path

from blasphemous.benchmark import generate_response, load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Interactive BLASPHEMOUS demo")
    parser.add_argument(
        "--model",
        default="./runs/liberated_qwen_release",
        help="Path to a saved model artifact",
    )
    parser.add_argument(
        "--output",
        default="demo_results.json",
        help="Path to save captured demo responses",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("BLASPHEMOUS Demo Questions")
    print("=" * 50)
    print(f"Model: {args.model}")
    print("=" * 50)

    model, tokenizer = load_model_and_tokenizer(args.model)
    results = []

    while True:
        user_input = input("\nPrompt (blank to stop): ").strip()
        if not user_input:
            break
        print("-" * 40)
        answer = generate_response(model, tokenizer, user_input, max_new_tokens=256)
        print(answer)
        print("-" * 40)
        results.append({"prompt": user_input, "response": answer})

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved demo session to {args.output}")


if __name__ == "__main__":
    main()
