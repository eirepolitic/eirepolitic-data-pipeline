# process/openai_smoke_test.py
import os
import sys
from openai import OpenAI

def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("❌ OPENAI_API_KEY is missing or empty.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input="Reply with exactly: OK",
            max_output_tokens=16,  # minimum is 16
        )
        text = (resp.output_text or "").strip()
    except Exception as e:
        print(f"❌ OpenAI smoke test failed: {e}")
        sys.exit(1)

    if text != "OK":
        print(f"⚠️ OpenAI smoke test got unexpected response: {text!r}")
        sys.exit(1)

    print("✅ OpenAI smoke test passed (got OK).")

if __name__ == "__main__":
    main()
