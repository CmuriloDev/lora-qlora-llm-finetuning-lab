import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, RateLimitError, AuthenticationError, BadRequestError

load_dotenv()

DATA_DIR = Path("dataset")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TOPIC = "customer support for fitness gyms"
SAMPLES = 60
TRAIN_RATIO = 0.9
MODEL = "gpt-4o-mini"


def write_jsonl(path: Path, data: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_data(entries: list[dict]) -> list[dict]:
    output = []
    for e in entries:
        if isinstance(e, dict):
            p = e.get("prompt")
            r = e.get("response")
            if isinstance(p, str) and isinstance(r, str):
                p, r = p.strip(), r.strip()
                if p and r:
                    output.append({"prompt": p, "response": r})
    return output


def split_data(data: list[dict]):
    random.shuffle(data)
    idx = int(len(data) * TRAIN_RATIO)
    return data[:idx], data[idx:]


def build_payload():
    sys_msg = f"""
You are a synthetic dataset generator.

Generate data in the domain "{TOPIC}".

Return ONLY valid JSON list.

Each item must contain:
- prompt
- response

Constraints:
- Exactly {SAMPLES} items
- Natural user-like prompts
- Clear and useful responses
- No repetition
- No explanations outside JSON
"""

    user_msg = f"Generate {SAMPLES} prompt-response pairs."

    return [
        {"role": "system", "content": sys_msg.strip()},
        {"role": "user", "content": user_msg.strip()}
    ]


def strip_codeblock(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    return text.strip()


def run():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=key)

    try:
        result = client.chat.completions.create(
            model=MODEL,
            temperature=0.8,
            messages=build_payload(),
        )
    except AuthenticationError:
        raise RuntimeError("Invalid API key")
    except RateLimitError:
        raise RuntimeError("Quota exceeded")
    except BadRequestError:
        raise RuntimeError("Bad request to API")
    except APIError:
        raise RuntimeError("OpenAI API error")

    raw = result.choices[0].message.content or ""
    raw = strip_codeblock(raw)

    data = json.loads(raw)

    if not isinstance(data, list):
        raise ValueError("Invalid JSON format")

    data = clean_data(data)

    if len(data) < 50:
        raise ValueError("Insufficient valid samples")

    train, test = split_data(data)

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "test.jsonl", test)

    print("Dataset ready")
    print(f"Domain: {TOPIC}")
    print(f"Train: {len(train)} | Test: {len(test)}")


if __name__ == "__main__":
    run()