import argparse
import re
from pathlib import Path

from agent import build_graph


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "topic"


def main():
    parser = argparse.ArgumentParser(description="Generate a 3-minute podcast with LangGraph.")
    parser.add_argument("topic", nargs="+", help="Topic to research and turn into a podcast")
    args = parser.parse_args()
    topic = " ".join(args.topic).strip()
    out_dir = Path(__file__).with_name("output") / slugify(topic)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_graph().invoke({"topic": topic, "out_dir": out_dir})
    print(f"Created podcast for '{topic}' in {out_dir}")
    if result.get("audio_path"):
        print(result["audio_path"])
    elif result.get("audio_error"):
        print(f"Audio skipped: {result['audio_error']}")


if __name__ == "__main__":
    main()
