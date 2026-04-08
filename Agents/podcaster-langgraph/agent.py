import os
import re
from io import BytesIO
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote_plus
from urllib.request import urlopen
import wave
from functools import lru_cache
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
import lameenc
from openai import OpenAI

load_dotenv()

RESEARCH_MODEL = os.getenv("PODCAST_LANGGRAPH_RESEARCH_MODEL", "openai/gpt-oss-120b:free")
SCRIPT_MODEL = os.getenv("PODCAST_LANGGRAPH_SCRIPT_MODEL", "openai/gpt-oss-120b:free")
GEMINI_FALLBACK_MODEL = os.getenv("PODCAST_LANGGRAPH_GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")
AUDIO_MODEL = os.getenv("PODCAST_LANGGRAPH_AUDIO_MODEL", "gemini-2.5-flash-preview-tts")
SPEAKERS = [("Host A", "leda"), ("Host B", "Zephyr")]


class PodcastState(TypedDict, total=False):
    topic: str
    out_dir: Path
    research: str
    script: str
    audio_path: str


@lru_cache(maxsize=1)
def gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env or environment.")
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=1)
def openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def _rss_news(topic):
    url = f"https://news.google.com/rss/search?q={quote_plus(topic)}&hl=en-IN&gl=IN&ceid=IN:en"
    with urlopen(url, timeout=20) as response:
        root = ET.fromstring(response.read())
    items = []
    for item in root.findall("./channel/item")[:8]:
        items.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "date": (item.findtext("pubDate") or "").strip(),
                "source": ((item.find("source").text if item.find("source") is not None else "") or "").strip(),
            }
        )
    if not items:
        raise RuntimeError("No latest news results found for the topic.")
    return items


def _openrouter_text(prompt, model, temperature=0.3):
    client = openrouter_client()
    if not client:
        return None
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a concise research and script writing assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _openrouter_models(primary):
    models = [primary, "openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free", "openrouter/free"]
    seen = []
    for model in models:
        if model and model not in seen:
            seen.append(model)
    return seen


def research_topic(topic):
    print(f"[1/4] Fetching latest news for: {topic}", flush=True)
    items = _rss_news(topic)
    news = "\n".join(
        f"- {item['date']} | {item['source'] or 'Unknown source'} | {item['title']} | {item['link']}"
        for item in items
    )
    prompt = f"""Summarize these latest news items about "{topic}".
Return markdown only in this shape:
# Topic
## Latest News
- up to 5 bullets: YYYY-MM-DD | Source | headline
  one-sentence why it matters
## Big Picture
2 to 3 concise sentences
## Listener Takeaways
- 3 bullets max
Keep it under 350 words and use only the news below.

News items:
{news}"""
    print("[1/4] Summarizing research", flush=True)
    text = None
    for model in _openrouter_models(RESEARCH_MODEL):
        try:
            text = _openrouter_text(prompt, model)
        except Exception:
            text = None
        if text:
            break
    if not text:
        response = gemini_client().models.generate_content(
            model=GEMINI_FALLBACK_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=900),
        )
        text = (response.text or "").strip()
    return f"{text}\n\n## Grounding Links\n" + "\n".join(f"- {item['link']}" for item in items[:8]) + "\n"


def _normalize_script(text):
    lines = []
    for raw in (text or "").replace("```", "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("Host A:", "Host B:")):
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def _script_ok(script):
    lines = [line for line in script.splitlines() if line.strip()]
    speakers = [line.split(":", 1)[0] for line in lines]
    words = len(script.split())
    return (
        26 <= len(lines) <= 32
        and 380 <= words <= 520
        and set(speakers) == {"Host A", "Host B"}
        and all(a != b for a, b in zip(speakers, speakers[1:]))
    )


def write_script(topic, research):
    print("[2/4] Generating podcast script", flush=True)
    compact = research.split("## Grounding Links", 1)[0][:2500]
    prompt = f"""Write a sharp 3-minute podcast dialogue about "{topic}".
Rules:
- 420 to 480 words total
- 26 to 32 lines
- exactly two speakers: Host A and Host B
- alternate speakers every line
- every line starts with Host A: or Host B:
- Host A should be asking questions, Host B should be answering with insights from the research, but in a conversational way
- Host A should sound curious and a bit skeptical and ask followup questions, not just read the research points
- explain what changed, why it matters, tradeoffs, and the takeaway
- Our product name is InShorts, ask them to subscribe at the beginning if it fits naturally

Research:
{compact}"""
    feedback = ""
    client = openrouter_client()
    if client:
        for model in _openrouter_models(SCRIPT_MODEL):
            print(f"[2/4] Trying script model: {model}", flush=True)
            for _ in range(3):
                try:
                    script = _normalize_script(_openrouter_text(f"{prompt}\n{feedback}".strip(), model, 0.5))
                except Exception:
                    script = ""
                if _script_ok(script):
                    return script
                feedback = "Rewrite from scratch. The last attempt was too short or broke the format. Return only the full final dialogue."
    for _ in range(3):
        print(f"[2/4] Falling back to Gemini script model: {GEMINI_FALLBACK_MODEL}", flush=True)
        response = gemini_client().models.generate_content(
            model=GEMINI_FALLBACK_MODEL,
            contents=f"{prompt}\n{feedback}".strip(),
            config=types.GenerateContentConfig(temperature=0.5, max_output_tokens=1400),
        )
        script = _normalize_script(response.text or "")
        if _script_ok(script):
            return script
        feedback = "Rewrite from scratch. The last attempt was too short or broke the format. Return only the full final dialogue."
    raise RuntimeError("Podcast script generation failed.")


def render_audio(script, audio_path):
    print("[3/4] Generating audio", flush=True)
    response = gemini_client().models.generate_content(
        model=AUDIO_MODEL,
        contents=f"Read this exactly as a podcast dialogue between Host A and Host B.\n\n{script}",
        config=types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                            ),
                        )
                        for name, voice in SPEAKERS
                    ]
                )
            ),
        ),
    )
    clip = next(
        (
            (part.inline_data.data, part.inline_data.mime_type)
            for candidate in response.candidates or []
            for part in candidate.content.parts
            if part.inline_data and part.inline_data.data
        ),
        None,
    )
    if not clip:
        raise RuntimeError("Audio generation failed.")
    audio, mime_type = clip
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_quality(2)
    if mime_type and mime_type.startswith("audio/L16"):
        encoder.set_in_sample_rate(int(re.search(r"rate=(\d+)", mime_type).group(1)))
        encoder.set_channels(1)
        mp3 = encoder.encode(audio) + encoder.flush()
    else:
        with wave.open(BytesIO(audio), "rb") as wav_file:
            encoder.set_in_sample_rate(wav_file.getframerate())
            encoder.set_channels(wav_file.getnchannels())
            mp3 = encoder.encode(wav_file.readframes(wav_file.getnframes())) + encoder.flush()
    Path(audio_path).write_bytes(mp3)


def research_node(state: PodcastState):
    print("[1/4] Research stage started", flush=True)
    return {"research": research_topic(state["topic"])}


def script_node(state: PodcastState):
    print("[2/4] Script stage started", flush=True)
    return {"script": write_script(state["topic"], state["research"])}


def audio_node(state: PodcastState):
    print("[3/4] Audio stage started", flush=True)
    audio_path = state["out_dir"] / "podcast.mp3"
    try:
        render_audio(state["script"], audio_path)
        print("[3/4] Audio stage completed", flush=True)
        return {"audio_path": str(audio_path)}
    except Exception as exc:
        print(f"[3/4] Audio skipped: {exc}", flush=True)
        return {"audio_path": "", "audio_error": str(exc)}


def save_node(state: PodcastState):
    print("[4/4] Saving outputs", flush=True)
    (state["out_dir"] / "research.md").write_text(state["research"], encoding="utf-8")
    (state["out_dir"] / "script.txt").write_text(state["script"], encoding="utf-8")
    print("[4/4] Save completed", flush=True)
    return {}


def build_graph():
    graph = StateGraph(PodcastState)
    graph.add_node("research", research_node)
    graph.add_node("script", script_node)
    graph.add_node("audio", audio_node)
    graph.add_node("save", save_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "script")
    graph.add_edge("script", "audio")
    graph.add_edge("audio", "save")
    graph.add_edge("save", END)
    return graph.compile()
