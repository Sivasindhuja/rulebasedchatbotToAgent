import os
from io import BytesIO
from pathlib import Path
import re
import wave

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from google.genai import types
import lameenc
from pydantic import BaseModel

load_dotenv()

TEXT_MODEL = os.getenv("PODCAST_TEXT_MODEL", "gemini-2.5-flash")
SCRIPT_MODEL = os.getenv("PODCAST_SCRIPT_MODEL", TEXT_MODEL)
AUDIO_MODEL = os.getenv("PODCAST_AUDIO_MODEL", "gemini-2.5-flash-preview-tts")
SPEAKERS = [("Host A", "leda"), ("Host B", "kore")]


class Turn(BaseModel):
    speaker: str
    text: str


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env or environment.")
    return genai.Client(api_key=api_key)


def _grounding_links(response):
    links = []
    for candidate in response.candidates or []:
        meta = candidate.grounding_metadata
        for chunk in meta.grounding_chunks or [] if meta else []:
            web = chunk.web
            if web and web.uri and web.uri not in links:
                links.append(web.uri)
    return links


def research_topic(client, topic):
    prompt = f"""Research the topic "{topic}" using live web search.
Return markdown with these sections only:
# Topic
## Latest News
- up to 5 bullets in this format: YYYY-MM-DD | Source | headline
  one-sentence why it matters
## Big Picture
2 to 3 concise sentences
## Listener Takeaways
- 3 bullets max
Keep the whole response under 350 words and focus on the latest credible developments."""
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=900,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    if not _grounding_links(response):
        raise RuntimeError("Live web grounding failed, so latest-news research was not generated.")
    body = (response.text or "").strip()
    links = "\n".join(f"- {url}" for url in _grounding_links(response)[:8])
    return f"{body}\n\n## Grounding Links\n{links}\n"


def write_script(client, topic, research):
    compact_research = research.split("## Grounding Links", 1)[0][:2500]
    base_prompt = f"""Write a sharp 3-minute podcast dialogue about "{topic}" using the research below.
Return JSON only as a list of 26 to 32 objects with keys speaker and text.
Rules:
- 420 to 480 words total
- exactly two speakers: Host A and Host B
- alternate speakers line by line
- no stage directions, titles, bullets, or narration
- make it insightful, current, and easy to follow
- explain what changed, why it matters, key tradeoffs, and the final takeaway

Research:
{compact_research}"""
    models = []
    for model in [SCRIPT_MODEL, TEXT_MODEL]:
        if model not in models:
            models.append(model)
    last_error = "unknown script error"
    for model in models:
        feedback = ""
        for _ in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"{base_prompt}\n{feedback}".strip(),
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        max_output_tokens=1400,
                        response_mime_type="application/json",
                        response_schema=list[Turn],
                    ),
                )
            except ClientError as exc:
                last_error = str(exc)
                if exc.status == 429:
                    break
                continue
            turns = response.parsed or []
            script = "\n".join(
                f"{turn['speaker']}: {turn['text'].strip()}"
                for turn in turns
                if turn.speaker in {"Host A", "Host B"} and turn.text
            )
            speakers = [line.split(":", 1)[0] for line in script.splitlines()]
            words = len(script.split())
            alternating = all(a != b for a, b in zip(speakers, speakers[1:]))
            if script and 26 <= len(speakers) <= 32 and alternating and set(speakers) == {"Host A", "Host B"} and 380 <= words <= 520:
                return f"{script}\n"
            feedback = f"Previous attempt was invalid. Rewrite from scratch with 28 alternating turns and about 450 words. It had {len(speakers)} turns and {words} words."
            last_error = f"invalid script from {model}: {len(speakers)} turns, {words} words"
    raise RuntimeError(f"Podcast script generation failed to reach the required 3-minute two-speaker format: {last_error}")


def render_audio(client, script, audio_path):
    response = client.models.generate_content(
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
        rate = int(re.search(r"rate=(\d+)", mime_type).group(1))
        encoder.set_in_sample_rate(rate)
        encoder.set_channels(1)
        mp3 = encoder.encode(audio) + encoder.flush()
    else:
        with wave.open(BytesIO(audio), "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                raise RuntimeError(f"Unsupported audio format returned by Gemini: {mime_type or 'unknown'}")
            encoder.set_in_sample_rate(wav_file.getframerate())
            encoder.set_channels(wav_file.getnchannels())
            mp3 = encoder.encode(wav_file.readframes(wav_file.getnframes())) + encoder.flush()
    Path(audio_path).write_bytes(mp3)
