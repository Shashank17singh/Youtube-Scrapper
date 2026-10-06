"""
Handles interaction with faster-whisper to transcribe audio files into text
segments. Supports batched/sequential inference on CUDA or CPU int8.
"""

import json
import os
from pathlib import Path

from ytrag.config import (
    TRANSCRIPT_DIR,
    WHISPER_BATCH,
    WHISPER_BEAM,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    WHISPER_LANG,
    WHISPER_MODEL,
)
from ytrag.models import Segment, Video
from ytrag.playlist import delete_audio, download_audio

_MODEL_CACHE: dict[tuple, object] = {}


def _resolve_device() -> tuple[str, str]:
    device = WHISPER_DEVICE
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    compute = WHISPER_COMPUTE or ("float16" if device == "cuda" else "int8")
    return device, compute


def get_model(model_name: str = WHISPER_MODEL):
    from faster_whisper import WhisperModel

    device, compute = _resolve_device()
    key = (model_name, device, compute)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"[transcribe] CUDA unavailable ({exc}); falling back to CPU int8.")
        device, compute = "cpu", "int8"
        key = (model_name, device, compute)
        model = WhisperModel(model_name, device=device, compute_type=compute)

    _MODEL_CACHE[key] = model
    return model


def get_batched_model(model_name: str = WHISPER_MODEL):
    from faster_whisper import BatchedInferencePipeline

    model = get_model(model_name)
    key = ("batched", id(model))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = BatchedInferencePipeline(model=model)
    return _MODEL_CACHE[key]


def run_whisper(audio_path: str, language: str, model_name: str = WHISPER_MODEL):
    if WHISPER_BATCH > 0:
        try:
            return get_batched_model(model_name).transcribe(
                audio_path,
                language=language,
                beam_size=WHISPER_BEAM,
                batch_size=WHISPER_BATCH,
                vad_filter=True,
            )
        except (ImportError, AttributeError, TypeError) as exc:
            print(
                f"[transcribe] batched pipeline unavailable ({exc}); using sequential."
            )

    return get_model(model_name).transcribe(
        audio_path,
        language=language,
        beam_size=WHISPER_BEAM,
        condition_on_previous_text=False,
        vad_filter=True,
    )


def transcript_path(video_id: str, directory: Path | None = None) -> Path:
    return (directory or TRANSCRIPT_DIR) / f"{video_id}.json"


def load_transcript(video_id: str, directory: Path | None = None) -> dict | None:
    path = transcript_path(video_id, directory)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "segments" not in data:
        return None
    return data


def cached_video_ids(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or TRANSCRIPT_DIR).glob("*.json"))


def segments_from_transcript(data: dict) -> list[Segment]:
    return [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"])
        for s in data.get("segments", [])
    ]


def _write_transcript(
    video: Video, language: str, model_name: str, segments: list[Segment]
) -> None:
    payload = {
        "video_id": video.video_id,
        "title": video.title,
        "language": language,
        "model": model_name,
        "duration": video.duration,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ],
    }
    final = transcript_path(video.video_id)
    tmp = final.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, final)


def transcribe(
    video: Video,
    force: bool = False,
    language: str | None = None,
    model_name: str = WHISPER_MODEL,
    keep_audio: bool = False,
) -> list[Segment]:
    language = language or WHISPER_LANG

    if not force:
        cached = load_transcript(video.video_id)
        if cached is not None:
            return segments_from_transcript(cached)

    audio_path = download_audio(video, force=force)
    raw_segments, info = run_whisper(str(audio_path), language, model_name)

    segments = [
        Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in raw_segments
        if s.text and s.text.strip()
    ]

    if not video.duration and getattr(info, "duration", None):
        video.duration = int(info.duration)

    _write_transcript(video, language, model_name, segments)

    if not keep_audio:
        delete_audio(video.video_id)

    return segments
