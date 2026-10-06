"""
Integration with yt-dlp to extract playlist metadata and reliably download
audio tracks, handling bot challenges and rate limiting gracefully.
"""

from pathlib import Path

from ytrag.config import (
    AUDIO_DIR,
    COOKIES_FILE,
    COOKIES_FROM_BROWSER,
    DOWNLOAD_SLEEP_MAX,
    DOWNLOAD_SLEEP_MIN,
)
from ytrag.models import Video
from ytrag.util import with_retry

_BOT_MARKERS = (
    "not a bot",
    "sign in to confirm",
    "too many requests",
    "http error 429",
)


def _is_bot_challenge(exc: Exception) -> bool:
    return any(marker in str(exc).lower() for marker in _BOT_MARKERS)


def _cookie_opts() -> dict:
    if COOKIES_FILE:
        return {"cookiefile": COOKIES_FILE}
    if COOKIES_FROM_BROWSER:
        return {"cookiesfrombrowser": (COOKIES_FROM_BROWSER,)}
    return {}


_QUIET = {"quiet": True, "no_warnings": True, "noprogress": True}


def list_playlist(playlist_url: str) -> list[Video]:
    import yt_dlp

    opts = {**_QUIET, "extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    entries = info.get("entries") or [info]
    return [
        Video(
            video_id=e["id"],
            title=e.get("title") or e["id"],
            duration=int(e.get("duration") or 0),
        )
        for e in entries
        if e and e.get("id")
    ]


def find_audio(video_id: str) -> Path | None:
    for path in sorted(AUDIO_DIR.glob(f"{video_id}.*")):
        if path.suffix not in {".part", ".ytdl", ".tmp"}:
            return path
    return None


def download_audio(video: Video, force: bool = False) -> Path:
    existing = find_audio(video.video_id)
    if existing and not force:
        return existing

    opts = {
        **_QUIET,
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / f"{video.video_id}.%(ext)s"),
        "overwrites": True,  # a retry must replace the partial file, not skip it
        "sleep_interval": DOWNLOAD_SLEEP_MIN,
        "max_sleep_interval": DOWNLOAD_SLEEP_MAX,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    opts.update(_cookie_opts())

    def _download():
        import yt_dlp

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video.url])

    with_retry(
        _download,
        attempts=8,
        label=f"download {video.video_id}",
        is_rate_limit=_is_bot_challenge,
        rate_limit_delay=600,
    )

    path = find_audio(video.video_id)
    if path is None:
        raise RuntimeError(
            f"yt-dlp reported success but no audio file for {video.video_id}"
        )
    return path


def delete_audio(video_id: str) -> None:
    for path in AUDIO_DIR.glob(f"{video_id}.*"):
        path.unlink(missing_ok=True)
