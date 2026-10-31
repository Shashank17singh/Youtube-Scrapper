import pytest

from ytrag import util
from ytrag.chunk import chunk_segments, is_repetitive
from ytrag.models import Chunk, Segment, Video, format_timestamp

VIDEO = Video(video_id="abc", title="Lecture 1")


def words(count: int, seed: str = "word") -> str:
    return " ".join(f"{seed}{index}" for index in range(count))


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (75, "01:15"), (735, "12:15"), (4335, "1:12:15"), (-5, "00:00")],
)
def test_timestamps_are_readable(seconds: int, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_repeated_sentences_are_detected() -> None:
    assert is_repetitive("Thank you. Thank you. Thank you. Thank you. Next.")


def test_normal_speech_is_not_repetitive() -> None:
    assert not is_repetitive("First point. Second point. Third point. Fourth point.")


def test_short_text_is_never_repetitive() -> None:
    assert not is_repetitive("Hello. Hello.")


def test_empty_segments_produce_no_chunks() -> None:
    assert chunk_segments(VIDEO, [Segment(0, 5, "   ")]) == []


def test_a_chunk_records_its_position_and_title() -> None:
    segments = [Segment(10, 20, words(40))]

    chunk = chunk_segments(
        VIDEO, segments, window_seconds=60, overlap_seconds=5, min_words=5
    )[0]

    assert chunk.chunk_id == "abc:10"
    assert (chunk.start_sec, chunk.end_sec) == (10, 20)
    assert chunk.text.startswith("Lecture 1\n\n")


def test_chunks_with_too_few_words_are_dropped() -> None:
    assert chunk_segments(VIDEO, [Segment(0, 5, "too short")], min_words=10) == []


def test_long_videos_split_into_overlapping_chunks() -> None:
    segments = [Segment(i * 10, (i + 1) * 10, words(20, f"s{i}x")) for i in range(30)]

    chunks = chunk_segments(
        VIDEO, segments, window_seconds=60, overlap_seconds=20, min_words=5
    )

    assert len(chunks) > 1
    assert chunks[1].start_sec < chunks[0].end_sec
    assert chunks[-1].end_sec == 300


def test_chunking_always_terminates_when_overlap_exceeds_the_window() -> None:
    segments = [Segment(i, i + 1, words(10, f"s{i}x")) for i in range(20)]

    assert chunk_segments(
        VIDEO, segments, window_seconds=2, overlap_seconds=50, min_words=1
    )


def test_a_chunk_survives_a_payload_round_trip() -> None:
    chunk = Chunk("abc:10", "abc", "Title", 10, 20, "body")

    assert Chunk.from_payload(chunk.to_payload()) == chunk


def test_chunk_links_never_go_before_the_start_of_the_video() -> None:
    assert Chunk("abc:0", "abc", "Title", 0, 20, "body").link_sec == 0


def test_a_chunk_id_always_maps_to_the_same_point() -> None:
    first = Chunk("abc:10", "abc", "Title", 10, 20, "one")
    second = Chunk("abc:10", "abc", "Title", 10, 20, "two")

    assert first.point_id == second.point_id


def test_retry_returns_after_a_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(util, "network_up", lambda: True)
    monkeypatch.setattr(util.time, "sleep", lambda _: None)
    results = iter([ValueError("flaky"), "done"])

    def flaky() -> str:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    assert util.with_retry(flaky, on_retry=lambda _: None) == "done"


def test_retry_raises_once_attempts_run_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(util, "network_up", lambda: True)
    monkeypatch.setattr(util.time, "sleep", lambda _: None)

    def always_fails() -> None:
        raise ValueError("broken")

    with pytest.raises(ValueError, match="broken"):
        util.with_retry(always_fails, attempts=2, on_retry=lambda _: None)


def test_rate_limits_wait_longer_than_ordinary_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(util, "network_up", lambda: True)
    waits: list[float] = []
    monkeypatch.setattr(util.time, "sleep", waits.append)

    def always_fails() -> None:
        raise ValueError("429")

    with pytest.raises(ValueError):
        util.with_retry(
            always_fails,
            attempts=2,
            base_delay=1,
            rate_limit_delay=100,
            is_rate_limit=lambda _: True,
            on_retry=lambda _: None,
        )

    assert waits == [100]
