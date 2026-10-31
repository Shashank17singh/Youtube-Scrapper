from ytrag.rate_limit import RateLimiter


def test_one_client_cannot_use_the_whole_allowance() -> None:
    limiter = RateLimiter(per_client=2, overall=10, window_seconds=60)

    assert [limiter.allow("a", now=0) for _ in range(3)] == [True, True, False]
    assert limiter.allow("b", now=0)


def test_the_overall_limit_holds_when_clients_change_their_address() -> None:
    limiter = RateLimiter(per_client=1, overall=3, window_seconds=60)

    results = [limiter.allow(f"forged-{n}", now=0) for n in range(5)]

    assert results == [True, True, True, False, False]


def test_limits_recover_after_the_window() -> None:
    limiter = RateLimiter(per_client=1, overall=1, window_seconds=60)

    assert limiter.allow("a", now=0)
    assert not limiter.allow("a", now=59)
    assert limiter.allow("a", now=60)
