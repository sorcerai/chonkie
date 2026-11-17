"""Tests for DecisionChunker and DecisionGenie. No network, no API key."""

import json
import os
from typing import Any, Callable, Optional

import httpx
import pytest

from chonkie import DecisionChunker, DecisionError, DecisionGenie
from chonkie.genie import decision as decision_module


@pytest.fixture(autouse=True)
def _reset_pacer(monkeypatch: pytest.MonkeyPatch) -> None:
    decision_module._PACER.last = None
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


class FakeDecider:
    """An httpx.MockTransport handler that records requests and answers noul questions."""

    def __init__(self, prob: Callable[[str], float] = lambda name: 0.0) -> None:
        """Answer every question with ``prob(name)``."""
        self.prob = prob
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(request.headers)
        answers = {name: {"noul": self.prob(name)} for name in body["questions"]}
        return httpx.Response(200, json={"answers": answers, "usage": {}})

    @property
    def question_names(self) -> list[str]:
        return [n for r in self.requests for n in r["questions"]]


def make_genie(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> DecisionGenie:
    kw.setdefault("api_key", "test-key")
    kw.setdefault("min_interval", 0.0)
    kw.setdefault("sleep", lambda s: None)
    return DecisionGenie(client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def paragraphs(n: int, width: int = 80) -> str:
    return "\n\n".join(f"P{i} " + ("word " * (width // 5)).strip() + "." for i in range(n))


def assert_invariants(text: str, chunks: list) -> None:
    pos = 0
    for c in chunks:
        assert text[c.start_index : c.end_index] == c.text
        assert c.start_index >= pos
        assert text[pos : c.start_index].strip() == "", "gap between chunks must be whitespace"
        pos = c.end_index
    assert text[pos:].strip() == ""


# ---- offsets and tiling ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        paragraphs(12),
        "  \n\n第一段。これは日本語の文章です。\n\n  第二段落は中文です。\n\n第三段落。  \n",
        "Intro 🎉 with emoji 👩‍👩‍👧 text.\n\nSecond 🚀 paragraph 日本.\n\n\n\nThird ✨.",
        "# Title\n\nbody é ü\n\n## Sub\n\nmore 漢字 body",
    ],
)
def test_offsets_and_tiling(text: str) -> None:
    chunker = DecisionChunker(
        genie=make_genie(FakeDecider(lambda n: 1.0)), min_chunk_tokens=0, chunk_size=60
    )
    chunks = chunker.chunk(text)
    assert chunks
    assert_invariants(text, chunks)


def test_offsets_when_sentences_split_without_whitespace() -> None:
    text = "第一句。第二句。第三句。第四句。" * 30 + "\n\nend."
    chunker = DecisionChunker(
        genie=make_genie(FakeDecider()), max_unit_tokens=20, chunk_size=40, min_chunk_tokens=0
    )
    chunks = chunker.chunk(text)
    assert_invariants(text, chunks)
    assert all(c.token_count <= 40 for c in chunks)


def test_empty_and_whitespace_text() -> None:
    fake = FakeDecider()
    chunker = DecisionChunker(genie=make_genie(fake))
    assert chunker.chunk("") == []
    assert chunker.chunk(" \n\n \n") == []
    assert fake.requests == []


def test_single_unit_makes_no_request() -> None:
    fake = FakeDecider()
    chunks = DecisionChunker(genie=make_genie(fake)).chunk("Just one paragraph.")
    assert len(chunks) == 1 and fake.requests == []


# ---- headings, thresholds, size --------------------------------------------------


def test_headings_are_hard_boundaries() -> None:
    text = (
        "# Intro\n\nalpha body text here.\n\nbeta body text here.\n\n"
        "3 Results\nresult text starts here.\n\n3.1 Setup\n\nsetup text here."
    )
    fake = FakeDecider(lambda n: 0.0)  # model never breaks
    chunker = DecisionChunker(genie=make_genie(fake), min_chunk_tokens=0, chunk_size=10_000)
    chunks = chunker.chunk(text)
    assert [c.text.split("\n")[0] for c in chunks] == ["# Intro", "3 Results", "3.1 Setup"]
    assert "alpha body text here.\n\nbeta body text here." in chunks[0].text
    assert_invariants(text, chunks)
    # Pairs touching a heading are never asked: only alpha->beta and result text pairs.
    assert fake.question_names == ["break_before_2"]


def test_consecutive_headings_stay_with_their_body() -> None:
    text = "# Title\n\n## Abstract\n\nThe abstract body.\n\n## Next\n\nOther body."
    chunks = DecisionChunker(
        genie=make_genie(FakeDecider()), min_chunk_tokens=0, chunk_size=10_000
    ).chunk(text)
    assert [c.text for c in chunks] == [
        "# Title\n\n## Abstract\n\nThe abstract body.",
        "## Next\n\nOther body.",
    ]


def test_not_every_numbered_line_is_a_heading() -> None:
    text = "alpha one.\n\n3 we compared the methods in detail here\n\nthird."
    chunks = DecisionChunker(
        genie=make_genie(FakeDecider()), min_chunk_tokens=0, chunk_size=10_000
    ).chunk(text)
    assert len(chunks) == 1


def test_threshold_behaviour() -> None:
    text = paragraphs(4)

    def run(threshold: float, p: float) -> int:
        chunker = DecisionChunker(
            genie=make_genie(FakeDecider(lambda n: p)),
            boundary_threshold=threshold,
            min_chunk_tokens=0,
            chunk_size=10_000,
        )
        return len(chunker.chunk(text))

    assert run(0.5, 0.49) == 1
    assert run(0.5, 0.5) == 4  # >= threshold breaks
    assert run(0.8, 0.7) == 1
    assert run(0.6, 0.7) == 4


def test_min_chunk_tokens_suppresses_model_breaks() -> None:
    text = paragraphs(4, width=40)  # each paragraph ~45 chars
    chunker = DecisionChunker(
        genie=make_genie(FakeDecider(lambda n: 1.0)), min_chunk_tokens=100, chunk_size=10_000
    )
    chunks = chunker.chunk(text)
    assert len(chunks) < 4
    assert all(c.token_count >= 100 for c in chunks[:-1])
    assert_invariants(text, chunks)


def test_chunk_size_forces_breaks() -> None:
    text = paragraphs(10)
    chunker = DecisionChunker(
        genie=make_genie(FakeDecider(lambda n: 0.0)), chunk_size=200, min_chunk_tokens=0
    )
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(c.token_count <= 200 for c in chunks)
    assert_invariants(text, chunks)


# ---- batching --------------------------------------------------------------------


def test_window_batching_asks_each_pair_once_within_127_questions() -> None:
    n = 300
    text = paragraphs(n, width=10)  # tiny units: the 128-unit cap is what binds
    fake = FakeDecider()
    DecisionChunker(genie=make_genie(fake), window_tokens=10**9).chunk(text)
    assert all(len(r["questions"]) <= 127 for r in fake.requests)
    assert len(fake.requests) == 3  # 299 pairs / 127
    names = fake.question_names
    assert sorted(names) == sorted(f"break_before_{i}" for i in range(1, n))
    assert len(names) == len(set(names))
    # Windows overlap by one unit: the last unit of a window opens the next one.
    first_ids = [r["state"][0]["id"] for r in fake.requests]
    last_ids = [r["state"][-1]["id"] for r in fake.requests]
    assert first_ids[1:] == last_ids[:-1]


def test_window_token_budget_is_respected() -> None:
    text = paragraphs(40, width=100)
    fake = FakeDecider()
    DecisionChunker(genie=make_genie(fake), window_tokens=1000).chunk(text)
    assert len(fake.requests) > 1
    for r in fake.requests:
        assert sum(len(u["text"]) for u in r["state"][1:]) <= 1000
    assert sorted(fake.question_names) == sorted(f"break_before_{i}" for i in range(1, 40))


def test_request_shape() -> None:
    fake = FakeDecider()
    genie = make_genie(fake)
    DecisionChunker(genie=genie).chunk("first para.\n\nsecond para.")
    req = fake.requests[0]
    assert req["model"] == "perplexity/pplx-decider-v1.3:free"
    assert req["state"] == [{"id": 0, "text": "first para."}, {"id": 1, "text": "second para."}]
    assert req["questions"] == {
        "break_before_1": {
            "type": "noul",
            "instructions": "Does paragraph 1 start a different topic or section from paragraph 0?",
        }
    }
    assert fake.headers[0]["authorization"] == "Bearer test-key"


# ---- relevance -------------------------------------------------------------------


def test_relevance_metadata() -> None:
    text = paragraphs(6)

    def prob(name: str) -> float:
        if name.startswith("relevant_"):
            return round(0.1 * (int(name.split("_")[1]) + 1), 1)
        return 1.0

    fake = FakeDecider(prob)
    chunker = DecisionChunker(
        genie=make_genie(fake),
        relevance_instructions="Is this about words?",
        min_chunk_tokens=0,
        chunk_size=10_000,
    )
    chunks = chunker.chunk(text)
    assert [c.metadata["relevance"] for c in chunks] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert all(c.metadata["decider"] == "ok" for c in chunks)
    rel_reqs = [r for r in fake.requests if any(k.startswith("relevant_") for k in r["questions"])]
    assert rel_reqs[0]["questions"]["relevant_0"]["instructions"] == "Is this about words?"
    assert rel_reqs[0]["state"][0]["text"] == chunks[0].text


def test_no_relevance_key_without_instructions() -> None:
    chunks = DecisionChunker(genie=make_genie(FakeDecider())).chunk(paragraphs(3))
    assert all("relevance" not in c.metadata for c in chunks)


def test_relevance_batches_respect_window() -> None:
    text = paragraphs(30, width=100)
    fake = FakeDecider(lambda n: 0.5)
    chunker = DecisionChunker(
        genie=make_genie(fake),
        relevance_instructions="q",
        boundary_threshold=0.0,
        min_chunk_tokens=0,
        window_tokens=500,
    )
    chunks = chunker.chunk(text)
    rel = [
        r
        for r in fake.requests
        if "relevant_0" in r["questions"] or "relevant_1" in r["questions"]
    ]
    asked = [n for r in fake.requests for n in r["questions"] if n.startswith("relevant_")]
    assert sorted(asked) == sorted(f"relevant_{k}" for k in range(len(chunks)))
    assert rel


# ---- fallbacks -------------------------------------------------------------------

FALLBACK_TEXT = "# A\n\nalpha text here.\n\nbeta text here.\n\n# B\n\ngamma text here."


def _assert_fallback(chunks: list, relevance: bool) -> None:
    assert [c.text.split("\n")[0] for c in chunks] == ["# A", "# B"]  # headings still split
    assert_invariants(FALLBACK_TEXT, chunks)
    for c in chunks:
        assert c.metadata["decider"] == "fallback"
        assert c.metadata["decider_fallback_reason"]
        if relevance:
            assert c.metadata["relevance"] is None
        else:
            assert "relevance" not in c.metadata


@pytest.mark.parametrize("relevance", [False, True])
def test_fallback_missing_key(relevance: bool) -> None:
    fake = FakeDecider()
    chunker = DecisionChunker(
        genie=make_genie(fake, api_key=None),
        relevance_instructions="q" if relevance else None,
        min_chunk_tokens=0,
        chunk_size=10_000,
    )
    _assert_fallback(chunker.chunk(FALLBACK_TEXT), relevance)
    assert fake.requests == []


def _status_handler(status: int, calls: list) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"error": "x"})

    return handler


@pytest.mark.parametrize("status", [429, 500, 503])
@pytest.mark.parametrize("relevance", [False, True])
def test_fallback_retries_exhausted(status: int, relevance: bool) -> None:
    calls: list = []
    chunker = DecisionChunker(
        genie=make_genie(_status_handler(status, calls), max_attempts=3),
        relevance_instructions="q" if relevance else None,
        min_chunk_tokens=0,
        chunk_size=10_000,
    )
    _assert_fallback(chunker.chunk(FALLBACK_TEXT), relevance)
    assert len(calls) == 3


def test_retry_then_success() -> None:
    responses = [httpx.Response(429), httpx.Response(502)]
    ok = FakeDecider()

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0) if responses else ok(request)

    sleeps: list[float] = []
    genie = make_genie(handler, sleep=sleeps.append)
    assert genie.decide("s", {"q": "x"}) == {"q": 0.0}
    assert len(ok.requests) == 1
    assert len(sleeps) == 2 and sleeps[1] >= sleeps[0]  # exponential backoff


def test_fallback_on_client_error_is_not_retried() -> None:
    calls: list = []
    chunker = DecisionChunker(genie=make_genie(_status_handler(404, calls)))
    chunks = chunker.chunk(FALLBACK_TEXT)
    assert len(calls) == 1
    assert chunks[0].metadata["decider"] == "fallback"


def test_fallback_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    chunker = DecisionChunker(genie=make_genie(handler, max_attempts=2))
    assert chunker.chunk(FALLBACK_TEXT)[0].metadata["decider"] == "fallback"


@pytest.mark.parametrize(
    "payload",
    [
        {"answers": {}},
        {"answers": {"break_before_2": {"noul": "high"}}},
        {"answers": {"break_before_2": {"noul": 1.5}}},
        {"answers": {"break_before_2": {"noul": True}}},
        {"answers": {"break_before_2": {"choice": "a"}}},
        {"nope": 1},
        [],
    ],
)
@pytest.mark.parametrize("relevance", [False, True])
def test_fallback_malformed_answer(payload: Any, relevance: bool) -> None:
    chunker = DecisionChunker(
        genie=make_genie(lambda req: httpx.Response(200, json=payload)),
        relevance_instructions="q" if relevance else None,
        min_chunk_tokens=0,
        chunk_size=10_000,
    )
    _assert_fallback(chunker.chunk(FALLBACK_TEXT), relevance)


def test_fallback_non_json_body() -> None:
    chunker = DecisionChunker(genie=make_genie(lambda req: httpx.Response(200, text="<html>")))
    assert chunker.chunk(FALLBACK_TEXT)[0].metadata["decider"] == "fallback"


def test_relevance_failure_keeps_model_boundaries_and_none_relevance() -> None:
    text = paragraphs(4)
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:  # the boundary request
            body = json.loads(request.content)
            ans = {n: {"noul": 1.0} for n in body["questions"]}
            return httpx.Response(200, json={"answers": ans})
        return httpx.Response(500)

    chunker = DecisionChunker(
        genie=make_genie(handler, max_attempts=2),
        relevance_instructions="q",
        min_chunk_tokens=0,
        chunk_size=10_000,
        window_tokens=10_000,
    )
    chunks = chunker.chunk(text)
    assert len(chunks) == 4  # model boundaries kept
    assert all(c.metadata["relevance"] is None for c in chunks)
    assert all(c.metadata["decider"] == "fallback" for c in chunks)
    assert all(c.metadata["decider_fallback_reason"].startswith("relevance:") for c in chunks)


def test_strict_raises() -> None:
    chunker = DecisionChunker(genie=make_genie(FakeDecider(), api_key=None), strict=True)
    with pytest.raises(DecisionError):
        chunker.chunk(FALLBACK_TEXT)
    chunker = DecisionChunker(
        genie=make_genie(_status_handler(500, []), max_attempts=2), strict=True
    )
    with pytest.raises(DecisionError):
        chunker.chunk(FALLBACK_TEXT)


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    fake = FakeDecider()
    genie = DecisionGenie(
        client=httpx.Client(transport=httpx.MockTransport(fake)),
        min_interval=0.0,
        sleep=lambda s: None,
    )
    genie.decide("s", {"q": "x"})
    assert fake.headers[0]["authorization"] == "Bearer env-key"


# ---- cache -----------------------------------------------------------------------


def test_cache_hit_makes_no_http_call() -> None:
    store: dict[str, dict[str, float]] = {}
    fake = FakeDecider(lambda n: 0.25)
    genie = make_genie(fake, cache_get=store.get, cache_set=store.__setitem__)
    assert genie.decide("state", {"q": "x"}) == {"q": 0.25}
    assert len(fake.requests) == 1 and len(store) == 1

    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP call on cache hit")

    genie2 = make_genie(boom, api_key=None, cache_get=store.get, cache_set=store.__setitem__)
    assert genie2.decide("state", {"q": "x"}) == {"q": 0.25}
    # A different state, question or model is a different key.
    assert genie.cache_key("state2", {"q": "x"}) not in store
    other = make_genie(fake, model="other")
    assert other.cache_key("state", {"q": "x"}) != genie.cache_key("state", {"q": "x"})


def test_cache_makes_second_chunking_free() -> None:
    store: dict[str, dict[str, float]] = {}
    text = paragraphs(10)
    fake = FakeDecider(lambda n: 0.5)
    first = DecisionChunker(
        genie=make_genie(fake, cache_get=store.get, cache_set=store.__setitem__),
        relevance_instructions="q",
    ).chunk(text)
    n_requests = len(fake.requests)
    assert n_requests > 0
    second = DecisionChunker(
        genie=make_genie(fake, cache_get=store.get, cache_set=store.__setitem__),
        relevance_instructions="q",
    ).chunk(text)
    assert len(fake.requests) == n_requests
    assert [(c.start_index, c.end_index) for c in first] == [
        (c.start_index, c.end_index) for c in second
    ]


def test_failures_are_not_cached() -> None:
    store: dict[str, dict[str, float]] = {}
    genie = make_genie(
        _status_handler(500, []), max_attempts=1, cache_get=store.get, cache_set=store.__setitem__
    )
    with pytest.raises(DecisionError):
        genie.decide("s", {"q": "x"})
    assert store == {}


# ---- pacing ----------------------------------------------------------------------


class FakeClock:
    """A fake monotonic clock whose sleep advances time."""

    def __init__(self) -> None:
        """Start at an arbitrary time."""
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_pacing_enforces_min_interval() -> None:
    fc = FakeClock()
    genie = make_genie(FakeDecider(), min_interval=3.1, clock=fc.clock, sleep=fc.sleep)
    genie.decide("a", {"q": "x"})
    assert fc.sleeps == []  # first call is immediate
    fc.now += 1.0
    genie.decide("b", {"q": "x"})
    assert fc.sleeps == [pytest.approx(2.1)]
    fc.now += 10.0  # already past the interval: no wait
    genie.decide("c", {"q": "x"})
    assert fc.sleeps == [pytest.approx(2.1)]


def test_pacing_is_process_wide_and_counts_retries() -> None:
    fc = FakeClock()
    responses = [httpx.Response(429)]
    ok = FakeDecider()

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0) if responses else ok(request)

    g1 = make_genie(handler, min_interval=3.1, clock=fc.clock, sleep=fc.sleep)
    g2 = make_genie(ok, min_interval=3.1, clock=fc.clock, sleep=fc.sleep)
    g1.decide("a", {"q": "x"})  # 429 then retry: every attempt is paced
    stamps_after_g1 = fc.now
    g2.decide("b", {"q": "x"})  # a different instance still waits
    assert fc.now >= stamps_after_g1 + 3.1 - 1e-9


def test_cache_hit_is_not_paced() -> None:
    fc = FakeClock()
    store = {"k": {"q": 0.1}}
    genie = make_genie(
        FakeDecider(), min_interval=3.1, clock=fc.clock, sleep=fc.sleep, cache_get=store.get
    )
    genie.cache_key = lambda state, questions: "k"  # type: ignore[method-assign]
    genie.decide("a", {"q": "x"})
    genie.decide("a", {"q": "x"})
    assert fc.sleeps == []


# ---- request-count estimate ------------------------------------------------------


def test_request_count_for_100_paragraphs_at_defaults() -> None:
    fake = FakeDecider()
    DecisionChunker(genie=make_genie(fake)).chunk(paragraphs(100, width=200))
    # Default tokenizer is "character": 100 x ~200 chars = ~20k, window 6000 -> a few windows.
    assert 1 <= len(fake.requests) <= 8


# ---- opt-in live test ------------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("OPENROUTER_API_KEY") and os.environ.get("CHONKIE_LIVE") == "1"),
    reason="live test: set OPENROUTER_API_KEY and CHONKIE_LIVE=1",
)
def test_live_openrouter_decider() -> None:
    text = (
        "# Cats\n\nCats are small carnivorous mammals kept as pets.\n\n"
        "They sleep for most of the day.\n\n"
        "# Quantum computing\n\nQubits can be in superposition of states.\n\n"
        "Error correction is a central challenge for quantum hardware."
    )
    chunker = DecisionChunker(
        relevance_instructions="Is this passage about quantum computing?", min_chunk_tokens=0
    )
    chunks = chunker.chunk(text)
    assert_invariants(text, chunks)
    assert all(c.metadata["decider"] == "ok" for c in chunks), chunks[0].metadata
    rel: Optional[float] = chunks[-1].metadata["relevance"]
    assert rel is not None and 0.0 <= rel <= 1.0
