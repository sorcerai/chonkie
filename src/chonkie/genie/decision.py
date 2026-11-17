"""DecisionGenie: a client for typed decision models (OpenRouter ``/decisions``).

A decision model answers questions with probabilities and returns no text. This
client is not a chat genie, so it does not subclass :class:`BaseGenie`; it has a
single ``decide(state, questions)`` method.
"""

import hashlib
import json
import math
import os
import threading
import time
from typing import Any, Callable, Mapping, Optional

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/alpha"
DEFAULT_MODEL = "perplexity/pplx-decider-v1.3:free"
MAX_QUESTIONS_PER_REQUEST = 128


class DecisionError(RuntimeError):
    """Raised when a decision request fails (after retries) or its answer is malformed."""


class _RetryableError(DecisionError):
    """A failure worth retrying (429, 5xx, transport errors)."""


class _Pacer:
    """Process-wide minimum interval between decision requests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last: Optional[float] = None


_PACER = _Pacer()


class DecisionGenie:
    """Client for Perplexity Decider models served by OpenRouter's decisions API.

    Every request is paced process-wide (``min_interval`` seconds between the
    start of consecutive requests, across all instances), and 429/5xx responses
    and transport errors are retried with exponential backoff.

    Args:
        api_key: OpenRouter API key. Defaults to the ``OPENROUTER_API_KEY`` env var.
            A missing key is not an error until :meth:`decide` makes a request.
        model: The decision model id.
        base_url: Base URL; requests go to ``<base_url>/decisions``.
        timeout: Per-request timeout in seconds.
        min_interval: Minimum seconds between requests, process-wide. The free tier
            allows 20 requests/min, so the default is 3.1.
        max_attempts: Total attempts per request, including the first.
        cache_get: Optional ``key -> dict[str, float] | None`` lookup.
        cache_set: Optional ``(key, dict[str, float]) -> None`` store. Keys are
            :meth:`cache_key` values (sha256 of state + questions + model).
        client: Optional ``httpx.Client`` (e.g. with a custom transport).
        clock: Monotonic clock, injectable for tests.
        sleep: Sleep function, injectable for tests.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        min_interval: float = 3.1,
        max_attempts: int = 5,
        cache_get: Optional[Callable[[str], Optional[dict[str, float]]]] = None,
        cache_set: Optional[Callable[[str, dict[str, float]], None]] = None,
        client: Optional[httpx.Client] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize the DecisionGenie."""
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.cache_get = cache_get
        self.cache_set = cache_set
        self._client = client
        self._clock = clock
        self._sleep = sleep

    def cache_key(self, state: Any, questions: Mapping[str, Any]) -> str:
        """Return the cache key: sha256 of state, questions and model."""
        payload = json.dumps(
            {"state": state, "questions": questions, "model": self.model},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def decide(self, state: Any, questions: Mapping[str, str]) -> dict[str, float]:
        """Ask noul (probability) questions about ``state``.

        Args:
            state: A string or JSON-serializable value the questions are about.
            questions: Mapping of question name to instructions (at most 128).

        Returns:
            Mapping of question name to probability in [0, 1].

        Raises:
            DecisionError: On a missing key, a failed request after retries, or a
                malformed answer.
            ValueError: If there are no questions or more than 128.

        """
        if not questions:
            raise ValueError("questions must not be empty")
        if len(questions) > MAX_QUESTIONS_PER_REQUEST:
            raise ValueError(f"at most {MAX_QUESTIONS_PER_REQUEST} questions per request")

        body_questions = {
            name: {"type": "noul", "instructions": q} for name, q in questions.items()
        }
        key = self.cache_key(state, body_questions)
        if self.cache_get is not None:
            cached = self.cache_get(key)
            if cached is not None:
                return dict(cached)

        if not self.api_key:
            raise DecisionError(
                "DecisionGenie requires an API key: pass `api_key` or set OPENROUTER_API_KEY."
            )

        body = {"model": self.model, "state": state, "questions": body_questions}
        retrying = Retrying(
            retry=retry_if_exception_type(_RetryableError),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, max=30),
            sleep=self._sleep,
            reraise=True,
        )
        result = retrying(self._request_once, body, list(questions))

        if self.cache_set is not None:
            self.cache_set(key, result)
        return result

    def _pace(self) -> None:
        with _PACER.lock:
            now = self._clock()
            if _PACER.last is not None:
                wait = _PACER.last + self.min_interval - now
                if wait > 0:
                    self._sleep(wait)
                    now = now + wait
            _PACER.last = now

    def _request_once(self, body: dict[str, Any], names: list[str]) -> dict[str, float]:
        self._pace()
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{self.base_url}/decisions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except httpx.TransportError as e:
            raise _RetryableError(f"transport error: {e}") from e
        finally:
            if self._client is None:
                client.close()

        status = response.status_code
        if status == 429 or status >= 500:
            raise _RetryableError(f"HTTP {status}")
        if status >= 400:
            raise DecisionError(f"HTTP {status}")
        return self._parse(response, names)

    @staticmethod
    def _parse(response: httpx.Response, names: list[str]) -> dict[str, float]:
        try:
            answers = response.json()["answers"]
            out: dict[str, float] = {}
            for name in names:
                value = answers[name]["noul"]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name}: noul is not a number")
                value = float(value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name}: noul {value} outside [0, 1]")
                out[name] = value
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise DecisionError(f"malformed decision answer: {e}") from e
        return out
