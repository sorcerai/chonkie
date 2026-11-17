"""Chonkie: Decision Chunker.

Uses a typed decision model (Perplexity Decider via OpenRouter) to choose chunk
boundaries and, optionally, to score chunk relevance. The model returns
probabilities, never text, so every chunk is an exact slice of the input.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from chonkie.chunker.base import BaseChunker
from chonkie.genie.decision import MAX_QUESTIONS_PER_REQUEST, DecisionError, DecisionGenie
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk

logger = get_logger(__name__)

# Boundary windows ask one question per adjacent pair, so a window holds at most
# MAX_QUESTIONS_PER_REQUEST - 1 pairs, i.e. MAX_QUESTIONS_PER_REQUEST units.
_MAX_WINDOW_UNITS = MAX_QUESTIONS_PER_REQUEST
_MAX_WINDOW_QUESTIONS = MAX_QUESTIONS_PER_REQUEST - 1

_BLANK_LINES = re.compile(r"\n[ \t\r]*\n(?:[ \t\r]*\n)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")
_MD_HEADING = re.compile(r"#{1,6}\s+\S")
_NUM_HEADING = re.compile(r"\d+(?:\.\d+){0,3}\.?\s+(\S.*)")
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"}


def _is_heading(line: str) -> bool:
    """Return True for markdown headings and short title-case numbered lines."""
    line = line.strip()
    if not line:
        return False
    if _MD_HEADING.match(line):
        return True
    m = _NUM_HEADING.fullmatch(line)
    if not m or len(line) > 80 or line[-1] in ".,;:":
        return False
    words = m.group(1).split()
    if len(words) > 10 or not words[0][0].isupper():
        return False
    return all(w[0].isupper() or w.lower() in _SMALL_WORDS or len(w) <= 3 for w in words)


@dataclass
class _Unit:
    start: int
    end: int
    tokens: int
    gap_tokens: int  # tokens of the whitespace between the previous unit and this one
    heading: bool


@chunker("decision")
class DecisionChunker(BaseChunker):
    """Chunker that lets a decision model pick boundaries and score relevance.

    The text is split into candidate units (paragraphs; heading lines are their own
    units; units over ``max_unit_tokens`` are split on sentences). For each pair of
    adjacent units the model is asked whether the second starts a different topic;
    units are merged into chunks, breaking where the answer is >= ``boundary_threshold``,
    at headings, and when ``chunk_size`` would be exceeded. A heading always starts a
    new chunk, except that a heading directly after other headings (nothing but
    headings so far) stays with them, so a chunk is never a bare heading.

    Offsets: ``text[chunk.start_index:chunk.end_index] == chunk.text`` for every chunk.
    Chunks tile the text in order; the only text not covered by any chunk is the
    whitespace between chunks and at the ends of the document.

    Token counts use the chunker's tokenizer (default ``"character"``, as in
    :class:`RecursiveChunker`, so sizes are in characters unless you pass e.g. ``"gpt2"``).
    The same counts size the decider windows (``window_tokens``).

    If a decider call fails after retries (or the API key is missing), the chunker
    falls back to boundaries from headings and the size limit only, and sets
    ``chunk.metadata["decider"] = "fallback"`` plus ``"decider_fallback_reason"``.
    Relevance is then ``None``, never 0. With ``strict=True`` the failure is raised
    as :class:`DecisionError` instead. On success ``metadata["decider"] == "ok"``.

    Example:
        >>> from chonkie import DecisionChunker
        >>> chunker = DecisionChunker(
        ...     tokenizer="gpt2",
        ...     relevance_instructions="Does this passage report an experimental result?",
        ... )  # reads OPENROUTER_API_KEY
        >>> for c in chunker.chunk(paper_text):
        ...     print(c.start_index, c.end_index, c.metadata["relevance"])

    Args:
        genie: A :class:`DecisionGenie`. Defaults to ``DecisionGenie()``.
        tokenizer: Tokenizer used for all token counts.
        chunk_size: Maximum tokens per chunk.
        boundary_threshold: Break where the model's probability is at least this.
        min_chunk_tokens: Model-driven breaks are suppressed while the current chunk is
            smaller than this, and a trailing chunk smaller than this is merged into the
            previous one when it fits (unless it starts at a heading).
        max_unit_tokens: Paragraphs larger than this are split on sentences.
        window_tokens: Token budget per decider request.
        relevance_instructions: If set, each chunk is scored (``relevant_<k>`` noul
            questions) and the probability stored in ``chunk.metadata["relevance"]``.
        strict: Raise :class:`DecisionError` on decider failure instead of falling back.

    """

    def __init__(
        self,
        genie: Optional[DecisionGenie] = None,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        chunk_size: int = 2048,
        boundary_threshold: float = 0.5,
        min_chunk_tokens: int = 64,
        max_unit_tokens: int = 512,
        window_tokens: int = 6000,
        relevance_instructions: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        """Create a DecisionChunker."""
        super().__init__(tokenizer=tokenizer)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not 0.0 <= boundary_threshold <= 1.0:
            raise ValueError("boundary_threshold must be in [0, 1]")
        if min_chunk_tokens < 0 or min_chunk_tokens > chunk_size:
            raise ValueError("min_chunk_tokens must be in [0, chunk_size]")
        if max_unit_tokens <= 0 or window_tokens <= 0:
            raise ValueError("max_unit_tokens and window_tokens must be greater than 0")
        self.genie = genie or DecisionGenie()
        self.chunk_size = chunk_size
        self.boundary_threshold = boundary_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_unit_tokens = max_unit_tokens
        self.window_tokens = window_tokens
        self.relevance_instructions = relevance_instructions
        self.strict = strict

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return (
            f"DecisionChunker(chunk_size={self.chunk_size}, "
            f"boundary_threshold={self.boundary_threshold}, "
            f"min_chunk_tokens={self.min_chunk_tokens})"
        )

    # ---- units -------------------------------------------------------------------

    def _split_long(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Split an over-long span on sentences, then on whitespace/characters."""
        pieces: list[tuple[int, int]] = []
        pos = start
        for m in _SENTENCE_END.finditer(text, start, end):
            pieces.append((pos, m.start()))
            pos = m.end()
        pieces.append((pos, end))
        pieces = [(s, e) for s, e in pieces if text[s:e].strip()]

        out: list[tuple[int, int]] = []
        for s, e in pieces:
            tokens = self.tokenizer.count_tokens(text[s:e])
            if tokens <= self.max_unit_tokens:
                out.append((s, e))
                continue
            # No sentence ends: cut at whitespace near the estimated token budget.
            window = max(1, int((e - s) * self.max_unit_tokens / tokens))
            while s < e:
                cut = min(s + window, e)
                if cut < e:
                    ws = text.rfind(" ", s + 1, cut)
                    cut = ws if ws > s else cut
                if text[s:cut].strip():
                    out.append((s, cut))
                s = cut
                while s < e and text[s].isspace():
                    s += 1
        return out

    def _make_units(self, text: str) -> list[_Unit]:
        spans: list[tuple[int, int, bool]] = []  # (start, end, heading)
        para_start = 0
        bounds = [(m.start(), m.end()) for m in _BLANK_LINES.finditer(text)]
        bounds.append((len(text), len(text)))
        for b_start, b_end in bounds:
            para = (para_start, b_start)
            para_start = b_end
            if not text[para[0] : para[1]].strip():
                continue
            # Heading lines inside a paragraph become their own units.
            run_start: Optional[int] = None
            run_end = 0
            pos = para[0]
            for line in text[para[0] : para[1]].splitlines(keepends=True):
                line_start, pos = pos, pos + len(line)
                stripped = line.strip()
                if not stripped:
                    continue
                lo = line_start + (len(line) - len(line.lstrip()))
                hi = line_start + len(line.rstrip())
                if _is_heading(stripped):
                    if run_start is not None:
                        spans.append((run_start, run_end, False))
                        run_start = None
                    spans.append((lo, hi, True))
                else:
                    if run_start is None:
                        run_start = lo
                    run_end = hi
            if run_start is not None:
                spans.append((run_start, run_end, False))

        units: list[_Unit] = []
        prev_end = 0
        for s, e, heading in spans:
            parts = [(s, e)]
            if not heading and self.tokenizer.count_tokens(text[s:e]) > self.max_unit_tokens:
                parts = self._split_long(text, s, e)
            for ps, pe in parts:
                gap = self.tokenizer.count_tokens(text[prev_end:ps]) if units else 0
                units.append(_Unit(ps, pe, self.tokenizer.count_tokens(text[ps:pe]), gap, heading))
                prev_end = pe
        return units

    # ---- decider calls -----------------------------------------------------------

    def _boundary_windows(self, units: list[_Unit]) -> list[tuple[int, int]]:
        """Windows [a, b] of consecutive units; consecutive windows share one unit."""
        windows: list[tuple[int, int]] = []
        a, n = 0, len(units)
        while a < n - 1:
            b, total = a, units[a].tokens
            while (
                b + 1 < n
                and b - a < _MAX_WINDOW_QUESTIONS
                and (b == a or total + units[b + 1].tokens <= self.window_tokens)
            ):
                b += 1
                total += units[b].tokens
            windows.append((a, b))
            a = b
        return windows

    def _decide_boundaries(self, text: str, units: list[_Unit]) -> dict[int, float]:
        """Return p(break before unit i) for every pair the model was asked about."""
        probs: dict[int, float] = {}
        for a, b in self._boundary_windows(units):
            questions: dict[str, str] = {}
            for i in range(a + 1, b + 1):
                # A heading starts a chunk anyway, and a heading stays with its body.
                if units[i].heading or units[i - 1].heading:
                    continue
                questions[f"break_before_{i}"] = (
                    f"Does paragraph {i} start a different topic or section from paragraph {i - 1}?"
                )
            if not questions:
                continue
            state = [
                {"id": i, "text": text[units[i].start : units[i].end]} for i in range(a, b + 1)
            ]
            answers = self.genie.decide(state, questions)
            for name, p in answers.items():
                probs[int(name.rsplit("_", 1)[1])] = p
        return probs

    def _decide_relevance(self, chunks: list[Chunk]) -> list[float]:
        assert self.relevance_instructions is not None
        scores: list[float] = [0.0] * len(chunks)
        a = 0
        while a < len(chunks):
            b, total = a + 1, chunks[a].token_count
            while (
                b < len(chunks)
                and b - a < _MAX_WINDOW_QUESTIONS
                and total + chunks[b].token_count <= self.window_tokens
            ):
                total += chunks[b].token_count
                b += 1
            state: Any = [{"id": k, "text": chunks[k].text} for k in range(a, b)]
            questions = {f"relevant_{k}": self.relevance_instructions for k in range(a, b)}
            answers = self.genie.decide(state, questions)
            for k in range(a, b):
                scores[k] = answers[f"relevant_{k}"]
            a = b
        return scores

    # ---- merging -----------------------------------------------------------------

    def _merge(self, units: list[_Unit], probs: dict[int, float]) -> list[tuple[int, int]]:
        """Merge units into inclusive index ranges."""
        ranges: list[tuple[int, int]] = []
        first, tokens, only_headings = 0, units[0].tokens, units[0].heading
        for i in range(1, len(units)):
            u = units[i]
            if only_headings:
                brk = False
            else:
                p = probs.get(i)
                brk = (
                    u.heading
                    or tokens + u.gap_tokens + u.tokens > self.chunk_size
                    or (
                        p is not None
                        and p >= self.boundary_threshold
                        and tokens >= self.min_chunk_tokens
                    )
                )
            if brk:
                ranges.append((first, i - 1))
                first, tokens, only_headings = i, u.tokens, u.heading
            else:
                tokens += u.gap_tokens + u.tokens
                only_headings = only_headings and u.heading
        ranges.append((first, len(units) - 1))

        # Fold a too-small trailing chunk into its predecessor when it fits.
        if len(ranges) > 1 and self.min_chunk_tokens:
            f, last = ranges[-1]
            pf, _ = ranges[-2]
            last_tokens = sum(x.tokens for x in units[f : last + 1])
            merged = (
                sum(x.tokens + x.gap_tokens for x in units[pf : last + 1]) - units[pf].gap_tokens
            )
            if (
                last_tokens < self.min_chunk_tokens
                and not units[f].heading
                and (merged <= self.chunk_size)
            ):
                ranges[-2:] = [(pf, last)]
        return ranges

    # ---- public ------------------------------------------------------------------

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk ``text``; see the class docstring for guarantees."""
        units = self._make_units(text)
        if not units:
            return []

        probs: dict[int, float] = {}
        reason: Optional[str] = None
        try:
            probs = self._decide_boundaries(text, units)
        except DecisionError as e:
            if self.strict:
                raise
            reason = str(e)
            probs = {}
            logger.warning(
                "DecisionChunker falling back to heading/size boundaries", reason=reason
            )

        chunks = []
        for first, last in self._merge(units, probs):
            start, end = units[first].start, units[last].end
            body = text[start:end]
            chunks.append(
                Chunk(
                    text=body,
                    start_index=start,
                    end_index=end,
                    token_count=self.tokenizer.count_tokens(body),
                )
            )

        relevance: Optional[list[float]] = None
        if self.relevance_instructions is not None and reason is None:
            try:
                relevance = self._decide_relevance(chunks)
            except DecisionError as e:
                if self.strict:
                    raise
                reason = f"relevance: {e}"
                logger.warning("DecisionChunker relevance scoring failed", reason=reason)

        for k, c in enumerate(chunks):
            c.metadata["decider"] = "fallback" if reason else "ok"
            if reason:
                c.metadata["decider_fallback_reason"] = reason
            if self.relevance_instructions is not None:
                c.metadata["relevance"] = relevance[k] if relevance is not None else None
        return chunks
