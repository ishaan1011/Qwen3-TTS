# coding=utf-8
"""
Sentence/phrase boundary detection for streaming LLM token feeds.

`split_first_sentence(text)` returns (first_sentence, remainder) when a
hard sentence boundary is present in `text`, otherwise (None, text).

Hard boundary = `.`, `!`, or `?` followed by whitespace, EXCEPT when:
  - the preceding word is a known abbreviation (Mr., Dr., e.g., etc.)
  - the period is part of a decimal number (3.14)
  - the punctuation is part of an ellipsis run (...)
"""
from __future__ import annotations

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "jr", "sr", "st",
    "vs", "etc", "fig", "no", "vol", "co", "ltd",
    "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
}

_SENT_END = ".!?"


def find_sentence_end(text: str) -> int:
    """
    Return the index just past a valid sentence end (i.e., position of the
    first character that belongs to the *next* sentence, after any
    whitespace), or -1 if none.
    """
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c not in _SENT_END:
            i += 1
            continue

        # Run through any consecutive sentence-end chars (handles "...", "!?")
        run_start = i
        run_end = i
        while run_end < n and text[run_end] in _SENT_END:
            run_end += 1

        # Must be followed by whitespace OR be at EOF (don't emit at EOF here;
        # we want a boundary the buffer is committed to)
        if run_end >= n or not text[run_end].isspace():
            i = run_end
            continue

        # Check abbreviation: word ending at the FIRST punct char (run_start)
        word_start = run_start
        while word_start > 0 and (text[word_start - 1].isalpha() or text[word_start - 1] == "."):
            word_start -= 1
        word = text[word_start:run_start].lower().rstrip(".")
        if word in ABBREVIATIONS:
            i = run_end
            continue

        # Check decimal: digit before AND digit after a single '.'
        if (run_end - run_start == 1
            and word_start < run_start
            and text[run_start - 1].isdigit()):
            # Look past the trailing space — but we already required whitespace
            # at run_end, so the next char IS whitespace, not a digit. So this
            # period genuinely ends a sentence. (Decimal like "3.14" has no
            # space after the dot, which means run_end check above already
            # rejected it.)
            pass

        # Skip any whitespace after the punctuation
        j = run_end
        while j < n and text[j].isspace():
            j += 1
        return j

    return -1


def split_first_sentence(text: str) -> tuple[str | None, str]:
    """
    If `text` contains a complete sentence followed by whitespace, return
    (sentence_text_with_trailing_whitespace_stripped, remainder).
    Otherwise return (None, text).
    """
    end = find_sentence_end(text)
    if end == -1:
        return None, text
    return text[:end].rstrip(), text[end:]


def drain_sentences(text: str) -> tuple[list[str], str]:
    """Repeatedly split off complete sentences from the front; return the list and the leftover buffer."""
    out: list[str] = []
    rest = text
    while True:
        sent, rest = split_first_sentence(rest)
        if sent is None:
            return out, rest
        if sent:
            out.append(sent)


# ---------------------------------------------------------------------------
# Schedule-based forced emission
#
# When a hard sentence boundary doesn't arrive in time, force-emit at a soft
# boundary (clause or word) once the buffer crosses the next scheduled
# character threshold. Lifted from ElevenLabs' chunk_length_schedule pattern.
# ---------------------------------------------------------------------------

# Cumulative buffer size at which to force-emit chunk N (1-indexed).
# Defaults to the same balanced schedule ElevenLabs publishes.
DEFAULT_FORCE_SCHEDULE: list[int] = [150, 200, 260]
DEFAULT_FORCE_STEADY: int = 290

_CLAUSE_PUNCT = ",;:—"


def force_emit_threshold(
    chunks_emitted: int,
    schedule: list[int] = DEFAULT_FORCE_SCHEDULE,
    steady: int = DEFAULT_FORCE_STEADY,
) -> int:
    """Return the buffer size at which the next chunk should be force-emitted
    (assuming no sentence boundary fires first)."""
    if chunks_emitted < len(schedule):
        return schedule[chunks_emitted]
    return steady


def find_soft_cut(text: str, target_len: int, search_window: int = 60) -> int | None:
    """
    Find an exclusive end position to slice the buffer at, preferring (in order):
      1. a clause-ending punctuation followed by whitespace (`, `, `; `, etc.)
      2. any whitespace (= word boundary)
    Searches backwards from `target_len` within `search_window` characters.
    Returns None if `len(text) <= target_len` (not yet over threshold) or no
    acceptable cut exists in the window (don't cut mid-word).
    """
    n = len(text)
    if n <= target_len:
        return None
    window_start = max(0, target_len - search_window)
    upper = min(target_len, n - 1)

    # 1. clause boundary (followed by whitespace or EOF)
    for i in range(upper, window_start - 1, -1):
        if text[i] in _CLAUSE_PUNCT and (i + 1 >= n or text[i + 1].isspace()):
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            return j

    # 2. word boundary
    for i in range(upper, window_start - 1, -1):
        if text[i].isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            return j

    return None
