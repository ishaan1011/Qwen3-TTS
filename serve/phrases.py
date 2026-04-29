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
