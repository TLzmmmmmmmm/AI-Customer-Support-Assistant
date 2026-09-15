from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal


_REFERENCE_TERMS = (
    "参考资料",
    "引用",
    "来源",
    "reference",
    "references",
    "citation",
    "citations",
    "source",
    "sources",
)
_REFERENCE_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?"
    r"(?:参考资料|引用|来源|references?|citations?|sources?)"
    r"[ \t]*[:：]?[ \t]*$",
    re.IGNORECASE,
)
_URL_SCHEMES = ("http://", "https://", "www.")
_PLAIN_URL_DELIMITERS = frozenset('<>[]()"\'，。！？；：')
_CandidateStatus = Literal["possible", "complete", "invalid"]


def _is_reference_heading(line: str) -> bool:
    return _REFERENCE_HEADING.fullmatch(line) is not None


def _could_be_reference_heading(line: str) -> bool:
    position = 0
    while position < len(line) and line[position] in " \t":
        position += 1
    if position == len(line):
        return True

    if line[position] == "#":
        hash_start = position
        while position < len(line) and line[position] == "#":
            position += 1
        if position - hash_start > 6:
            return False
        while position < len(line) and line[position] in " \t":
            position += 1
        if position == len(line):
            return True

    body = line[position:]
    lowered = body.lower()
    for term in _REFERENCE_TERMS:
        if term.startswith(lowered):
            return True
        if not lowered.startswith(term):
            continue
        tail = body[len(term):]
        position = 0
        while position < len(tail) and tail[position] in " \t":
            position += 1
        if position == len(tail):
            return True
        if (
            tail[position] in ":："
            and all(character in " \t" for character in tail[position + 1:])
        ):
            return True
    return False


def _scheme_parts(text: str) -> tuple[str | None, bool]:
    lowered = text.lower()
    for scheme in _URL_SCHEMES:
        if lowered.startswith(scheme):
            return scheme, False
    return None, any(scheme.startswith(lowered) for scheme in _URL_SCHEMES)


class _WhitespaceStripper:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._started = False
        self._pending = ""

    def feed(self, text: str) -> None:
        for character in text:
            if character.isspace():
                if self._started:
                    self._pending += character
                continue
            if self._pending:
                self._emit(self._pending)
                self._pending = ""
            self._emit(character)
            self._started = True

    def finish(self) -> None:
        self._pending = ""


class _PlainUrlSanitizer:
    def __init__(self, downstream: _WhitespaceStripper) -> None:
        self._downstream = downstream
        self._pending = ""
        self._url_scheme: str | None = None
        self._body_length = 0

    @staticmethod
    def _is_body_character(character: str) -> bool:
        return not character.isspace() and character not in _PLAIN_URL_DELIMITERS

    def feed(self, text: str) -> None:
        remaining = text
        while remaining:
            character = remaining[0]
            remaining = remaining[1:]

            if self._url_scheme is not None:
                if self._is_body_character(character):
                    self._pending += character
                    self._body_length += 1
                    continue
                if self._body_length:
                    self._reset()
                else:
                    self._downstream.feed(self._pending)
                    self._reset()
                remaining = character + remaining
                continue

            if not self._pending:
                if character.lower() not in {"h", "w"}:
                    self._downstream.feed(character)
                    continue
                self._pending = character
            else:
                self._pending += character

            scheme, possible = _scheme_parts(self._pending)
            if scheme is not None:
                self._url_scheme = scheme
                continue
            if possible:
                continue

            released = self._pending[0]
            remaining = self._pending[1:] + remaining
            self._pending = ""
            self._downstream.feed(released)

    def finish(self) -> None:
        if self._url_scheme is None or not self._body_length:
            self._downstream.feed(self._pending)
        self._reset()
        self._downstream.finish()

    def _reset(self) -> None:
        self._pending = ""
        self._url_scheme = None
        self._body_length = 0


class _AngleUrlSanitizer:
    def __init__(self, downstream: _PlainUrlSanitizer) -> None:
        self._downstream = downstream
        self._pending = ""

    @staticmethod
    def _status(candidate: str) -> _CandidateStatus:
        tail = candidate[1:]
        scheme, possible = _scheme_parts(tail)
        if scheme is None:
            return "possible" if possible else "invalid"

        body = tail[len(scheme):]
        if not body:
            return "possible"
        for index, character in enumerate(body):
            if character == ">":
                return "complete" if index else "invalid"
            if character.isspace() or character == "<":
                return "invalid"
        return "possible"

    def feed(self, text: str) -> None:
        remaining = text
        while remaining:
            character = remaining[0]
            remaining = remaining[1:]
            if not self._pending:
                if character != "<":
                    self._downstream.feed(character)
                    continue
                self._pending = character
            else:
                self._pending += character

            status = self._status(self._pending)
            if status == "possible":
                continue
            if status == "complete":
                self._pending = ""
                continue

            released = self._pending[0]
            remaining = self._pending[1:] + remaining
            self._pending = ""
            self._downstream.feed(released)

    def finish(self) -> None:
        while self._pending:
            pending = self._pending
            self._pending = ""
            self._downstream.feed(pending[0])
            self.feed(pending[1:])
        self._downstream.finish()


class _MarkdownLinkSanitizer:
    def __init__(self, downstream: _AngleUrlSanitizer) -> None:
        self._downstream = downstream
        self._pending = ""

    @staticmethod
    def _parse(candidate: str) -> tuple[_CandidateStatus, str | None]:
        close_label = candidate.find("]", 1)
        if close_label == -1:
            if "\r" in candidate[1:] or "\n" in candidate[1:]:
                return "invalid", None
            return "possible", None
        if close_label == 1:
            return "invalid", None
        label = candidate[1:close_label]
        after_label = candidate[close_label + 1:]
        if not after_label:
            return "possible", None
        if after_label[0] != "(":
            return "invalid", None

        target = after_label[1:]
        position = 0
        while position < len(target) and target[position].isspace():
            position += 1
        target = target[position:]
        scheme, possible = _scheme_parts(target)
        if scheme is None:
            return ("possible", None) if possible else ("invalid", None)

        body = target[len(scheme):]
        if not body:
            return "possible", None
        for index, character in enumerate(body):
            if character == ")":
                return ("complete", label) if index else ("invalid", None)
            if character in "\r\n":
                return "invalid", None
        return "possible", None

    def feed(self, text: str) -> None:
        remaining = text
        while remaining:
            character = remaining[0]
            remaining = remaining[1:]
            if not self._pending:
                if character != "[":
                    self._downstream.feed(character)
                    continue
                self._pending = character
            else:
                self._pending += character

            status, label = self._parse(self._pending)
            if status == "possible":
                continue
            if status == "complete":
                self._pending = ""
                self._downstream.feed(label or "")
                continue

            released = self._pending[0]
            remaining = self._pending[1:] + remaining
            self._pending = ""
            self._downstream.feed(released)

    def finish(self) -> None:
        while self._pending:
            pending = self._pending
            self._pending = ""
            self._downstream.feed(pending[0])
            self.feed(pending[1:])
        self._downstream.finish()


class _ReferenceTruncator:
    def __init__(self, downstream: _MarkdownLinkSanitizer) -> None:
        self._downstream = downstream
        self._line: str | None = ""
        self._stopped = False

    def feed(self, text: str) -> None:
        for character in text:
            if self._stopped:
                return
            if self._line is None:
                self._downstream.feed(character)
                if character == "\n":
                    self._line = ""
                continue

            self._line += character
            if character == "\n":
                line = self._line[:-1]
                if _is_reference_heading(line):
                    self._line = ""
                    self._stopped = True
                    return
                self._downstream.feed(self._line)
                self._line = ""
                continue
            if not _could_be_reference_heading(self._line):
                self._downstream.feed(self._line)
                self._line = None

    def finish(self) -> None:
        if not self._stopped and self._line is not None:
            if not _is_reference_heading(self._line):
                self._downstream.feed(self._line)
            self._line = ""
        self._downstream.finish()


class IncrementalAnswerSanitizer:
    """Emit only answer text that no future model delta can invalidate."""

    def __init__(self) -> None:
        self._deltas: list[str] = []
        whitespace = _WhitespaceStripper(self._deltas.append)
        plain_url = _PlainUrlSanitizer(whitespace)
        angle_url = _AngleUrlSanitizer(plain_url)
        markdown = _MarkdownLinkSanitizer(angle_url)
        self._root = _ReferenceTruncator(markdown)
        self._finished = False

    def feed(self, raw_chunk: str) -> tuple[str, ...]:
        if self._finished:
            raise RuntimeError("incremental sanitizer is already finished")
        self._root.feed(raw_chunk)
        deltas = tuple(delta for delta in self._deltas if delta)
        self._deltas.clear()
        return deltas

    def finish(self) -> str:
        if self._finished:
            return ""
        self._root.finish()
        self._finished = True
        final_delta = "".join(self._deltas)
        self._deltas.clear()
        return final_delta


def sanitize_generated_answer(answer: str) -> str:
    sanitizer = IncrementalAnswerSanitizer()
    visible = list(sanitizer.feed(answer))
    visible.append(sanitizer.finish())
    return "".join(visible)
