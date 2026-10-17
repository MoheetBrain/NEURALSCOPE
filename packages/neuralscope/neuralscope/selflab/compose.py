"""Compose a user message from labelled parts and record the exact character span of every part.

Shared by the SelfLab chat composer (``/api/selflab/chat``) and the Experiment A dataset families,
so a document or a hint is always located by construction, never by searching the text afterwards.
Token spans are added later by ``ModelAdapter.tokenize_messages`` (instrumentation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from neuralscope.selflab.provenance import sha256_text
from neuralscope.selflab.schemas import ContextSegment

PartKind = Literal["document", "user_hint", "question", "system", "other"]


@dataclass(frozen=True)
class Part:
    kind: PartKind
    text: str


def compose(parts: list[Part], message_index: int, separator: str = "\n\n") -> tuple[str, list[ContextSegment]]:
    """Join ``parts`` with ``separator``; return the content and one ContextSegment per non-"other" part."""
    content = ""
    segments: list[ContextSegment] = []
    for i, p in enumerate(parts):
        if i:
            content += separator
        start = len(content)
        content += p.text
        if p.kind != "other" and p.text:
            segments.append(ContextSegment(kind=p.kind, message_index=message_index, char_start=start,
                                           char_end=len(content), sha256=sha256_text(p.text)))
    return content, segments


def chat_user_message(question: str, document: str | None = None, hint: str | None = None,
                      message_index: int = 0) -> tuple[str, list[ContextSegment]]:
    """The fixed composer used by the SelfLab chat panel (family ``chat``)."""
    parts: list[Part] = []
    if document:
        parts += [Part("other", "Document:"), Part("document", document.strip())]
    if hint:
        parts.append(Part("user_hint", hint.strip()))
    parts.append(Part("question", question.strip()))
    return compose(parts, message_index)
