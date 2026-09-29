"""Character-level alignment between a model answer and the oracle answer.

Epistemic role
--------------
:func:`first_divergence` locates *where* an answer string first departs from the
true answer (e.g. the first wrong digit of ``"1012"`` vs ``"1022"``). This is a
fact about the output text, used to choose which generation step to inspect in
a neural trace; it says nothing about which internal computation went wrong.

Qwen2.5 tokenizes every digit as its own token, so for pure-digit answers the
character index is approximately the index of the generated answer token
(offset by whatever the model emitted before the number -- use
``ParsedAnswer.span`` to locate the answer within the generated text).
Compare canonical strings (``ParsedAnswer.value``), not raw text.
"""

from __future__ import annotations

from typing import Any

__all__ = ["first_divergence"]


def first_divergence(model_answer: str | None, oracle_answer: str) -> dict[str, Any]:
    """Find the first character at which ``model_answer`` departs from ``oracle_answer``.

    Returns:
        ``{"index", "model_char", "oracle_char", "common_prefix"}``. ``index`` is
        ``None`` when the strings are identical. When one string is a prefix of
        the other, ``index`` is the shorter length and the exhausted side's
        char is ``None``. A missing answer (``None``) diverges at index 0.

    Examples:
        >>> first_divergence("1012", "1022")["index"]
        2
        >>> first_divergence("95", "95")["index"] is None
        True
    """
    if not isinstance(oracle_answer, str):
        raise TypeError("oracle_answer must be a str")
    model = model_answer if model_answer is not None else ""
    if model_answer is not None and model == oracle_answer:
        return {"index": None, "model_char": None, "oracle_char": None, "common_prefix": oracle_answer}
    i = 0
    limit = min(len(model), len(oracle_answer))
    while i < limit and model[i] == oracle_answer[i]:
        i += 1
    return {
        "index": i,
        "model_char": model[i] if i < len(model) else None,
        "oracle_char": oracle_answer[i] if i < len(oracle_answer) else None,
        "common_prefix": oracle_answer[:i],
    }
