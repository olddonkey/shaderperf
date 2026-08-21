"""Simple, offset-preserving WGSL control-flow region discovery.

This is intentionally not a complete WGSL parser.  It recognizes function
bodies and the brace-delimited control-flow constructs needed by ShaderPerf's
gate0 shaders.  Comments are masked while scanning so every index still points
into the original source; ablation always splices that original text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_IDENT_START = frozenset("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_IDENT_CONTINUE = _IDENT_START | frozenset("0123456789")
_CONTROL_EXITS = re.compile(r"\b(?:return|break|continue|discard)\b")


@dataclass(frozen=True)
class Region:
    id: str
    function: str
    kind: str
    ordinal: int
    line: int
    body_start: int
    body_end: int
    snippet: str


@dataclass(frozen=True)
class _Function:
    name: str
    body_start: int
    body_end: int


def _line(src: str, offset: int) -> int:
    return src.count("\n", 0, offset) + 1


def _parse_error(src: str, offset: int, message: str) -> ValueError:
    return ValueError(f"{message} at offset {offset} (line {_line(src, offset)})")


def strip_comments(src: str) -> str:
    """Mask WGSL comments with spaces while preserving length and newlines."""
    chars = list(src)
    i = 0
    while i < len(src):
        if src.startswith("//", i):
            chars[i] = chars[i + 1] = " "
            i += 2
            while i < len(src) and src[i] not in "\r\n":
                chars[i] = " "
                i += 1
            continue
        if src.startswith("/*", i):
            start = i
            chars[i] = chars[i + 1] = " "
            i += 2
            while i < len(src) and not src.startswith("*/", i):
                if src[i] not in "\r\n":
                    chars[i] = " "
                i += 1
            if i >= len(src):
                raise _parse_error(src, start, "unterminated block comment")
            chars[i] = chars[i + 1] = " "
            i += 2
            continue
        i += 1
    return "".join(chars)


def _skip_space(masked: str, offset: int, limit: int | None = None) -> int:
    end = len(masked) if limit is None else limit
    while offset < end and masked[offset].isspace():
        offset += 1
    return offset


def _identifier(masked: str, offset: int) -> tuple[str, int] | None:
    if offset >= len(masked) or masked[offset] not in _IDENT_START:
        return None
    end = offset + 1
    while end < len(masked) and masked[end] in _IDENT_CONTINUE:
        end += 1
    return masked[offset:end], end


def _match_delimiter(
    src: str, masked: str, opening: int, opener: str, closer: str, limit: int
) -> int:
    if opening >= limit or masked[opening] != opener:
        raise _parse_error(src, opening, f"expected {opener!r}")
    depth = 1
    i = opening + 1
    while i < limit:
        if masked[i] == opener:
            depth += 1
        elif masked[i] == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise _parse_error(src, opening, f"unmatched {opener!r}")


def _find_functions(src: str, masked: str) -> list[_Function]:
    functions: list[_Function] = []
    i = 0
    while i < len(masked):
        token = _identifier(masked, i)
        if token is None:
            i += 1
            continue
        word, token_end = token
        if word != "fn":
            i = token_end
            continue

        name_start = _skip_space(masked, token_end)
        name_token = _identifier(masked, name_start)
        if name_token is None:
            raise _parse_error(src, name_start, "expected function name after 'fn'")
        name, name_end = name_token
        paren = _skip_space(masked, name_end)
        if paren >= len(masked) or masked[paren] != "(":
            raise _parse_error(src, paren, "expected '(' after function name")
        signature_end = _match_delimiter(
            src, masked, paren, "(", ")", len(masked)
        )

        # Return types and their attributes may follow the parameter list.
        # WGSL function signatures cannot contain a brace before the body.
        body_open = masked.find("{", signature_end + 1)
        semicolon = masked.find(";", signature_end + 1)
        if body_open < 0 or (semicolon >= 0 and semicolon < body_open):
            raise _parse_error(src, signature_end + 1, "expected function body")
        body_close = _match_delimiter(
            src, masked, body_open, "{", "}", len(masked)
        )
        functions.append(_Function(name, body_open + 1, body_close))
        i = body_close + 1
    return functions


def _body_open_for_construct(
    src: str,
    masked: str,
    kind: str,
    token_end: int,
    function_end: int,
) -> int:
    cursor = _skip_space(masked, token_end, function_end)
    if kind in {"for", "while", "if", "else_if"}:
        if cursor >= function_end or masked[cursor] != "(":
            raise _parse_error(src, cursor, f"expected '(' after {kind!r}")
        cursor = _match_delimiter(
            src, masked, cursor, "(", ")", function_end
        ) + 1
        cursor = _skip_space(masked, cursor, function_end)
    if cursor >= function_end or masked[cursor] != "{":
        raise _parse_error(src, cursor, f"expected '{{' for {kind!r} body")
    return cursor


def find_regions(src: str) -> list[Region]:
    """Return non-empty control-flow bodies at every function nesting depth."""
    masked = strip_comments(src)
    regions: list[Region] = []

    for function in _find_functions(src, masked):
        ordinal = 0
        i = function.body_start
        while i < function.body_end:
            token = _identifier(masked, i)
            if token is None:
                i += 1
                continue
            word, token_end = token
            kind: str | None = None
            construct_end = token_end

            if word in {"for", "while", "loop", "if"}:
                kind = word
            elif word == "else":
                after_else = _skip_space(masked, token_end, function.body_end)
                next_token = _identifier(masked, after_else)
                if next_token is not None and next_token[0] == "if":
                    kind = "else_if"
                    construct_end = next_token[1]
                else:
                    kind = "else"

            if kind is None:
                i = token_end
                continue

            body_open = _body_open_for_construct(
                src,
                masked,
                kind,
                construct_end,
                function.body_end,
            )
            body_close = _match_delimiter(
                src, masked, body_open, "{", "}", function.body_end + 1
            )
            body_start = body_open + 1
            body_end = body_close

            # Comment-only bodies are semantically empty too: masking lets us
            # make that decision without changing the source used for output.
            if masked[body_start:body_end].strip():
                ordinal += 1
                snippet = " ".join(src[body_start:body_end].split())[:60]
                regions.append(
                    Region(
                        id=f"{function.name}:{kind}#{ordinal}",
                        function=function.name,
                        kind=kind,
                        ordinal=ordinal,
                        line=_line(src, body_open),
                        body_start=body_start,
                        body_end=body_end,
                        snippet=snippet,
                    )
                )

            # Resume inside the body so nested constructs are still found.
            # For `else if`, this also avoids reporting its `if` a second time.
            i = body_start

    return regions


def ablate(src: str, region: Region) -> str:
    """Replace one region body with a space, preserving its braces verbatim."""
    if not 0 < region.body_start <= region.body_end < len(src):
        raise ValueError("region body offsets are outside the source")
    if src[region.body_start - 1] != "{" or src[region.body_end] != "}":
        raise ValueError("region body offsets do not point between braces")
    return src[: region.body_start] + " " + src[region.body_end :]


def region_contains_control_exit(src: str, region: Region) -> bool:
    """Return whether a region body contains an advisory control-exit token."""
    body = src[region.body_start : region.body_end]
    return _CONTROL_EXITS.search(strip_comments(body)) is not None
