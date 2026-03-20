"""JSON extraction utilities."""

from __future__ import annotations


def find_json_object(text: str) -> str | None:
    """Find the LAST balanced JSON object in *text* via brace matching.

    Returning the last object avoids picking up preamble JSON fragments
    (e.g. ``The hint says {"city":"London"} but...``) when the actual
    prediction appears later in the LLM response.
    """
    last_match: str | None = None
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_match = text[start : i + 1]
                    pos = i + 1
                    break
        else:
            # Unbalanced — skip past this opening brace
            pos = start + 1
            continue
    return last_match
