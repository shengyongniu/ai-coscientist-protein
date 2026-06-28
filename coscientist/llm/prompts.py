"""Jinja2-based prompt templating.

Templates live in ``coscientist/prompts/*.j2``. Each agent renders its template
with a context dict to produce the user prompt; the system prompt carries a
``[[MOCK:<kind>]]`` directive so the offline mock provider knows what shape of
response to synthesize.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(template_name: str, **context) -> str:
    return _env().get_template(template_name).render(**context)
