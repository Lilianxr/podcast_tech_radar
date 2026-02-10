from __future__ import annotations

from bs4 import BeautifulSoup

from ..schemas import EpisodeInput, Segment, TranscriptParseResult
from ..utils import hash_text


class GenericHtmlTextExtractor:
    def parse(self, html: str, source_url: str | None = None) -> TranscriptParseResult:
        soup = BeautifulSoup(html, "lxml")
        text = "\n".join(soup.stripped_strings)
        segment = Segment(text=text, hash=hash_text(text))
        return TranscriptParseResult(
            episode=EpisodeInput(source_url=source_url, raw_html=html),
            toc=[],
            segments=[segment],
        )


class GenericTextParser:
    def parse(self, text: str, source_url: str | None = None) -> TranscriptParseResult:
        segment = Segment(text=text, hash=hash_text(text))
        return TranscriptParseResult(
            episode=EpisodeInput(source_url=source_url),
            toc=[],
            segments=[segment],
        )
