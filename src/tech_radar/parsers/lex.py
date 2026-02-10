from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup

from ..schemas import EpisodeInput, Segment, TranscriptParseResult
from ..utils import build_youtube_url, hash_text, to_seconds
from .generic import GenericHtmlTextExtractor


class LexTranscriptParser:
    time_only_re = re.compile(r"^\((?P<time>\d{1,2}:\d{2}:\d{2})\)\s*$")
    time_inline_re = re.compile(r"^\((?P<time>\d{1,2}:\d{2}:\d{2})\)\s*(?P<text>.+)$")

    def fetch_html(self, url: str) -> str:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def parse(self, url: str) -> TranscriptParseResult:
        html = self.fetch_html(url)
        soup = BeautifulSoup(html, "lxml")
        title_tag = (
            soup.select_one("article h1.entry-title")
            or soup.select_one("article h1")
            or soup.select_one("main h1")
        )

        title = title_tag.get_text(" ", strip=True) if title_tag else None

        toc = []
        for anchor in soup.select("a[href*='t='], a[href*='start=']"):
            label = anchor.get_text(strip=True)
            href = anchor.get("href")
            if label and href:
                toc.append({"label": label, "href": href})

        youtube_base = None
        for anchor in soup.select("a[href*='youtube.com/watch'], a[href*='youtu.be']"):
            youtube_base = anchor.get("href")
            if youtube_base:
                break

        segments: list[Segment] = []
        current_speaker: str | None = None

        lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]

        def looks_like_speaker(line: str) -> bool:
            # exclude title
            if line.startswith("(") and ")" in line:
                return False
            if len(line) > 80:
                return False
            if line.lower() in {"introduction", "sponsors", "transcript"}:
                return False
            # shouldn't include many punctuations
            if any(ch in line for ch in [".", "?", "!", "—", "–"]):
                return False
            return True

        pending_time: str | None = None

        for line in lines:
            # 1) inline: (00:00:00) text...
            m = self.time_inline_re.match(line)
            if m:
                if not current_speaker:
                    # If not speaker, just skip
                    continue
                time_str = m.group("time")
                text = m.group("text").strip()
                t_start = to_seconds(time_str)
                segments.append(
                    Segment(
                        speaker=current_speaker,
                        t_start_sec=t_start,
                        youtube_url=build_youtube_url(youtube_base, t_start) if youtube_base else None,
                        text=text,
                        hash=hash_text(f"{current_speaker}|{time_str}|{text}"),
                    )
                )
                pending_time = None
                continue

            # 2) time only: (00:00:00)
            m = self.time_only_re.match(line)
            if m:
                pending_time = m.group("time")
                continue

            # 3) if we have a pending time, this line is very likely the text
            if pending_time:
                # If this is speaker line, update speaker and continue
                if looks_like_speaker(line):
                    current_speaker = line
                    continue

                if not current_speaker:
                    # If not speaker, do not generate
                    pending_time = None
                    continue

                time_str = pending_time
                text = line.strip()
                t_start = to_seconds(time_str)
                segments.append(
                    Segment(
                        speaker=current_speaker,
                        t_start_sec=t_start,
                        youtube_url=build_youtube_url(youtube_base, t_start) if youtube_base else None,
                        text=text,
                        hash=hash_text(f"{current_speaker}|{time_str}|{text}"),
                    )
                )
                pending_time = None
                continue

            # 4) otherwise: maybe a speaker line
            if looks_like_speaker(line):
                current_speaker = line


        # If # of segments are too small, the parsing may have problems
        if len(segments) < 30:
            sample = "\n".join(lines[:60])
            raise ValueError(f"Lex parse suspicious: only {len(segments)} segments.\nSample:\n{sample}")

        if not segments:
            return GenericHtmlTextExtractor().parse(html, source_url=url)

        return TranscriptParseResult(
            episode=EpisodeInput(source_url=url, title=title, raw_html=html),
            toc=toc,
            segments=segments,
        )
