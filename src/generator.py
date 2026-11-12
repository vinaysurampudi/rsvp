"""Evidence-aware answer generation with citations (Report section 4.4)."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import config as C
from .retrievers import tokenize

SYSTEM_PROMPT = f"""You answer questions about technical documentation using ONLY the passages provided.
Rules:
- Use only information stated in the passages. Do not add outside knowledge or assumptions.
- Cite the chunk_id of every passage you rely on.
- If the passages do not contain enough information to answer, set "evidence_sufficient" to false, set "answer" to "{C.ABSTAIN_TOKEN}" and leave "cited_chunk_ids" empty.
- Be concise and precise.
Respond with a single JSON object and nothing else:
{{"answer": "...", "cited_chunk_ids": ["..."], "evidence_sufficient": true}}"""


def build_user_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"[chunk_id: {cid}]\n{txt[:C.MAX_PASSAGE_CHARS]}" for cid, txt in passages)
    return f"Passages:\n\n{blocks}\n\nQuestion: {question}"


@dataclass
class GenResult:
    answer: str = ""
    cited_ids: list[str] = field(default_factory=list)
    abstained: bool = False
    invalid_citations: int = 0      # cited ids that were NOT in the supplied evidence set
    latency: float = 0.0
    error: str = ""
    raw: str = ""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _finalise(answer: str, cited: list, sufficient: bool, supplied_ids: set[str]) -> tuple[str, list[str], bool, int]:
    cited = [str(c).strip() for c in (cited or [])]
    valid = [c for c in dict.fromkeys(cited) if c in supplied_ids]   # validation step: cited ids must be in retrieved set
    invalid = len(set(cited)) - len(valid)
    abstained = (not sufficient) or answer.strip().upper().startswith(C.ABSTAIN_TOKEN)
    if abstained:
        valid = []
    return answer.strip(), valid, abstained, invalid


class AnthropicGenerator:
    name = "anthropic"

    def __init__(self, model: str = C.LLM_MODEL, max_retries: int = 4):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model, self.max_retries, self.use_temperature = model, max_retries, True
        self.info = {"generator": self.name, "model": model, "max_tokens": C.LLM_MAX_TOKENS,
                     "temperature": C.LLM_TEMPERATURE, "context_k": C.CONTEXT_K}

    def _call(self, system: str, user: str, max_tokens: int = C.LLM_MAX_TOKENS) -> str:
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        for attempt in range(self.max_retries):
            try:
                if self.use_temperature:
                    kwargs["temperature"] = C.LLM_TEMPERATURE
                resp = self.client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except self.anthropic.BadRequestError as e:
                if "temperature" in str(e).lower() and self.use_temperature:
                    self.use_temperature = False
                    kwargs.pop("temperature", None)
                    self.info["temperature"] = "model default (parameter not accepted)"
                    continue
                raise
            except (self.anthropic.RateLimitError, self.anthropic.APIConnectionError, self.anthropic.InternalServerError):
                time.sleep(2 ** attempt * 2)
        raise RuntimeError("LLM call failed after retries")

    def generate(self, question: str, passages: list[tuple[str, str]]) -> GenResult:
        t0 = time.perf_counter()
        res = GenResult()
        try:
            res.raw = self._call(SYSTEM_PROMPT, build_user_prompt(question, passages))
            d = _parse_json(res.raw)
            res.answer, res.cited_ids, res.abstained, res.invalid_citations = _finalise(
                str(d.get("answer", "")), d.get("cited_chunk_ids", []), bool(d.get("evidence_sufficient", True)),
                {c for c, _ in passages})
        except Exception as e:  # keep the run alive; errors are counted in the results
            res.error = f"{type(e).__name__}: {e}"
        res.latency = time.perf_counter() - t0
        return res


class GeminiGenerator:
    name = "google"

    def __init__(self, model: str = C.GEMINI_MODEL):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Google Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                "before running the app or command."
            )
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model
        self.info = {
            "generator": self.name,
            "model": model,
            "max_tokens": C.LLM_MAX_TOKENS,
            "temperature": C.LLM_TEMPERATURE,
            "context_k": C.CONTEXT_K,
        }

    def _call(self, system: str, user: str, max_tokens: int = C.LLM_MAX_TOKENS) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=self.types.GenerateContentConfig(
                system_instruction=system,
                temperature=C.LLM_TEMPERATURE,
                max_output_tokens=max_tokens,
            ),
        )
        if response.text is None:
            raise RuntimeError("Gemini returned no text. Check the API response and model access.")
        return response.text

    def generate(self, question: str, passages: list[tuple[str, str]]) -> GenResult:
        t0 = time.perf_counter()
        res = GenResult()
        try:
            res.raw = self._call(SYSTEM_PROMPT, build_user_prompt(question, passages))
            d = _parse_json(res.raw)
            res.answer, res.cited_ids, res.abstained, res.invalid_citations = _finalise(
                str(d.get("answer", "")),
                d.get("cited_chunk_ids", []),
                bool(d.get("evidence_sufficient", True)),
                {c for c, _ in passages},
            )
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
        res.latency = time.perf_counter() - t0
        return res


class ExtractiveGenerator:
    """No-LLM baseline / offline fallback: returns the best-matching sentences of the top passage and cites it.
    Abstains when question/passage lexical overlap is very low. Useful for smoke tests and as a lower bound."""
    name = "extractive"

    def __init__(self, min_overlap: float = 0.15, n_sentences: int = 2):
        self.min_overlap, self.n = min_overlap, n_sentences
        self.info = {"generator": self.name, "min_overlap": min_overlap, "n_sentences": n_sentences,
                     "context_k": C.CONTEXT_K}

    def generate(self, question: str, passages: list[tuple[str, str]]) -> GenResult:
        t0 = time.perf_counter()
        qtok = set(tokenize(question))
        best = []  # (overlap, sentence, chunk_id)
        for cid, txt in passages:
            for s in re.split(r"(?<=[.!?])\s+", txt):
                st = set(tokenize(s))
                if st and qtok:
                    best.append((len(qtok & st) / len(qtok), s, cid))
        best.sort(key=lambda x: -x[0])
        res = GenResult()
        if not best or best[0][0] < self.min_overlap:
            res.answer, res.abstained = C.ABSTAIN_TOKEN, True
        else:
            top = best[: self.n]
            res.answer = " ".join(s for _, s, _ in top)
            res.cited_ids = list(dict.fromkeys(c for _, _, c in top))
        res.latency = time.perf_counter() - t0
        return res


def build_generator(kind: str, model: str | None = None):
    if kind == "anthropic":
        return AnthropicGenerator(model or C.LLM_MODEL)
    if kind == "google":
        return GeminiGenerator(model or C.GEMINI_MODEL)
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "none":
        return None
    raise ValueError(f"Unknown generator: {kind}")
