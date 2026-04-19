"""LLM-as-judge — primary judge for Phase F.

Matches the GAAMA / HyperMem / EverMemOS evaluation protocol:
- gpt-4o (default) evaluates correctness
- CORRECT if the generated answer touches on the same topic as gold
- Temperature 0 for reproducibility (caller can override)

The prompt is a near-verbatim copy of the one used across GAAMA, HyperMem,
and EverMemOS papers so cross-system numbers remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from htb.eval.interfaces import Judgment
from htb.llm import LLMAdapter

JUDGE_PROMPT_TEMPLATE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the relevant information. For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer, on the other hand, may be much longer, but it needs to touch upon the same topic as the gold answer, otherwise it should be labeled as wrong.

Your task is to determine whether they match, so be generous: as long as the generated answer matches on the same topic as the gold answer, count it as CORRECT.

For example:
If the gold answer is "Mex food" and the generated answer is "Mexican cuisine", you should say CORRECT.
If the gold answer is "May 7th, 2023" and the generated answer is "7 May 2023", you should say CORRECT.
If the gold answer is "running" and the generated answer is "cardio", you should say WRONG.
If the gold answer is "Raw food" and the generated answer is "vegetarian", you should say WRONG.

Now, it's your turn.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Answer with EXACTLY one word: CORRECT or WRONG."""


def _normalise_verdict(raw: str) -> Judgment:
    """Parse LLM output into CORRECT/WRONG. Robust to leading/trailing
    whitespace, extra punctuation, and mixed case."""
    text = (raw or "").strip().upper()
    # Take the first token that matches CORRECT or WRONG (handles "CORRECT."
    # or "Answer: CORRECT" patterns some models emit).
    for token in text.replace(".", " ").replace(":", " ").split():
        if token in ("CORRECT", "TRUE", "YES"):
            return "CORRECT"
        if token in ("WRONG", "FALSE", "NO", "INCORRECT"):
            return "WRONG"
    # Conservative fallback: anything ambiguous counts as WRONG so we don't
    # over-report accuracy.
    return "WRONG"


@dataclass
class OpenAIJudge:
    """LLM-as-judge using an OpenAI-compatible adapter.

    Separate from the extraction LLM — the paper convention is to use a
    stronger model (``gpt-4o``) for judging than for extracting. Pass the
    same ``LLMAdapter`` class the rest of the project uses so OpenRouter,
    Anthropic-proxied endpoints, etc. work without code changes.
    """

    llm: LLMAdapter
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 8
    name: str = field(default="openai-judge")

    def judge(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> Judgment:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer or "(no answer)",
        )
        raw = self.llm.complete(
            prompt,
            max_tokens=self.max_tokens,
            model=self.model,
            temperature=self.temperature,
        )
        return _normalise_verdict(raw)
