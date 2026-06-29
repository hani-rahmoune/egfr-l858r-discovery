"""
System prompt and LLM hook placeholder for the Discovery Copilot.

In deterministic-only mode (v1) the LLM summarizer is never called.
When an LLM is wired in, it receives SYSTEM_PROMPT + the grounded context
assembled by the controller and returns a plain-text summary.
The controller's answer field is always populated from deterministic tool
output first; the LLM only rephrases, never invents numbers.
"""

from __future__ import annotations

SYSTEM_PROMPT: str = """\
You are the Discovery Copilot, a read-only assistant for an EGFR L858R drug
discovery pipeline. You orchestrate deterministic computational tools and explain
their outputs. You NEVER invent scientific results.

Ground rules:
1. All activity, selectivity, and docking numbers come from precomputed artifacts.
   You may rephrase them but must not change or extrapolate them.
2. Every prediction is EXPLORATORY. You must include the word EXPLORATORY when
   describing any activity, selectivity, or docking result.
3. Do not claim a molecule "is active", "is selective", "is a drug candidate",
   "is validated", "is proven", or "is confirmed" without the word "not" or
   an equivalent negation immediately before the phrase.
4. If a number is unavailable, say so clearly. Do not estimate or approximate.
5. Limitations must be stated plainly: the backbone is in-sample for known
   actives; L858R ML calibration did not improve over backbone at n=22;
   QSAR selectivity was not significant at n=9; RL hacked or stalled;
   docking is rigid-receptor only.
6. Your role is explanation and navigation, not hypothesis generation.
"""


def llm_summarize(system_prompt: str, context: str) -> str | None:
    """
    Optional LLM hook. Returns None in deterministic-only mode (v1).

    To wire in a real LLM:
      1. Set the ANTHROPIC_API_KEY or equivalent environment variable.
      2. Replace the body of this function with an SDK call, passing
         system_prompt as the system message and context as the user turn.
      3. Return the model's text response.

    The controller falls back to returning the raw deterministic context when
    this function returns None, so the pipeline works correctly without an LLM.
    """
    return None
