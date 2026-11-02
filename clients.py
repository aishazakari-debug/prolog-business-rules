"""One place for every model call. Counts billed tokens, prices them, and stops at the spend cap.

No temperature is sent to the Claude models, because they reject non default values (project note, 2026-10-15).
The key is read from ANTHROPIC_API_KEY or from local-secrets/api keys.md and is never printed.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECRETS = HERE.parent.parent / "local-secrets" / "api keys.md"

# US dollars per million tokens. Haiku and Sonnet from agentic-rag-eval/config.py (checked 2026-10-15). GPT-6 Sol from the
# earlier annotation run. Re-check the pages before any run that spends real money.
PRICES = {
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00},
    "gpt-6-sol": {"in": 2.00, "out": 10.00},
}
THINKING = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Project spend. Persists to a JSON file, so every run starts from the project total (the pilots included).
    A call reserves its worst case cost before it is sent, so calls already in flight cannot carry the total past the cap."""

    def __init__(self, cap_usd: float, path: str | Path | None = None):
        self.cap, self.calls, self.tin, self.tout, self.reserved = cap_usd, 0, 0, 0, 0.0
        self.path = Path(path) if path else None
        self.base = 0.0
        if self.path and self.path.exists():
            self.base = json.loads(self.path.read_text())["total_usd"]
        self.usd = 0.0                      # spend in this run
        self.lock = threading.Lock()

    @property
    def total(self):
        return self.base + self.usd

    def reserve(self, est_usd: float) -> float:
        with self.lock:
            if self.total + self.reserved + est_usd > self.cap:
                raise BudgetExceeded(f"project total {self.total:.4f} plus reserve {self.reserved + est_usd:.4f} would pass the cap {self.cap:.2f}")
            self.reserved += est_usd
            return est_usd

    def release(self, est_usd: float):
        with self.lock:
            self.reserved -= est_usd

    def add(self, usd, tin, tout, est_usd: float = 0.0):
        with self.lock:
            self.reserved -= est_usd
            self.usd += usd
            self.calls += 1
            self.tin += tin
            self.tout += tout
            if self.path:
                self.path.write_text(json.dumps({"total_usd": self.base + self.usd}))


def _anthropic_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    if SECRETS.exists():
        for line in SECRETS.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("anthropic api key:"):
                return line.split(":", 1)[1].strip()
    raise RuntimeError("No Anthropic key found")


def _openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env = HERE.parent.parent / ".env.local"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("No OpenAI key found")


class Chooser:
    """choose(system, user) returns (reply_text, record) for one model."""

    def __init__(self, model: str, ledger: Ledger, max_tokens: int = 12):
        if model not in PRICES:
            raise RuntimeError(f"no price for {model}")
        self.model, self.ledger, self.max_tokens = model, ledger, max_tokens
        if model.startswith("claude"):
            import anthropic
            self.client = anthropic.Anthropic(api_key=_anthropic_key())
            self.kind = "anthropic"
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=_openai_key())
            self.kind = "openai"

    def choose(self, system: str, user: str, max_tokens: int | None = None):
        mt = max_tokens or self.max_tokens
        p = PRICES[self.model]
        last = None
        est = ((len(system) + len(user)) / 2.0 * p["in"] + mt * p["out"]) / 1e6      # worst case: 2 characters per token, full output
        self.ledger.reserve(est)
        for attempt in range(5):
            t0 = time.monotonic()
            try:
                if self.kind == "anthropic":
                    kw = {"thinking": THINKING[self.model]} if self.model in THINKING else {}
                    r = self.client.messages.create(model=self.model, max_tokens=mt, system=system,
                                                    messages=[{"role": "user", "content": user}], **kw)
                    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    tin, tout = r.usage.input_tokens, r.usage.output_tokens
                else:
                    r = self.client.responses.create(model=self.model, instructions=system, input=user,
                                                     reasoning={"effort": "none"}, max_output_tokens=max(mt, 16),
                                                     store=False)
                    text = r.output_text or ""
                    tin, tout = r.usage.input_tokens, r.usage.output_tokens
                usd = (tin * p["in"] + tout * p["out"]) / 1e6
                self.ledger.add(usd, tin, tout, est)
                return text, {"in": tin, "out": tout, "usd": usd, "s": round(time.monotonic() - t0, 3)}
            except BudgetExceeded:
                self.ledger.release(est)
                raise
            except Exception as e:  # network, rate limit, 5xx: back off and retry. Anything else surfaces after 5 tries.
                last = type(e).__name__
                time.sleep(2 ** attempt)
        self.ledger.release(est)
        raise RuntimeError(f"model call failed after 5 tries ({last})")
