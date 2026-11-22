"""Compare category spellings at the one late check position."""
import json
from pathlib import Path

from step4_agent import passed
from step4_edits import Schema
from step12_agent import create_agent as step12
from step13_categories import compile_candidates, schedule


class Step13Agent:
    def __init__(self, baseline, policy):
        self.baseline = baseline
        self.policy = policy
        self.audit = {"pool": []}
        self.fallback = None

    def act(self, obs):
        history = obs["history"]
        if self.fallback is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["selected"]["sql"]}
            return self.fallback
        action = self.baseline.act(obs)
        scheduled = self.baseline.audit.get("selected")
        replacing = (action["tool"] == "check" and scheduled
                     and action["sql"] == scheduled["sql"])
        if not replacing and (action["tool"] != "submit" or scheduled):
            return action
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            self.audit["reason"] = "existing public pass"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            self.audit["reason"] = "schema unavailable"
            return action
        fallback = self.baseline.fallback if replacing else action
        compiled = compile_candidates(obs, Schema(tables), fallback["sql"],
                                      planned=action["sql"] if replacing else None)
        if not compiled:
            self.audit["reason"] = "no unresolved unquoted category binding"
            return action
        order = schedule(compiled, self.policy)
        self.audit.update(**compiled, slot=len(history) + 1,
                          slot_type="replacement" if replacing else "append",
                          order=[c["sql"] for c in order], baseline_action=action)
        chosen = next((c for c in order if not c["already_checked"]), None)
        if chosen is None or not obs["remaining"]:
            self.audit["reason"] = "no new action" if chosen is None else "budget exhausted"
            return action
        self.audit["selected"] = chosen
        self.fallback = fallback
        return {"tool": "check", "sql": chosen["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if set(settings) != {"policy"} or settings["policy"] not in {"original", "feedback", "normalized"}:
        raise ValueError("Invalid step 13 configuration")
    return Step13Agent(step12(path.with_name("step12-except.json")), settings["policy"])
