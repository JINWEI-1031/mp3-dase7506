"""Run the fixed grouped candidate pool after the step 7 fallback."""
from pathlib import Path
import json

from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity, passed
from step4_edits import Schema
from step7_agent import create_agent as step7
from step8_groups import compile_pool


class Step8Agent:
    def __init__(self, base, settings):
        self.base = base
        self.settings = settings
        self.fallback = None
        self.pool = []
        self.audit = {"reason": "baseline continuation", "candidates": self.pool}

    def act(self, obs):
        history = obs["history"]
        if self.fallback is None:
            action = self.base.act(obs)
            if action["tool"] != "submit":
                return action
            self.fallback = action
            self.start = len(history)
            self.prepare(obs)
        checked = {e["sql"]: e["observation"] for e in history[self.start:] if e["tool"] == "check"}
        for candidate in self.pool:
            sql = candidate["sql"]
            if sql in checked:
                if passed(checked[sql]):
                    self.audit["reason"] = "new public pass"
                    return {"tool": "submit", "sql": sql}
            elif obs["remaining"] and len(checked) < self.settings["checks"]:
                return {"tool": "check", "sql": sql}
        if self.pool:
            self.audit["reason"] = "no new public pass within budget"
        return self.fallback

    def prepare(self, obs):
        history = obs["history"]
        self.audit["remaining_at_boundary"] = obs["remaining"]
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            self.audit["reason"] = "baseline public pass"
            return
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            self.audit["reason"] = "schema unavailable at boundary"
            return
        seen = {identity(e["sql"]) for e in history if "sql" in e}
        seen.add(identity(self.fallback["sql"]))
        for candidate in compile_pool(obs, Schema(tables), seen, self.settings["families"]):
            try:
                validate_sql(candidate["sql"])
            except ValueError:
                continue
            self.pool.append(candidate)
        self.audit["reason"] = "no new grouped candidate" if not self.pool else "grouped pool ready"


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if set(settings) != {"checks", "families"} or settings["checks"] not in (1, 2) or not set(
            settings["families"]) <= {"fact", "detail"}:
        raise ValueError("Invalid step 8 configuration")
    return Step8Agent(step7(path.with_name("step7-aggregate-2.json")), settings)
