"""Compare a shared repair template with and without one public value probe."""
import json
from pathlib import Path

from step4_agent import passed
from step4_edits import Schema
from step21_agent import create_agent as step21, prior_pool
from step23_values import choose, generate, layout

ARMS = ("no-sample", "sample")


class Step23Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.sample = arm == "sample"
        self.pending = None
        self.probe = None
        self.phase = "baseline"
        self.audit = {"pool": [], "checked": [], "trigger": False, "eligible": False}

    def resume(self, obs):
        pending, self.pending = self.pending, None
        self.phase = "done"
        if pending["tool"] == "submit" or obs["remaining"]:
            return pending
        return self.baseline.act(obs)

    def evaluate(self, obs, sample=None):
        options = generate(obs, self.schema, self.probe, sample)
        self.audit["pool"] = options
        if not options or not obs["remaining"]:
            return self.resume(obs)
        sql = options[0]["sql"]
        self.audit.update(checked=[sql], trigger=True)
        self.phase = "repair"
        return {"tool": "check", "sql": sql}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        if self.phase == "sample":
            sample = history[-1]["observation"]
            self.audit["formats"] = {key: len(values) for key, values in layout(sample).items()}
            return self.evaluate(obs, sample)
        if self.phase == "repair":
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.resume(obs)
        action = self.baseline.act(obs)
        if self.phase != "baseline":
            return action
        if not ((action["tool"] == "submit" and remaining >= 2) or (
                action["tool"] == "check" and remaining == 2)):
            return action
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if not tables:
            return action
        self.schema = Schema(tables)
        pending = action["sql"]
        probe = choose(obs, pending, self.schema, self.baseline.audit.get("pool", []))
        if probe is None:
            probe = choose(obs, pending, self.schema, prior_pool(obs, obs["issue_sql"], self.schema))
        if probe is None:
            return action
        self.probe, self.pending = probe, action
        self.audit.update(eligible=True, binding=probe.source[1:], requested=probe.requested,
                          remaining_at_intervention=remaining, pending=action)
        if not self.sample:
            return self.evaluate(obs)
        if any(e["tool"] == "execute" and e["sql"] == probe.sample_sql for e in history):
            return self.resume(obs)
        self.audit.update(sample_sql=probe.sample_sql, trigger=True)
        self.phase = "sample"
        return {"tool": "execute", "sql": probe.sample_sql}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError(arm)
    return Step23Agent(step21(path.with_name("step21-scoped.json")), arm)
