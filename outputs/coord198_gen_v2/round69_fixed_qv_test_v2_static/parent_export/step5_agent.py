"""Step 5 policies layered over the frozen step 4 controller."""
from functools import lru_cache
import json
from pathlib import Path

from step4_agent import Step4Agent, continuation_conflicts, load_config, passed


@lru_cache
def load_policy(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    base = dict(load_config(path.with_name("step4-combined_continue.json")))
    if not config["delta"]:
        base["hint_edits"] = "feedback"
    return base, config


class Step5Agent(Step4Agent):
    def __init__(self, base, config):
        super().__init__(base)
        self.policy = config
        self.hint_default = None
        self.model_attempted = False
        self.model_sql = None

    def act(self, observation):
        action = super().act(observation)
        history = observation["history"]
        remaining = observation["remaining"]
        question = observation["task"]["question"]
        checks = {event["sql"]: event["observation"] for event in history
                  if event["tool"] == "check"}
        other = {event["tool"]: event["observation"] for event in history
                 if event["tool"] != "check"}

        if action["tool"] == "submit" and self.policy["schedule"]:
            default = self.hint_default or action["sql"]
            if default in checks and passed(checks[default]):
                conflicts = continuation_conflicts(default, question)
                if (conflicts and remaining >= 2 and self.policy.get("fetch_hints")
                        and "repair_hint" not in other):
                    self.hint_default = default
                    return {"tool": "repair_hint"}
                if conflicts and remaining:
                    for candidate in self.audit["candidates"].values():
                        if "hint" not in candidate["sources"]:
                            continue
                        sql = candidate["sql"]
                        if sql not in checks:
                            self.hint_default = default
                            return {"tool": "check", "sql": sql}
                if self.hint_default:
                    action = {"tool": "submit", "sql": default}
                    for sql, result in checks.items():
                        alternate = continuation_conflicts(sql, question) if passed(result) else None
                        if alternate is not None and alternate < conflicts:
                            action = {"tool": "submit", "sql": sql}
                            break

        mode = self.policy.get("model")
        if mode:
            return self.with_model(action, observation, checks, other, mode)
        return action

    def with_model(self, action, observation, checks, other, mode):
        from step5_model import proposal
        if self.model_sql:
            feedback = checks[self.model_sql]
            if passed(feedback):
                return {"tool": "submit", "sql": self.model_sql}
        if self.model_attempted:
            return action
        remaining = observation["remaining"]
        accepted = [sql for sql, feedback in checks.items() if passed(feedback)]
        if mode == "review":
            if action["tool"] == "submit" and len(accepted) > 1:
                self.model_attempted = True
                sql = proposal(self, observation, mode, {"candidates": accepted,
                                                         "default": action["sql"]})
                if sql:
                    return {"tool": "submit", "sql": sql}
            return action
        if accepted or not checks or not remaining:
            return action
        schema = other.get("schema")
        needed = 1 if schema is not None else 2
        if action["tool"] != "submit":
            return action
        if remaining < needed:
            return action
        if schema is None:
            return {"tool": "schema"}

        self.model_attempted = True
        sql = proposal(self, observation, mode, {"schema": schema, "checks": checks})
        if sql:
            self.model_sql = self.add(sql, "model_" + mode)
            return {"tool": "check", "sql": self.model_sql}
        return action


def create_agent(checkpoint):
    base, config = load_policy(Path(checkpoint).resolve())
    return Step5Agent(base, config)


