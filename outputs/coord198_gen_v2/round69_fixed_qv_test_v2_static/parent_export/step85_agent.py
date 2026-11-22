"""Check joined witness rows for a multi-column entity list with implicit deduplication."""
from sqlglot import exp

from repair_rules import parse
from step4_agent import identity, passed
from step32_plan import context
from step38.event_grain import analyze
from step84_agent import Agent, WitnessRowsSuffix
from step84_agent import create_agent as create_baseline


class FlatRowsSuffix(WitnessRowsSuffix):
    def act(self, observation):
        if self.candidate is not None or not observation["remaining"]:
            return super().act(observation)
        tree = parse(self.fallback["sql"])
        draft = parse(observation["issue_sql"])
        if (not isinstance(tree, exp.Select) or not isinstance(draft, exp.Select)
                or draft.args.get("distinct") or not tree.args.get("distinct")
                or len(tree.expressions) < 2 or not tree.args.get("joins")):
            return self.fallback
        plans, _ = analyze(context(observation), tree)
        candidates = [plan for plan in plans if plan["basis"].get("row_multiplicity")
                      == "one row per joined witness tuple"]
        if len(candidates) != 1:
            return self.fallback
        # Reuse the same bounded check/adoption behavior with this different compiler.
        candidate = candidates[0]
        self.evidence = {"sql": candidate["sql"], "kind": "flat_witness_rows", "basis": candidate["basis"]}
        previous = next((event["observation"] for event in observation["history"] if event["tool"] == "check"
                         and identity(event["sql"]) == identity(candidate["sql"])), None)
        if previous is not None:
            return {"tool": "submit", "sql": candidate["sql"]} if passed(previous) else self.fallback
        self.candidate = candidate["sql"]
        return {"tool": "check", "sql": self.candidate}


class FlatAgent(Agent):
    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = FlatRowsSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return FlatAgent(create_baseline(checkpoint))
