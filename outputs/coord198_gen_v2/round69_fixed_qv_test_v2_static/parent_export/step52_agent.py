"""Bind an event threshold from the draft or a publicly matched query."""
import json
from pathlib import Path

from sqlglot import exp

from repair_rules import parse
from step4_agent import identity, passed
from step4_edits import Schema
from step36.frequency import terms
from step48_agent import create_agent as create_baseline
from step51_agent import LIMITS, association_side, intent_for


ARMS = ("draft_seen", "matched_seen", "draft_shared", "matched_shared")


def count_anchor(sql, heads, intent):
    parent = parse(sql)
    if (not isinstance(parent, exp.Select) or len(heads) != len(parent.expressions) or len(heads) != 1
            or parent.expressions[0].alias_or_name.casefold() != heads[0]
            or any(parent.args.get(k) for k in ("where", "limit", "offset", "with", "distinct"))):
        return None
    source = parent.args.get("from")
    joins = parent.args.get("joins") or []
    if (not source or not isinstance(source.this, exp.Table) or len(joins) != 1
            or not isinstance(joins[0].this, exp.Table) or joins[0].side or joins[0].kind not in ("", "INNER")):
        return None
    on = joins[0].args.get("on")
    if not isinstance(on, exp.EQ) or not all(isinstance(col, exp.Column) for col in (on.this, on.expression)):
        return None
    group, having = parent.args.get("group"), parent.args.get("having")
    threshold = having.this.unnest() if having else None
    if (not group or not isinstance(threshold, LIMITS[intent.operator])
            or not isinstance(threshold.this, exp.Count)
            or threshold.expression != exp.Literal.number(intent.number)):
        return None
    return parent


class ThresholdSuffix:
    def __init__(self, fallback, arm):
        self.fallback = fallback
        self.arm = arm
        self.checked = None
        self.request = None
        self.anchors = []

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.checked:
            return {"tool": "submit", "sql": self.checked["sql"]} if passed(history[-1]["observation"]) else self.fallback
        intent = intent_for(observation["task"]["question"]) if remaining else None
        if not intent or intent.direction:
            return self.fallback

        heads = [h.casefold() for h in observation["example"]["columns"]]
        if len(heads) != 1 or not heads[0].endswith("name"):
            return self.fallback
        matches = list(dict.fromkeys(event["sql"] for event in reversed(history)
                                     if event["tool"] == "check" and passed(event["observation"])))
        anchors = [(sql, tree) for sql in matches if (tree := count_anchor(sql, heads, intent)) is not None]
        if not anchors:
            return self.fallback
        order = anchors[0][1]

        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if self.arm.endswith("_shared") and remaining >= 2 and self.request is None:
                # Require a named event clue on an observed associated table,
                # before committing a schema call and leaving one check slot.
                event_words = terms(intent.event)
                for sql, tree in anchors:
                    relation = tree.args["joins"][0].this.name
                    source = tree.args["from"].this.name
                    if event_words and any(event_words < terms(table) for table in (relation, source)):
                        self.request = {"anchor": sql, "event": intent.event, "output": heads,
                                        "reason": "bind the association foreign keys and output grouping field"}
                        return {"tool": "schema"}
            return self.fallback

        if self.arm.startswith("draft_"):
            original = observation["issue_sql"]
            anchors = [(original, parse(original))]
        self.anchors = [sql for sql, _ in anchors]
        schema = Schema(tables)
        candidates = {}
        for sql, parent in anchors:
            candidate = association_side(observation, intent, schema, parent, order, heads)
            if candidate:
                candidates.setdefault(identity(candidate["sql"]), {**candidate, "anchor": sql})
        if len(candidates) != 1:
            return self.fallback
        key, candidate = next(iter(candidates.items()))
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        if key in previous or key == identity(self.fallback["sql"]):
            return self.fallback
        self.checked = candidate
        return {"tool": "check", "sql": candidate["sql"]}


class Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.arm = arm
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = ThresholdSuffix(action, self.arm)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    arm = json.loads(Path(checkpoint).read_text(encoding="utf-8"))["arm"] if checkpoint else "matched_shared"
    if arm not in ARMS:
        raise ValueError(f"Unknown round-52 arm: {arm}")
    return Agent(create_baseline(), arm)
