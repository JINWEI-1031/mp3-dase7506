"""Step 7 continuation at the original first-one submission boundary."""
import json
import re
from pathlib import Path

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse
from step4_agent import identity, passed
from step4_edits import Schema, simple
from step5x_edits import aggregate_edits
from step5z_first_one_agent import create_agent as first_one
from step7_semantics import ANTI, DUAL, aggregate_intent, aggregate_rebuild, both_groups, negative_in

FAMILIES = {"local", "aggregate", "roles", "sets", "scope", "frequent"}


def evidence_to_continue(tree, question, arity):
    if not isinstance(tree, exp.Select):
        return False
    if DUAL.search(question):
        return tree.find(exp.In) is not None and not tree.find(exp.Exists)
    intent = aggregate_intent(question, arity)
    if intent:
        rank, threshold, count, ordered, limit = intent
        order = tree.args.get("order")
        return bool((rank or limit) and not (order and order.find(exp.AggFunc))
                    or threshold and tree.args.get("group") and not tree.args.get("having")
                    or count and arity > 1 and not tree.find(exp.AggFunc))
    return False


class Step7Agent:
    def __init__(self, base, settings):
        self.base = base
        self.settings = settings
        self.fallback = None
        self.start = None
        self.audit = {"reason": "baseline continuation", "candidates": []}

    def act(self, obs):
        history = obs["history"]
        if self.fallback is None:
            action = self.base.act(obs)
            if action["tool"] != "submit":
                return action
            self.fallback = action
            self.start = len(history)
        checks = {e["sql"]: e["observation"] for e in history if e["tool"] == "check"}
        sql = self.fallback["sql"]
        question = obs["task"]["question"].lower()
        arity = len(obs["example"]["columns"])
        if self.start == len(history):
            self.audit["remaining_at_boundary"] = obs["remaining"]
            public = any(passed(v) for v in checks.values())
            narrow = "frequent" in self.settings["families"]
            if narrow and not (DUAL.search(question) or re.search(
                    r"\b(?:most|least) (?:common|popular|frequent)\b", question)):
                self.audit["reason"] = "no explicit frequency or independent groups"
                return self.fallback
            if public != self.settings["public"] or (public and not evidence_to_continue(
                    parse(sql), question, arity)):
                self.audit["reason"] = "public stratum or structural gate"
                return self.fallback
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        budget = min(obs["remaining"], self.settings["budget"] - (len(history) - self.start))
        if tables is None:
            if budget >= 2 and (aggregate_intent(question, arity) or ANTI.search(question) or DUAL.search(question)):
                self.audit["reason"] = "schema requested"
                return {"tool": "schema"}
            self.audit["reason"] = "no schema within budget or no intent"
            return self.fallback
        if "generated" not in self.audit:
            self.audit["generated"] = True
            original = {e["sql"]: e["observation"] for e in history[:self.start] if e["tool"] == "check"}
            self.generate(obs, original, Schema(tables), question, arity)
        for candidate in self.audit["candidates"]:
            candidate_sql = candidate["sql"]
            if candidate_sql in checks and passed(checks[candidate_sql]):
                self.audit["reason"] = "first new public pass"
                return {"tool": "submit", "sql": candidate_sql}
            if candidate_sql not in checks and budget:
                return {"tool": "check", "sql": candidate_sql}
        self.audit["reason"] = "no candidate" if not self.audit["candidates"] else "no new public pass"
        return self.fallback

    def generate(self, obs, checks, schema, question, arity):
        original = obs["issue_sql"]
        fallback = self.fallback["sql"]
        parents = list(dict.fromkeys([fallback, original, *checks]))[:4]
        seen = {identity(sql) for sql in checks} | {identity(fallback)}
        for family in self.settings["families"]:
            family_count = 0
            for parent in parents:
                tree = parse(parent)
                if not isinstance(tree, exp.Select) or tree.args.get("with"):
                    continue
                if family == "local":
                    if simple(parent) is None:
                        continue
                    generated = aggregate_edits(tree, schema, checks.get(parent, {}), question, arity)
                elif family in ("aggregate", "roles", "frequent"):
                    if family == "frequent" and DUAL.search(question):
                        continue
                    generated = aggregate_rebuild(tree, schema, question, arity,
                                                  family in ("roles", "frequent"))
                elif family == "scope":
                    from step7_scope import anti_scope
                    generated = anti_scope(tree, schema, question, arity)
                elif DUAL.search(question):
                    generated = both_groups(tree, schema, question, arity)
                elif ANTI.search(question):
                    generated = negative_in(tree, schema)
                else:
                    continue
                for rule, sql, evidence in generated:
                    key = identity(sql)
                    if key in seen:
                        continue
                    try:
                        validate_sql(sql)
                    except ValueError:
                        continue
                    seen.add(key)
                    self.audit["candidates"].append({"sql": sql, "rule": rule,
                                                     "parent": parent, "evidence": evidence})
                    family_count += 1
                    if family_count == 4:
                        break
                if family_count == 4:
                    break


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    if (set(settings) != {"budget", "families", "public"} or settings["budget"] not in (1, 2)
            or not settings["families"] or not set(settings["families"]) <= FAMILIES
            or type(settings["public"]) is not bool):
        raise ValueError("Invalid step 7 configuration")
    return Step7Agent(first_one(path.with_name("step5z-first-one.json")), settings)
