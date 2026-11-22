"""Step 4 experiment entry point. Step 2 and 3 use submission.py."""
from functools import lru_cache
import json
from pathlib import Path

from sqlglot import exp

from repair_rules import MAX_SQL_BYTES, intent_conflicts, parse, semantic_edits, sql_text
from step4_edits import (Schema, absence_edits, frequency_edit, grouped_sum_edit, hint_delta,
                         hint_feedback, join_repairs, key_edits, simple, threshold_edits)
from step4_templates import category, official_query, record, retrieve_local

MAX_HINT_EDITS = 6
MAX_TEMPLATE_EDITS = 5


def identity(sql):
    tree = parse(sql)
    return sql_text(tree) if tree is not None else sql.strip()


def passed(feedback):
    return feedback["ok"] and feedback["matches_example"]


def usable(sql):
    return isinstance(sql, str) and bool(sql.strip()) and len(sql.encode("utf-8")) <= MAX_SQL_BYTES


def continuation_conflicts(sql, question):
    conflicts = intent_conflicts(sql, question)
    if conflicts and "distinct" in conflicts:
        tree = parse(sql)
        # Grouped entity counts in HAVING or ORDER BY can satisfy the
        # distinct wording even when the projection itself is a plain column.
        if any(isinstance(count.this, exp.Distinct)
               for count in tree.find_all(exp.Count)):
            conflicts.discard("distinct")
    return conflicts


@lru_cache(maxsize=1)
def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config.setdefault("continue", False)
    if (set(config) != {"hint_edits", "retrieval", "schema", "continue"}
            or config["hint_edits"] not in (True, False, "feedback", "delta")
            or type(config["continue"]) is not bool
            or config["retrieval"] not in {None, "intent", "error", "structure", "lexical", "hybrid"}
            or config["schema"] not in {"full", "related", "bridge"}):
        raise ValueError("Invalid step 4 configuration")
    return config


@lru_cache(maxsize=1)
def training_index():
    path = Path(__file__).with_name("step4-training-index.json")
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def template_edits(cases, parents, schema, question, arity, reach):
    operations = {}
    for case in cases:
        for operation in case["edits"]:
            operations.setdefault(operation, []).append(case["id"])
    for op, ids in operations.items():
        for parent, feedback in parents:
            tree = simple(parent)
            if tree is None or tree.args.get("with") or len(tree.selects) != arity:
                continue
            handlers = {
                "add_join": lambda: join_repairs(tree, schema, feedback, reach),
                "join_key": lambda: key_edits(tree, schema),
                "anti_join": lambda: absence_edits(tree, schema, question),
                "comparator": lambda: threshold_edits(tree, question),
                "frequency": lambda: frequency_edit(tree, question, arity),
                "group_sum": lambda: grouped_sum_edit(tree, question),
            }
            for location, candidate, evidence in handlers[op]():
                yield (candidate, location, {"operation": op, "train_cases": ids, **evidence}, parent)


class Step4Agent:
    def __init__(self, config):
        self.config = config
        self.audit = {"candidates": {}, "retrieved": [], "retrieval_mode": config["retrieval"]}

    def add(self, sql, source, location=None, evidence=None, parent=None):
        if not usable(sql):
            return None
        key = identity(sql)
        candidate = self.audit["candidates"].setdefault(key, {"sql": sql, "sources": {}})
        candidate["sources"].setdefault(source, {"location": location, "evidence": evidence,
                                                  "parent": parent})
        return candidate["sql"]

    def expand(self, generator, source, limit):
        candidates = self.audit["candidates"]
        for sql, location, evidence, parent in generator:
            if usable(sql) and (identity(sql) in candidates
                                or sum(source in c["sources"] for c in candidates.values()) < limit):
                self.add(sql, source, location, evidence, parent)
            if sum(source in c["sources"] for c in candidates.values()) >= limit:
                return

    def act(self, observation):
        draft = observation["issue_sql"]
        history = observation["history"]
        remaining = observation["remaining"]
        question = observation["task"]["question"]
        arity = len(observation["example"]["columns"])
        checks = {event["sql"]: event["observation"] for event in history if event["tool"] == "check"}
        self.add(draft, "draft")
        if draft not in checks:
            return {"tool": "check", "sql": draft}

        observations = {event["tool"]: event["observation"] for event in history
                        if event["tool"] != "check"}
        hints = []
        for index, sql in enumerate(observations.get("repair_hint", [])):
            candidate = self.add(sql, "hint", evidence={"index": index})
            if candidate is not None and candidate not in hints:
                hints.append(candidate)

        first_pass = next((sql for sql, result in checks.items() if passed(result)), None)
        if first_pass is not None:
            conflicts = (continuation_conflicts(first_pass, question)
                         if self.config["continue"] else None)
            if not conflicts or (checks[first_pass]["row_count"] != 0
                                 and "count_rank" not in conflicts):
                return {"tool": "submit", "sql": first_pass}
            self.audit["continued"] = {"default": first_pass, "conflicts": sorted(conflicts)}
            for sql, result in checks.items():
                alternative = continuation_conflicts(sql, question) if passed(result) else None
                if alternative is not None and alternative < conflicts:
                    return {"tool": "submit", "sql": sql}
        if "repair_hint" not in observations and remaining >= 2:
            return {"tool": "repair_hint"}
        for sql in hints:
            if remaining and sql not in checks:
                return {"tool": "check", "sql": sql}

        if first_pass is not None:
            rules = tuple(sorted(conflicts & {"distinct", "count_rank"}))
            self.expand(((sql, rule, {"conflicts": sorted(conflicts)}, first_pass)
                         for rule, sql in semantic_edits(first_pass, question, rules)),
                        "continuation", 2)
            for candidate in self.audit["candidates"].values():
                sql = candidate["sql"]
                if remaining and "continuation" in candidate["sources"] and sql not in checks:
                    return {"tool": "check", "sql": sql}
            return {"tool": "submit", "sql": first_pass}

        mode = self.config["retrieval"]
        official = mode in {"intent", "error", "structure"}
        # A retrieval and schema call each need a check slot after them.
        if official and remaining >= 3 and "retrieve" not in observations:
            return {"tool": "retrieve", "query": official_query(question, draft, checks[draft], mode)}
        hint_mode = self.config["hint_edits"]
        if hint_mode in (True, "delta"):
            self.expand(((sql, location, evidence, hint) for hint in hints if hint != draft
                         for location, sql, evidence in hint_delta(draft, hint, arity)),
                        "hint_delta", MAX_HINT_EDITS // 2)
        if (remaining >= 2 and "schema" not in observations
                and (mode or (hint_mode in (True, "feedback") and hints))):
            return {"tool": "schema"}

        if "schema" in observations:
            tables = observations["schema"]
            schema = Schema(tables)
            if hint_mode in (True, "feedback"):
                self.expand(((sql, location, evidence, hint) for hint in hints if hint in checks
                             for location, sql, evidence in
                             hint_feedback(hint, checks[hint], tables, schema, question, arity)),
                            "hint_feedback", MAX_HINT_EDITS // 2)
            if mode:
                if official:
                    cases = [record(item) for item in observations.get("retrieve", [])]
                else:
                    index = training_index()
                    # For offline train diagnostics, hold out the target database
                    # from local retrieval using training provenance only.
                    task = observation["task"]["id"]
                    dbs = {item["db"] for item in index if item["id"] == task}
                    excluded = {item["id"] for item in index if item["db"] in dbs} | {task}
                    cases = retrieve_local(index, question, draft, category(checks[draft]),
                                           excluded, mode)
                self.audit["retrieved"] = [{"id": item["id"], "edits": item["edits"]} for item in cases]
                parents = [(sql, checks[sql]) for sql in dict.fromkeys([*hints, draft]) if sql in checks]
                self.expand(template_edits(cases, parents, schema, question, arity,
                                           self.config["schema"]), "template", MAX_TEMPLATE_EDITS)

        for source in ("hint_feedback", "hint_delta", "template"):
            for candidate in self.audit["candidates"].values():
                sql = candidate["sql"]
                if remaining and source in candidate["sources"] and sql not in checks:
                    return {"tool": "check", "sql": sql}
        return {"tool": "submit", "sql": draft}


def create_agent(checkpoint):
    return Step4Agent(load_config(Path(checkpoint).resolve()))
