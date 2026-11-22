"""Step 3 strategies. The entry point without a checkpoint stays at step 2."""
from pathlib import Path
import json

from repair_rules import (
    MAX_RULE_CANDIDATES, MAX_SQL_BYTES, identifier_edits, identifier_error,
    intent_conflicts, semantic_edits,
)


class RuleAgent:
    def __init__(self, config):
        self.config = config
        self.lineage = {}
        self.audit = {"generated": {}, "continued": False, "default_sql": None}

    def act(self, observation):
        draft = observation["issue_sql"]
        question = observation["task"]["question"]
        remaining = observation["remaining"]
        checks = {}
        hint = schema = None
        for event in observation["history"]:
            if event["tool"] == "check":
                checks[event["sql"]] = event["observation"]
            elif event["tool"] == "repair_hint":
                hint = event["observation"] or []
            elif event["tool"] == "schema":
                schema = event["observation"]

        if draft not in checks:
            return {"tool": "check", "sql": draft}

        first_pass = next((sql for sql, result in checks.items() if result["ok"]
                           and result["matches_example"]), None)
        if first_pass is not None:
            self.audit["default_sql"] = first_pass
            conflicts = (intent_conflicts(first_pass, question)
                         if self.config["continue"] else None)
            risky = conflicts and (checks[first_pass]["row_count"] == 0
                                   or "count_rank" in conflicts)
            if not risky:
                return {"tool": "submit", "sql": first_pass}
            self.audit["continued"] = True
            for sql, result in checks.items():
                if sql == first_pass or not (result["ok"] and result["matches_example"]):
                    continue
                alternative = intent_conflicts(sql, question)
                if alternative is not None and alternative < conflicts:
                    return {"tool": "submit", "sql": sql}

        hints = [sql for sql in dict.fromkeys(sql for sql in hint or [] if isinstance(sql, str))
                 if sql.strip() and len(sql.encode("utf-8")) <= MAX_SQL_BYTES and sql not in checks]
        priority = (self.config["order"] != "hints_first" and first_pass is None
                    and identifier_error(checks[draft]))
        has_identifier_error = first_pass is None and identifier_error(checks[draft])
        # The reserved variant protects the final hint check after requesting it.
        protect_hint_check = self.config["order"] == "identifier_first_reserved"
        can_prioritize = priority and (remaining >= 3 or
                                       (hint is not None and not protect_hint_check))
        if remaining and not can_prioritize:
            if hint is None and remaining >= 2:
                return {"tool": "repair_hint"}
            if hints:
                return {"tool": "check", "sql": hints[0]}

        if (has_identifier_error and schema is None
                and remaining >= (4 if hint is None else 2)):
            return {"tool": "schema"}

        candidate_groups = []
        if has_identifier_error and schema is not None:
            for parent, feedback in checks.items():
                depth = 0 if parent == draft else self.lineage.get(parent, ("", 9))[1]
                if depth < self.config["depth"]:
                    candidate_groups.extend((kind, sql, depth + 1)
                                            for kind, sql in identifier_edits(parent, feedback, schema))
        semantic_parent = first_pass if first_pass is not None else draft
        if checks[semantic_parent]["ok"]:
            candidate_groups.extend((kind, sql, 1) for kind, sql in
                                    semantic_edits(semantic_parent, question, self.config["semantics"]))

        pending = []
        for kind, sql, depth in candidate_groups:
            if not sql.strip() or len(sql.encode("utf-8")) > MAX_SQL_BYTES or sql == draft:
                continue
            if sql not in self.audit["generated"]:
                if len(self.audit["generated"]) >= MAX_RULE_CANDIDATES:
                    continue
                self.audit["generated"][sql] = kind
            if sql not in checks and sql not in (hint or ()):
                pending.append((kind, sql, depth))
        for kind, sql, depth in pending:
            if remaining and (hint is not None or remaining >= 3):
                self.lineage[sql] = (kind, depth)
                return {"tool": "check", "sql": sql}

        if remaining >= 2 and hint is None:
            return {"tool": "repair_hint"}
        if remaining and hints:
            return {"tool": "check", "sql": hints[0]}
        return {"tool": "submit", "sql": first_pass if first_pass is not None else draft}


def load_config(checkpoint):
    config = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
    if (set(config) != {"order", "depth", "semantics", "continue"}
            or config["order"] not in {"hints_first", "identifier_first", "identifier_first_reserved"}
            or type(config["depth"]) is not int or config["depth"] not in {1, 3}
            or not isinstance(config["semantics"], list)
            or any(rule not in {"distinct", "count_rank"} for rule in config["semantics"])
            or type(config["continue"]) is not bool):
        raise ValueError("Invalid step 3 configuration")
    return config



