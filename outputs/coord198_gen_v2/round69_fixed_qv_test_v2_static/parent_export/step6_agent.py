"""Choose between the aggregate submission and a publicly checked hint."""
import json
from pathlib import Path
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import DISTINCT_REQUEST, parse
from step4_agent import passed
from step5_agent import load_policy
from step5x_agent import ExpansionAgent


RISKS = ("projection_distinct", "extra_inner_join")


def comparable_select(sql):
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return None
    tree = tree.transform(lambda node: exp.to_identifier(node.name.lower())
                          if isinstance(node, exp.Identifier) else node)
    tables = list(tree.find_all(exp.Table))
    if len({table.name for table in tables}) == len(tables):
        aliases = {table.alias_or_name: table.name for table in tables}
        for col in tree.find_all(exp.Column):
            if col.table in aliases:
                col.set("table", exp.to_identifier(aliases[col.table]))
        for table in tables:
            table.set("alias", None)
    return tree


def switch_risks(draft, hint, question):
    old, new = map(comparable_select, (draft, hint))
    if old is None or new is None:
        return {}
    if not old.args.get("distinct") and isinstance(new.args.get("distinct"), exp.Distinct):
        stripped = new.copy()
        stripped.set("distinct", None)
        dedup = DISTINCT_REQUEST.search(question) or re.search(
            r"\b(?:without duplicates|no duplicates|only once|deduplicat\w*)\b", question, re.I)
        if old == stripped and not dedup:
            return {"projection_distinct": new.args["distinct"].sql()}
    prior, joins = old.args.get("joins") or [], new.args.get("joins") or []
    if len(joins) != len(prior) + 1:
        return {}
    for index, join in enumerate(joins):
        table = join.this
        if (not isinstance(table, exp.Table) or join.side or join.kind not in ("", "INNER")
                or not join.args.get("on") or table.name in {t.name for t in old.find_all(exp.Table)}):
            continue
        stripped = new.copy()
        stripped.set("joins", [j.copy() for i, j in enumerate(joins) if i != index] or None)
        # When the rest of the query agrees, the new relation contributes
        # only a join constraint. Its name can supply explicit intent evidence.
        relation = r"[\s_-]+".join(map(re.escape, table.name.split("_")))
        if old == stripped and not re.search(r"\b" + relation + r"s?\b", question, re.I):
            return {"extra_inner_join": join.sql()}
    return {}


class Step6Agent(ExpansionAgent):
    def __init__(self, base, policy, settings):
        super().__init__(base, policy, "aggregate")
        self.settings = settings
        self.fallback = None
        self.audit["step6"] = {"triggered": False}

    def act(self, obs):
        history = obs["history"]
        if self.fallback is not None:
            return self.choose(obs)
        action = super().act(obs)
        if len(history) != 1 or action["tool"] != "submit":
            return action
        public = history[0]["observation"]
        if (action["sql"] == obs["issue_sql"] and passed(public)
                and public["row_count"] == 0 and obs["remaining"] >= 2):
            self.fallback = action
            return {"tool": "repair_hint"}
        return action

    def choose(self, obs):
        history, draft = obs["history"], obs["issue_sql"]
        hints = next(e["observation"] for e in history if e["tool"] == "repair_hint")
        checked = {e["sql"]: e["observation"] for e in history if e["tool"] == "check"}
        entries, seen = [], {draft}
        for raw in hints:
            sql = self.add(raw, "step6_empty_hint", parent=draft)
            if sql is None or sql in seen:
                continue
            seen.add(sql)
            try:
                validate_sql(sql)
            except ValueError:
                continue
            entries.append({"sql": sql})
        audit = self.audit["step6"] = {"triggered": True, "fallback_sql": self.fallback["sql"],
                                     "hints": entries}
        for index, entry in enumerate(entries):
            sql = entry["sql"]
            if sql not in checked:
                if obs["remaining"] and (index == 0 or (
                        index == 1 and self.settings["second"] and not audit.get("selected"))):
                    return {"tool": "check", "sql": sql}
                entry["status"] = "unchecked"
                continue
            if not passed(checked[sql]):
                entry["status"] = "public_fail"
                continue
            risks = switch_risks(draft, sql, obs["task"]["question"])
            entry.update(status="public_pass", risks=risks)
            if not self.settings["keep"] and not set(risks).intersection(self.settings["veto"]):
                audit.setdefault("selected", sql)
        if "selected" in audit:
            return {"tool": "submit", "sql": audit["selected"]}
        return self.fallback


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if (set(settings) != {"keep", "veto", "second"} or type(settings["keep"]) is not bool
            or type(settings["second"]) is not bool or not isinstance(settings["veto"], list)
            or not set(settings["veto"]).issubset(RISKS) or (settings["keep"] and settings["second"])):
        raise ValueError("Invalid step 6 configuration")
    return Step6Agent(*load_policy(path.with_name("step5-cpu_s0_d0.json")), settings)
