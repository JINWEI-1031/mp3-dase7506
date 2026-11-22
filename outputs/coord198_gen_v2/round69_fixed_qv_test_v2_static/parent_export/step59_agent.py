"""Sample a bounded public field before changing an anchored literal."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step10_bindings import field_words, local_occurrences, tokens
from step38.event_grain import bind_column
from step48_agent import create_agent as create_baseline


FIELDS = {"gender", "sex", "degree", "pos", "position", "result", "status", "city", "country", "login"}
TEXT = re.compile(r"text|char|clob", re.I)
STATE = re.compile(r"\blogged\s+(in|out)\b", re.I)
QUALIFIED_DEGREE = re.compile(r"\b(master|bachelor)(?:'s)? of (arts|science|education|engineering)\b", re.I)


def normalized(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def eligible_literals(sql, question):
    tree = parse(sql)
    items = []
    if tree is None:
        return tree, items
    for atom in tree.find_all(exp.EQ, exp.NEQ, exp.In):
        if not isinstance(atom.this, exp.Column) or not atom.find_ancestor(exp.Where, exp.Having):
            continue
        field = set(field_words(atom.this.name)) & FIELDS
        if not field:
            continue
        literals = atom.expressions if isinstance(atom, exp.In) and not atom.args.get("query") else [atom.expression]
        for literal in literals:
            if not isinstance(literal, exp.Literal) or not literal.is_string:
                continue
            state = STATE.search(question) if "login" in field else None
            if state and normalized(literal.this) == "logged " + state[1].casefold():
                if re.search(r"\b(?:not|never|either|or)\b", question, re.I) or not isinstance(atom, exp.EQ):
                    continue
                items.append((atom, literal, "login_state"))
            elif local_occurrences(tokens(question), literal.this):
                items.append((atom, literal, "category"))
    return tree, items


def bound_literals(tree, items, schema):
    bindings = set()
    for atom, _, _ in items:
        select = atom.find_ancestor(exp.Select)
        field = bind_column(atom.this, schema, schema.sources(select)) if select else None
        if not field or not TEXT.search(schema.columns[field[0]][field[1]]["type"]):
            return None
        bindings.add(field)
    return next(iter(bindings)) if len(bindings) == 1 else None


def sample_sql(schema, binding):
    table, field = binding
    column = exp.column(schema.columns[table][field]["name"], quoted=True)
    query = exp.select(column.copy()).distinct().from_(exp.to_table(schema.tables[table]["name"]))
    return sql_text(query.where(exp.Not(this=exp.Is(this=column.copy(), expression=exp.Null())))
                    .order_by(column.copy()).limit(8))


def equivalent(value, observed, question, kind):
    old, new = normalized(value), normalized(observed)
    if old == new:
        return True
    if kind == "login_state":
        return (old, new) in {("logged in", "1"), ("logged out", "0")}
    if old.endswith("ation") and new == old.removesuffix("ation") + "ated":
        return True
    if new.endswith(" city") and new.removesuffix(" city") == old:
        return True
    return any((old == normalized(m[0]) or old == m[1].casefold()) and new == (m[1][0] + m[2][0]).casefold()
               for m in QUALIFIED_DEGREE.finditer(question))


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.pending = None
        self.candidate = None
        self.requested_schema = False
        self.sample = None
        self.mode = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        question = observation["task"]["question"]
        if self.candidate:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.candidate["sql"]}
            if self.pending["tool"] == "submit":
                return self.pending
            return {"tool": "submit", "sql": observation["issue_sql"]}

        action = self.pending or self.baseline.act(observation)
        if self.sample and self.sample.get("retired"):
            return action
        if action["tool"] == "submit":
            sql = action["sql"]
        elif action["tool"] == "check" and (remaining == 2 or self.sample) and STATE.search(question):
            if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
                return action
            sql = observation["issue_sql"]
        else:
            return action
        tree, items = eligible_literals(sql, question)
        if not items or not remaining:
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if action["tool"] == "submit" and remaining >= 3 and not self.requested_schema:
                self.pending = action
                self.requested_schema = True
                return {"tool": "schema"}
            return action
        schema = Schema(tables)
        binding = bound_literals(tree, items, schema)
        if not binding:
            return action
        if self.sample is None:
            if remaining < 2:
                return action
            probe = sample_sql(schema, binding)
            if any(e["tool"] == "execute" and e["sql"] == probe for e in history):
                return action
            self.sample = {"sql": probe, "binding": binding}
            self.pending = action
            return {"tool": "execute", "sql": probe}

        sample = history[-1]["observation"]
        values = {row[0] for row in sample["rows"] if len(row) == 1 and isinstance(row[0], str)} if sample["ok"] else set()
        self.sample["values"] = sorted(values)
        edits = []
        for _, literal, kind in items:
            old = literal.this
            matches = {value for value in values if equivalent(old, value, question, kind)}
            if kind == "login_state" and values != {"0", "1"}:
                continue
            if old in matches or len(matches) != 1:
                continue
            new = next(iter(matches))
            literal.replace(exp.Literal.string(new))
            edits.append((old, new, kind))
        candidate = sql_text(tree) if edits else None
        seen = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        if candidate and identity(candidate) not in seen | {identity(action["sql"])}:
            self.mode = "suffix" if action["tool"] == "submit" else "reserved_last_slot"
            self.candidate = {"sql": candidate, "binding": binding, "edits": edits}
            return {"tool": "check", "sql": candidate}
        if action["tool"] != "submit":
            self.pending = None
            self.sample["retired"] = True
        return action


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
