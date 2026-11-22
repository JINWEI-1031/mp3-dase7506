"""Bind output and count fields to the explicitly requested entity role."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step5x_edits import physical_table
from step36.frequency import terms
from step38.event_grain import bind_column, join_graph
from step48_agent import create_agent as create_baseline


DISPLAY = re.compile(r"\bfor the ([\w-]+) (?:managed|directed|owned|operated|led|supervised) by\b", re.I)
ABSENT_FEATURE = re.compile(r"^\s*how many ([\w -]+?) (?:do not|don't) have (?:the )?([\w -]+?) function\s*[?.!]?$", re.I)


def intent_for(question):
    projection = DISPLAY.search(question)
    if projection:
        return "projection", (projection[1],)
    count = ABSENT_FEATURE.fullmatch(question)
    return ("count", count.groups()) if count else None


def named_role(schema, role, fields):
    words = terms(role)
    if not words:
        return None
    matching = [table for table, cols in schema.columns.items() if words <= terms(table) and fields <= cols.keys()]
    exact = [table for table in matching if words == terms(table)]
    available = exact or matching
    return available[0] if len(available) == 1 else None


def projection_plan(obs, fallback, schema, role):
    parent = parse(fallback)
    heads = [heading.casefold() for heading in obs["example"]["columns"]]
    if (not isinstance(parent, exp.Select) or len(heads) != len(parent.expressions) or len(heads) < 2
            or len(set(heads)) != len(heads) or any(parent.args.get(k) for k in ("group", "having", "order", "limit", "with", "distinct"))):
        return None
    root = named_role(schema, role, set(heads))
    sources = schema.sources(parent)
    if root not in sources or not join_graph(parent, schema, sources):
        return None
    before = [bind_column(p.unalias(), schema, sources) for p in parent.expressions]
    if any(field is None for field in before) or [name for _, name in before] != heads:
        return None
    if not any(table == root for table, _ in before) or not any(table != root for table, _ in before):
        return None
    query = parent.copy().select(*(schema.column(root, head, sources[root]) for head in heads), append=False, copy=False)
    return {"sql": sql_text(query), "kind": "projection_role", "entity": root}


def compact(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def count_plan(obs, schema, role, feature):
    parent = parse(obs["issue_sql"])
    if (not isinstance(parent, exp.Select) or len(parent.expressions) != 1
            or not isinstance(parent.expressions[0].unalias(), exp.Count)
            or not isinstance(parent.expressions[0].unalias().this, exp.Star)
            or [compact(h) for h in obs["example"]["columns"]] != ["count"]
            or any(parent.args.get(k) for k in ("joins", "group", "having", "with", "limit", "distinct"))):
        return None
    where, from_ = parent.args.get("where"), parent.args.get("from")
    atom = where.this.unnest() if where else None
    if (not from_ or not isinstance(from_.this, exp.Table) or not isinstance(atom, exp.EQ)
            or not isinstance(atom.this, exp.Column) or compact(atom.this.name) != compact(feature)
            or not isinstance(atom.expression, exp.Literal) or not atom.expression.is_string
            or atom.expression.this.casefold() not in {"no", "false"}):
        return None
    field = atom.this.name.casefold()
    root = named_role(schema, role, {field})
    if not root or root == from_.this.name.casefold():
        return None
    query = exp.select(exp.Count(this=exp.Star())).from_(physical_table(schema, root, "subject"))
    query.where(exp.EQ(this=schema.column(root, field, "subject"), expression=atom.expression.copy()), copy=False)
    return {"sql": sql_text(query), "kind": "count_role", "entity": root, "condition": field}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.candidate = None
        self.mode = None
        self.fallback = None
        self.requested = False

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.candidate["sql"]}
            if self.fallback:
                return self.fallback
            action = self.baseline.act(observation)
            return action if action["tool"] == "submit" else {"tool": "submit", "sql": observation["issue_sql"]}

        action = self.fallback or self.baseline.act(observation)
        if not remaining or (action["tool"] != "submit" and (action["tool"] != "check" or remaining != 1)):
            return action
        intent = intent_for(observation["task"]["question"])
        if intent is None:
            return action
        kind, words = intent
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            parent = parse(action["sql"])
            if (kind == "projection" and action["tool"] == "submit" and remaining >= 2 and not self.requested
                    and isinstance(parent, exp.Select) and parent.args.get("joins") and any(
                        terms(words[0]) == terms(t.name) for t in parent.find_all(exp.Table))):
                self.fallback = action
                self.requested = True
                return {"tool": "schema"}
            return action
        schema = Schema(tables)
        candidate = (projection_plan(observation, action["sql"], schema, *words) if kind == "projection"
                     else count_plan(observation, schema, *words))
        if not candidate or identity(candidate["sql"]) in {identity(e["sql"]) for e in history if e["tool"] == "check"} | {identity(action["sql"])}:
            return action
        if action["tool"] == "check":
            scheduled = parse(action["sql"])
            if (kind != "count" or not isinstance(scheduled, exp.Select) or scheduled.args.get("where")
                    or any(e["tool"] == "check" and passed(e["observation"]) for e in history)):
                return action
            self.mode = "replace"
        else:
            self.fallback = action
            self.mode = "append"
        self.candidate = candidate
        return {"tool": "check", "sql": candidate["sql"]}


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
