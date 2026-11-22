"""Generate a single-table measure candidate when the question fixes its scope."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step5x_edits import physical_table
from step36.frequency import terms
from step38.event_grain import bind_column
from step48_agent import create_agent as create_baseline


TOTAL = re.compile(r"\b(?:total|sum)\b.+?\b(?:across|of)\s+all\s+(?:the\s+)?([\w -]+)\s*[?.!]?\s*$", re.I)
SUM_HEADING = re.compile(r"sum\s*\(\s*([a-z_][\w]*)\s*\)", re.I)
EXTREMUM = re.compile(r"\b(?:the\s+)?(highest|lowest|maximum|minimum)\s+([\w -]+)\s*[?.!]?\s*$", re.I)
ORDERED_LIST = re.compile(r"^\s*(?:list|show|give|display)\b.+?\bin\s+(ascending|descending)\s+order\s*[?.!]?\s*$", re.I)
EXTRA_SCOPE = re.compile(r"\b(?:each|per|where|between|either|except|whose|who|that|before|after|least|over|under|distinct|unique)\b", re.I)
NUMERIC = re.compile(r"int|real|float|double|numeric|decimal|number", re.I)


def request(observation, tree):
    if not isinstance(tree, exp.Select) or tree.args.get("distinct"):
        return None
    question = observation["task"]["question"]
    if EXTRA_SCOPE.search(question):
        return None
    headings = observation["example"]["columns"]
    sums = len(tree.expressions) == 1 and isinstance(tree.expressions[0].unalias(), exp.Sum)
    total = TOTAL.search(question)
    if (sums and total and len(headings) == 1 and SUM_HEADING.fullmatch(headings[0]) and tree.args.get("joins")
            and not any(tree.args.get(k) for k in ("where", "group", "having", "order", "limit", "with"))):
        return "total", total
    ordered = ORDERED_LIST.search(question)
    if (sums and ordered and len(headings) == 1 and re.fullmatch(r"[\w]+", headings[0])
            and not any(tree.args.get(k) for k in ("group", "having", "with", "limit", "order"))
            and not tree.find(exp.Literal)):
        return "stored_measure", ordered
    extreme = EXTREMUM.search(question)
    if (extreme and tree.args.get("joins") and len(tree.expressions) == len(headings)
            and all(isinstance(p.unalias(), exp.Column) for p in tree.expressions)
            and not any(tree.args.get(k) for k in ("group", "having", "offset", "with"))
            and not (tree.args.get("limit") and tree.args["limit"].expression != exp.Literal.number(1))):
        return "extremum", extreme
    return None


def named(table, scope):
    role = terms(table)
    return bool(role and role <= terms(scope))


def source_plan(observation, parent, intent, tables):
    mode, cue = intent
    schema = Schema(tables)
    question = observation["task"]["question"]
    headings = [heading.casefold() for heading in observation["example"]["columns"]]
    sources = schema.sources(parent)
    if mode == "extremum":
        # Limit the change to an existing extremum predicate over the output
        # table. Other filters, joins named in the question and tie rules are
        # evidence of a broader scope and prevent a single-table rewrite.
        visible = parent.args.get("where")
        atom = visible.this if visible else None
        if not isinstance(atom, exp.EQ) or not isinstance(atom.expression, exp.Subquery):
            return None
        sub = atom.expression.this
        greatest = cue[1].casefold() in {"highest", "maximum"}
        agg = exp.Max if greatest else exp.Min
        if (not isinstance(sub, exp.Select) or len(sub.expressions) != 1
                or not isinstance(sub.expressions[0].unalias(), agg)
                or any(sub.args.get(k) for k in ("joins", "where", "group", "having", "limit", "with"))):
            return None
        outer = parent.copy()
        outer.set("where", None)
        sources = schema.sources(outer)
        carrier = bind_column(atom.this, schema, sources)
        measure = bind_column(sub.expressions[0].unalias().this, schema, schema.sources(sub))
        if not carrier or carrier != measure or terms(carrier[1]) != terms(cue[2]):
            return None
        root, field = carrier
        projections = [bind_column(p.unalias(), schema, sources) for p in parent.expressions]
        if (not named(root, question) or any(named(table, question) for table in sources if table != root)
                or any(p is None or p[0] != root for p in projections)
                or [p.alias_or_name.casefold() for p in parent.expressions] != headings):
            return None
        query = exp.select(*(schema.column(root, p[1], "measure_root") for p in projections))
        metric = schema.column(root, field, "measure_root")
        query.where(exp.Not(this=exp.Is(this=metric.copy(), expression=exp.Null())), copy=False)
        query.order_by(exp.Ordered(this=metric, desc=greatest, nulls_first=False), copy=False).limit(1, copy=False)
    else:
        field = SUM_HEADING.fullmatch(headings[0])[1].casefold() if mode == "total" else headings[0]
        scope = cue[1] if mode == "total" else question
        # A projected stored measure must be named in the requested output,
        # and its draft SUM must be over that same unambiguous column.
        candidates = [table for table, fields in schema.columns.items() if field in fields and named(table, scope)
                      and NUMERIC.search(fields[field]["type"])]
        if len(candidates) != 1:
            return None
        root = candidates[0]
        column = schema.column(root, field, "measure_root")
        if mode == "total":
            if root not in sources or any(named(table, scope) for table in sources if table != root):
                return None
            query = exp.select(exp.Sum(this=column))
        else:
            wanted = terms(field)
            old_col = parent.expressions[0].unalias().this
            source = parent.args.get("from")
            if (not wanted or not wanted <= terms(question) or not isinstance(old_col, exp.Column)
                    or old_col.name.casefold() != field or not source or not isinstance(source.this, exp.Table)
                    or source.this.name.casefold() != root
                    or any(named(t.name, question) for t in parent.find_all(exp.Table) if t.name.casefold() != root)):
                return None
            query = exp.select(column)
            desc = cue[1].casefold() == "descending"
            query.order_by(exp.Ordered(this=column.copy(), desc=desc, nulls_first=not desc), copy=False)
    return {"sql": sql_text(query.from_(physical_table(schema, root, "measure_root"), copy=False)),
            "kind": mode, "root": root, "measure": field}


class MeasureSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.requested = False

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            return {"tool": "submit", "sql": self.candidate["sql"]} if passed(history[-1]["observation"]) else self.fallback
        parent = parse(self.fallback["sql"])
        intent = request(observation, parent) if remaining else None
        if intent is None:
            return self.fallback
        tables = next((event["observation"] for event in history if event["tool"] == "schema"), None)
        if tables is None:
            if remaining >= 2 and not self.requested:
                self.requested = True
                return {"tool": "schema"}
            return self.fallback
        candidate = source_plan(observation, parent, intent, tables)
        previous = {identity(event["sql"]) for event in history if event["tool"] == "check"}
        if candidate and identity(candidate["sql"]) not in previous | {identity(self.fallback["sql"])}:
            self.candidate = candidate
            return {"tool": "check", "sql": candidate["sql"]}
        return self.fallback


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = MeasureSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
