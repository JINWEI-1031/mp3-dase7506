"""Bind a misplaced NOT IN predicate to a question-named witness key."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step32_plan import context
from step36.frequency import terms
from step40.witness import NEGATIVE, analyze, source
from step63_agent import create_agent as create_baseline


ABSENCE_NOUN = re.compile(
    r"\b(?:without|(?:do(?:es)?|did) not have|(?:don|doesn|didn)[’']t have|(?:have|has|had|with) no)"
    r"\s+(?:any\s+)?([a-z][a-z -]*)\s*[?.!]?$", re.I)
AMBIGUOUS = {"and", "or", "with", "in", "under", "over", "than", "where", "during", "before", "after"}
SIMPLE_OUTER = {"expressions", "from", "where", "distinct"}


def negative_in(sql):
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or any(v for k, v in tree.args.items() if k not in SIMPLE_OUTER):
        return None
    root = source(tree)
    where = tree.args.get("where")
    atom = where.this.unnest() if where else None
    if (root is None or root.db or root.catalog
            or not isinstance(atom, exp.Not) or not isinstance(atom.this.unnest(), exp.In)):
        return None
    membership = atom.this.unnest()
    query = membership.args.get("query")
    key = membership.this
    if (not isinstance(key, exp.Column) or key.table.casefold() not in ("", root.alias_or_name.casefold())
            or not isinstance(query, exp.Subquery) or not isinstance(query.this, exp.Select)
            or len(query.this.expressions) != 1):
        return None
    return tree, root, key, query.this


def noun_intent(question, root, inner):
    match = ABSENCE_NOUN.search(question)
    relation = source(inner)
    if match is None or relation is None or not terms(root.name) & terms(question[:match.start()]):
        return None
    role = terms(match[1])
    if role and not AMBIGUOUS & role and role <= terms(relation.name):
        return role
    return None


def projected_witness_key(inner):
    if any(v for k, v in inner.args.items() if k not in {"expressions", "from", "distinct"}):
        return None
    witness = source(inner)
    selected = inner.expressions[0].unalias()
    if (witness is None or witness.db or witness.catalog or not isinstance(selected, exp.Column)
            or selected.table.casefold() not in ("", witness.alias_or_name.casefold())):
        return None
    return selected.name.casefold()


def noun_candidate(shape, selected, schema):
    tree, root, key, inner = shape
    witness = source(inner)
    source_name, witness_name = root.name.casefold(), witness.name.casefold()
    if selected not in schema.columns.get(witness_name, {}):
        return None
    root_key = key.name.casefold()
    fields = schema.columns.get(source_name, {})
    if source_name == witness_name or [name for name, meta in fields.items() if meta["pk"]] != [root_key]:
        return None
    links = {fk["from"].casefold() for fk in schema.tables.get(witness_name, {}).get("foreign_keys", [])
             if fk["table"].casefold() == source_name and fk["to"].casefold() == root_key}
    if len(links) != 1:
        return None
    (reference,) = links
    if selected == reference:
        return None
    label = "absence_witness" if root.alias_or_name.casefold() != "absence_witness" else "exclusion_witness"
    witness = witness.copy()
    witness.set("alias", exp.TableAlias(this=exp.to_identifier(label)))
    link = exp.EQ(this=schema.column(witness_name, reference, label),
                  expression=schema.column(source_name, root_key, root.alias_or_name))
    child = exp.select(exp.Literal.number(1)).from_(witness).where(link)
    result = tree.where(exp.Not(this=exp.Exists(this=child)), append=False)
    return {"sql": sql_text(result), "kind": "declared_witness_key",
            "key_path": [[source_name, root_key, witness_name, reference]]}


def named_role_candidate(observation, shape, ctx):
    draft = negative_in(observation["issue_sql"])
    tree, root, key, inner = shape
    projection = [expr.unalias() for expr in tree.expressions]
    if (draft is None or draft[1].name.casefold() != root.name.casefold()
            or draft[2].name.casefold() != key.name.casefold()
            or tree.args.get("distinct")
            or any(not isinstance(col, exp.Column) or col.is_star
                   or col.table.casefold() not in ("", root.alias_or_name.casefold()) for col in projection)
            or [(expr.unalias().name.casefold(), expr.alias_or_name.casefold()) for expr in draft[0].expressions]
            != [(expr.unalias().name.casefold(), expr.alias_or_name.casefold()) for expr in tree.expressions]):
        return None, False
    proposal, audit = analyze(ctx)
    request_schema = bool(audit.get("request"))
    if proposal is None:
        return None, request_schema
    plan = proposal["basis"]["plan"]
    path = plan["key_path"]
    if len(path) != 1 or plan["output_key"] != key.name.casefold():
        return None, False
    witness = source(inner)
    selected = inner.expressions[0].unalias()
    if (witness is not None and witness.name.casefold() == path[0][2]
            and isinstance(selected, exp.Column) and selected.name.casefold() == path[0][3]):
        return None, False
    return {"sql": proposal["sql"], "kind": "named_witness_role",
            "key_path": path, "key_evidence": plan["key_evidence"]}, False


class WitnessSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.candidate}
            return self.fallback
        if not remaining:
            return self.fallback
        question = observation["task"]["question"]
        if not (NEGATIVE.search(question) or ABSENCE_NOUN.search(question)):
            return self.fallback
        shape = negative_in(self.fallback["sql"])
        if shape is None:
            return self.fallback
        ctx = context(observation)
        noun = noun_intent(question, shape[1], shape[3])
        if noun:
            selected = projected_witness_key(shape[3])
            if selected is None or selected == shape[2].name.casefold():
                return self.fallback
            candidate = noun_candidate(shape, selected, Schema(ctx["schema"])) if ctx["schema"] is not None else None
            request_schema = ctx["schema"] is None
        else:
            candidate, request_schema = named_role_candidate(observation, shape, ctx)
        if request_schema and remaining >= 2:
            return {"tool": "schema"}
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        sql = candidate["sql"]
        checked = next((result for old, result in ctx["checks"] if identity(old) == identity(sql)), None)
        if checked is not None:
            if passed(checked):
                return {"tool": "submit", "sql": sql}
            return self.fallback
        self.candidate = sql
        return {"tool": "check", "sql": sql}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = WitnessSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
