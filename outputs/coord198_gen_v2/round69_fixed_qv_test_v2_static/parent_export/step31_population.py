"""Fix a quantified comparison without changing the outer result grain."""
from dataclasses import dataclass
import re

from sqlglot import exp

from repair_rules import parse
from step4_edits import Schema
from step7_semantics import top_tables
from step10_bindings import existing, field_words, local_occurrences, tokens
from step19_grounding import package
from step21_carry import conjuncts
from step29_scope import declared_edges


QUANTIFIER = re.compile(
    r"\b(before|after|older than|younger than|greater than|less than|higher than|lower than"
    r"|earlier than|later than|more than|smaller than)\b([^?.;]{0,64}?)\b(any|all|every)\b",
    re.I,
)
LESS = {"before", "younger than", "less than", "lower than", "earlier than", "smaller than"}
FIRST_DATE = re.compile(r"\bbefore\s+(?:the\s+)?date\s+of\s+any\b", re.I)


@dataclass(frozen=True)
class Population:
    outer: exp.Select
    population: exp.Select
    left: exp.Column
    right: exp.Column
    quantifier: str
    comparison: type
    evidence: str
    boundary: type | None = None

    @property
    def needs_schema(self):
        return self.evidence == "checked_quantifier"


def source(select):
    node = select.args.get("from")
    if node and isinstance(node.this, exp.Table) and not (node.this.db or node.this.catalog):
        return node.this
    return None


def select_body(tree):
    if not isinstance(tree, exp.Select) or not source(tree):
        return None
    if any(tree.args.get(k) for k in ("group", "having", "limit", "offset", "with", "windows")):
        return None
    where = tree.args.get("where")
    return where.this if where else None


def qualification(select, tail):
    where = select.args.get("where")
    if where is None:
        return False
    return all(isinstance(n, exp.EQ) and isinstance(n.this, exp.Column)
               and isinstance(n.expression, exp.Literal) and local_occurrences(tail, n.expression.this)
               for n in conjuncts(where.this))


def value_pair(left, right, question, comparison_phrase):
    if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
        return False
    if left.name.lower() != right.name.lower():
        return False
    words = set(field_words(left.name))
    if words & set(field_words(question)):
        return True
    return "age" in words and comparison_phrase in ("older than", "younger than")


def population_plan(obs, fallback):
    question = obs["task"]["question"]
    intent = QUANTIFIER.search(question)
    if not intent or re.search(r"\b(?:some|or)\b", question[intent.start():intent.end()], re.I):
        return None
    phrase, _, quantifier = (v.lower() for v in intent.groups())
    comparison = exp.LT if phrase in LESS else exp.GT
    tail = tokens(question[intent.end():])
    # A dated comparison to the first occurrence and a natural-language
    # comparison to every comparable value both require a nonempty boundary.
    boundary = (exp.Min if FIRST_DATE.search(question) else
                (exp.Min if comparison is exp.LT else exp.Max) if quantifier != "any" else None)
    outer = parse(fallback)
    atom = select_body(outer)
    if atom is None or outer.args.get("joins") or not isinstance(atom, (comparison, exp.EQ)):
        return None
    rhs = atom.expression
    if not isinstance(atom.this, exp.Column) or not isinstance(rhs, exp.Subquery):
        return None
    original_source = source(outer)
    if atom.this.table and atom.this.table.lower() != original_source.alias_or_name.lower():
        return None
    old = rhs.this
    if (not select_body(old) or old.args.get("joins") or len(old.selects) != 1
            or not isinstance(old.selects[0], exp.Column) or not qualification(old, tail)):
        return None
    if isinstance(atom, comparison) and value_pair(atom.this, old.selects[0], question, phrase):
        local = source(old)
        if local.name.lower() != original_source.name.lower():
            return None
        columns = [old.selects[0], *(n.this for n in conjuncts(old.args["where"].this))]
        if any(c.table and c.table.lower() not in {
                local.alias_or_name.lower(), original_source.alias_or_name.lower()} for c in columns):
            return None
        return Population(outer, old, atom.this, old.selects[0], quantifier, comparison,
                          "same_table_scalar", boundary)
    if not isinstance(atom, exp.EQ):
        return None
    quantified = exp.Any if quantifier == "any" else exp.All
    # Only reuse a quantified sketch which was already checked and could not
    # execute. Its syntax must match the stated operator and quantifier.
    for event in obs["history"]:
        if event["tool"] != "check" or event["observation"]["ok"]:
            continue
        hint = parse(event["sql"])
        intent_atom = select_body(hint)
        if not isinstance(intent_atom, comparison) or not isinstance(intent_atom.expression, quantified):
            continue
        if hint.args.get("joins") or source(hint).name.lower() != original_source.name.lower():
            continue
        probe = intent_atom.expression.this
        inner = probe.this if isinstance(probe, exp.Subquery) else probe
        if not select_body(inner) or len(inner.selects) != 1 or not qualification(inner, tail):
            continue
        if source(inner).name.lower() != original_source.name.lower():
            continue
        if (isinstance(intent_atom.this, exp.Column)
                and intent_atom.this.table.lower() in ("", source(hint).alias_or_name.lower())
                and value_pair(intent_atom.this, inner.selects[0], question, phrase)):
            return Population(outer, inner, intent_atom.this, inner.selects[0], quantifier, comparison,
                              "checked_quantifier", boundary if quantifier == "any" else None)
    return None


def rebind_population(plan, tables):
    sub = plan.population.copy()
    outer_alias = source(plan.outer).alias_or_name
    local = source(sub)
    if not plan.needs_schema:
        independent = "population" if outer_alias.lower() != "population" else "inner_population"
        local.set("alias", exp.TableAlias(this=exp.to_identifier(independent)))
        for c in sub.find_all(exp.Column):
            c.set("table", exp.to_identifier(independent, quoted=True))
        return sub
    schema = Schema(tables)
    scopes = top_tables(sub, schema)
    edges = declared_edges(sub, scopes, schema)
    root = local.name.lower()
    if len(edges) != len(sub.args.get("joins") or []) or any(
            not existing(c, scopes, schema) for c in sub.find_all(exp.Column)):
        return None
    if not edges or root not in scopes.values():
        return None
    right = existing(sub.selects[0], scopes, schema)
    if right[1:] != (root, plan.left.name.lower()):
        return None
    # The checked sketch must qualify the same relationship and population
    # as the scalar key comparison, rather than introduce a different role.
    old_atom = plan.outer.args["where"].this
    old_population = old_atom.expression.this
    old_table = source(old_population).name.lower()
    old_key = old_population.selects[0]
    expected = (root, old_atom.this.name.lower(), old_table, old_key.name.lower())
    if expected not in edges and (expected[2], expected[3], expected[0], expected[1]) not in edges:
        return None
    if len(edges) != 1 or outer_alias.lower() in scopes:
        return None
    return sub


def candidate_pool(obs, fallback, plan, tables=None):
    sub = rebind_population(plan, tables)
    if sub is None:
        return []
    left = plan.left.copy()
    left.set("table", exp.to_identifier(source(plan.outer).alias_or_name, quoted=True))
    right = sub.selects[0].copy()
    sub.set("order", None)
    if plan.boundary:
        sub.select(plan.boundary(this=right), append=False, copy=False)
        test = plan.comparison(this=left, expression=sub.subquery())
    else:
        predicate = plan.comparison(this=left, expression=right)
        if plan.quantifier != "any":
            # An explicit SQL ALL accepts only when no comparison is false or unknown.
            predicate = exp.or_(exp.Not(this=exp.Paren(this=predicate)),
                                exp.Is(this=left.copy(), expression=exp.Null()),
                                exp.Is(this=right.copy(), expression=exp.Null()))
        sub.select(exp.Literal.number(1), append=False, copy=False)
        sub.where(predicate, copy=False)
        exists = exp.Exists(this=sub)
        test = exists if plan.quantifier == "any" else exp.Not(this=exists)
    query = plan.outer.copy().where(test, append=False, copy=False)
    return package({"history": []}, fallback, [(query, 180, "quantified_population", {
        "quantifier": plan.quantifier, "comparison": plan.comparison.key,
        "boundary": plan.boundary.key if plan.boundary else None,
        "value_field": plan.left.name, "population_source": source(sub).name,
        "evidence": plan.evidence, "outer_select_joins_and_distinct": "preserved",
    })])
