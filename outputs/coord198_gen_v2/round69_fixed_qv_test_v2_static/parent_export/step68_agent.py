"""Compare last-slot relation evidence and normalize grounded SQLite literals."""

import os
import re

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify

from repair_rules import parse, sql_text
from step21_carry import conjuncts
from step36.frequency import terms
from step63_agent import create_agent as create_d
from step67_agent import Agent as LastSlotAgent, candidate_status


ARMS = {"control": (False, False), "evidence": (True, False),
        "literals": (False, True), "combined": (True, True)}
ABSENCE = re.compile(r"\b(?:no|without|never|(?:do|does|did) not have|not involved in)\s+([^?.!]+)", re.I)
COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)


def sqlite_literals(sql, question, tables):
    tree = parse(sql)
    if tree is None:
        return sql
    visible_columns = {str(c["name"]).casefold() for table in tables for c in table["columns"]}
    visible_columns.update(node.alias.casefold() for node in tree.find_all(exp.Alias))
    visible_columns.update(column.name.casefold() for node in tree.find_all(exp.TableAlias)
                           for column in node.args.get("columns", []))
    changed = False
    for node in tree.walk():
        if not isinstance(node, exp.Column):
            continue
        token = node.this
        parent = node.parent
        if (not isinstance(token, exp.Identifier) or not token.args.get("quoted")
                or node.args.get("table") or node.db or node.catalog or token.name.casefold() in visible_columns):
            continue
        position = token.meta.get("start")
        if position is None or sql[position:position + 1] != '"':
            continue
        if isinstance(parent, COMPARISONS):
            value_context = parent.expression is node and isinstance(parent.this, exp.Column)
        elif isinstance(parent, exp.In):
            value_context = isinstance(parent.this, exp.Column) and node.arg_key == "expressions"
        else:
            value_context = False
        value = token.name
        if value_context and value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", question, re.I):
            node.replace(exp.Literal.string(value))
            changed = True
    return sql_text(tree) if changed else sql


def sources(query):
    frm = query.args.get("from")
    relations = [frm.this] if frm else []
    relations.extend(join.this for join in query.args.get("joins", []))
    return {node.alias_or_name.casefold(): node.name.casefold() for node in relations if isinstance(node, exp.Table)}


def binding(column, aliases):
    if not isinstance(column, exp.Column) or column.table.casefold() not in aliases:
        return None
    return aliases[column.table.casefold()], column.name.casefold()


def linked(links, root, witnesses, edges):
    graph = {root}
    for _ in links:
        for a, b in links:
            if frozenset((a, b)) in edges and (a[0] in graph or b[0] in graph):
                graph.update((a[0], b[0]))
    return graph & witnesses - {root}


def equality_atoms(query):
    tests = [j.args["on"] for j in query.args.get("joins", []) if j.args.get("on")]
    where = query.args.get("where")
    if where:
        tests.append(where.this)
    for node in tests:
        for atom in conjuncts(node):
            if isinstance(atom, exp.EQ):
                yield atom


def equalities(query, aliases):
    for atom in equality_atoms(query):
        a, b = binding(atom.this, aliases), binding(atom.expression, aliases)
        if a and b:
            yield a, b


def keyed_absence(sql, question, tables):
    mention = ABSENCE.search(question)
    tree = parse(sql) if mention else None
    if tree is None:
        return set()
    role = terms(mention[1])
    schema = {table["name"]: {column["name"]: "UNKNOWN" for column in table["columns"]} for table in tables}
    witnesses = {name.casefold() for name in schema if terms(name) and terms(name) <= role}
    if not witnesses:
        return set()
    try:
        tree = qualify(tree, dialect="sqlite", schema=schema, infer_schema=False, identify=False)
    except (SqlglotError, RecursionError):
        return set()
    edges = {frozenset(((table["name"].casefold(), fk["from"].casefold()),
                        (fk["table"].casefold(), fk["to"].casefold())))
             for table in tables for fk in table["foreign_keys"]}
    if isinstance(tree, exp.Except) and all(isinstance(side, exp.Select) for side in (tree.this, tree.expression)):
        left, right = tree.this, tree.expression
        outer, inner = sources(left), sources(right)
        if len(outer) != 1:
            return set()
        left_fields = [binding(x.unalias(), outer) for x in left.selects]
        if not all(left_fields) or left_fields != [binding(x.unalias(), inner) for x in right.selects]:
            return set()
        root = next(iter(outer.values()))
        return {(root, name) for name in linked(list(equalities(right, inner)), root, witnesses, edges)}
    if not isinstance(tree, exp.Select) or not tree.args.get("where"):
        return set()
    outer = sources(tree)
    if len(outer) != 1:
        return set()
    root = next(iter(outer.values()))
    result = set()
    for atom in conjuncts(tree.args["where"].this):
        atom = atom.unnest()
        if not isinstance(atom, exp.Not):
            continue
        negative = atom.this.unnest()
        if isinstance(negative, exp.Exists) and isinstance(negative.this, exp.Select):
            query = negative.this
            inner = sources(query)
            if outer.keys() & inner.keys():
                continue
            correlated = any(
                isinstance(test.this, exp.Column) and isinstance(test.expression, exp.Column)
                and {test.this.table.casefold(), test.expression.table.casefold()} & outer.keys()
                and {test.this.table.casefold(), test.expression.table.casefold()} & inner.keys()
                for test in equality_atoms(query)
            )
            if not correlated:
                continue
            links = list(equalities(query, outer | inner))
        elif isinstance(negative, exp.In) and isinstance(negative.args.get("query"), exp.Subquery):
            query = negative.args["query"].this
            if not isinstance(query, exp.Select) or len(query.selects) != 1:
                continue
            inner = sources(query)
            a, b = binding(negative.this, outer), binding(query.selects[0].unalias(), inner)
            links = list(equalities(query, inner))
            if a and b:
                links.append((a, b))
        else:
            continue
        result.update((root, name) for name in linked(links, root, witnesses, edges))
    return result


class Agent(LastSlotAgent):
    def __init__(self, baseline, arm, proposer=None):
        if proposer is None:
            super().__init__(baseline)
        else:
            super().__init__(baseline, proposer)
        self.protect, self.normalize = ARMS[arm]
        self.normalized = False
        self.blocked_relations = set()

    def choose(self, sql, observation, scheduled, tables):
        question = observation["task"]["question"]
        if self.normalize:
            updated = sqlite_literals(sql, question, tables)
            self.normalized = updated != sql
            sql = updated
        status = candidate_status(sql, observation, scheduled, tables)
        if status == "replace" and self.protect:
            required = keyed_absence(scheduled, question, tables)
            if required:
                self.blocked_relations = required - keyed_absence(sql, question, tables)
            if self.blocked_relations:
                return sql, "protected_relation"
        return sql, status


def create_agent(checkpoint=None):
    return Agent(create_d(checkpoint), os.environ.get("MP3_STEP68_ARM", "combined"))
