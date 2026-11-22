"""Compile filtered queries from public headings, literal anchors and declared FK paths."""
from itertools import product
import re

from sqlglot import exp

from repair_rules import parse
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step8_groups import output_bindings, MEASURE
from step10_bindings import (
    ATOMS, column_evidence, compatible, existing, field_words, local_occurrences, tokens,
)
from step11_absence import parent_sqls
from step19_grounding import package

GENERIC = {"id", "key", "code", "name", "type", "value", "number"}
NEGATIVE = re.compile(r"\b(?:do not|does not|don['’]t|doesn['’]t|without|never)\b", re.I)
COMPLEX = re.compile(r"\b(?:each|every|average|fewest|least|highest|lowest|most|latest|earliest|both)\b", re.I)


def meaningful(name):
    return set(field_words(name)) - GENERIC


def output_options(tree, headings, schema, question):
    sources = top_tables(tree, schema)
    slots = []
    for ordinal, heading in enumerate(headings):
        label = heading.lower().strip()
        if MEASURE.fullmatch(label):
            slots.append([(14, None, None)])
            continue
        old = [(p.unalias(), existing(p.unalias(), sources, schema))
               for p in tree.selects if p.alias_or_name.lower() == label and isinstance(p.unalias(), exp.Column)]
        options = []
        for table, columns in schema.columns.items():
            for field in columns:
                names = {field, field.replace("_", " ")}
                bound = any(b and b[1:] == (table, field) for _, b in old)
                wanted = set(field_words(label))
                if label not in names and not bound and (not wanted or wanted != set(field_words(field))):
                    continue
                support = 3 * len(meaningful(table) & question) + 3 * len(meaningful(field) & question)
                options.append((12 + support + 3 * bound, table, field))
        options.sort(key=lambda opt: (-opt[0], opt[1], opt[2]))
        if not options:
            return []
        slots.append(options[:3])
    return [variant for variant in product(*slots)
            if len({(t, f) for _, t, f in variant if t}) == sum(t is not None for _, t, _ in variant)][:12]


def leaves(tree, question):
    where = tree.args.get("where")
    if not where:
        return []
    result = []
    for atom in where.find_all(*ATOMS):
        if not isinstance(atom.this, exp.Column) or atom.find(exp.Select):
            continue
        values = list(atom.find_all(exp.Literal))
        if not values:
            continue
        positions = [(i, i + len(tokens(v.this))) for v in values
                     for i in local_occurrences(question, v.this)]
        if len({v.this for v in values if local_occurrences(question, v.this)}) != len({v.this for v in values}):
            continue
        result.append((atom, positions))
    return result


def field_options(atom, positions, tree, schema, question):
    sources = top_tables(atom.find_ancestor(exp.Select) or tree, schema)
    old = existing(atom.this, sources, schema)
    old_name = set(field_words(atom.this.name))
    nearby = {w for i, end in positions for w in question[max(0, i - 5):end + 3]}
    options = []
    for table, columns in schema.columns.items():
        for field, meta in columns.items():
            if not compatible(atom, meta):
                continue
            words = set(field_words(field))
            explicit = column_evidence(question, field, positions)
            overlap = meaningful(field) & nearby
            same = field == atom.this.name.lower()
            source = bool(old and old[1:] == (table, field))
            if not (explicit or same or overlap or (old_name & meaningful(field))):
                continue
            score = min(explicit, 32) + 6 * len(overlap) + 5 * same + 4 * source
            score += 2 * len(meaningful(table) & set(question))
            options.append((score, table, field))
    return sorted(options, key=lambda o: (-o[0], o[1], o[2]))[:4]


def relation_plans(root, targets, schema, question, depth):
    states = [({root}, (), 0)]
    for target in sorted(targets - {root}):
        updated = []
        for present, prior, merit in states:
            for path in sorted({tuple(p) for p in schema.path_options(sorted(present), target, depth)}):
                roles = set().union(*(meaningful(a) | meaningful(b) | meaningful(t)
                                      for _, a, t, b in path))
                rank = 3 * len(roles & question) - 6 * len(path)
                updated.append((present | {p[2] for p in path}, prior + path, merit + rank))
        states = sorted(updated, key=lambda item: (-item[2], item[1]))[:5]
        if not states:
            break
    return states


def bound_atom(atom, field, schema, aliases):
    _, table, column = field
    result = atom.copy()
    result.set("this", schema.column(table, column, aliases[table]))
    numeric = re.match(r"(?:int|real|dec|num|float|double)", schema.columns[table][column]["type"], re.I)
    if numeric:
        for value in result.find_all(exp.Literal):
            if value.is_string and re.fullmatch(r"-?\d+(?:\.\d+)?", value.this):
                value.replace(exp.Literal.number(value.this))
    return result


def projection(query, headings, variant, schema, aliases):
    output = []
    fallback = None
    for ordinal, (_, table, field) in enumerate(variant):
        if table:
            output.append(schema.column(table, field, aliases[table]))
        else:
            if fallback is None:
                fallback = output_bindings(query, schema, headings)
            if fallback is None or not isinstance(fallback[ordinal], exp.AggFunc):
                return None
            output.append(fallback[ordinal])
    return output


def compile_query(tree, schema, headings, question, depth):
    if not isinstance(tree, exp.Select) or tree.args.get("with"):
        return
    if any(tree.args.get(k) for k in ("group", "having", "order", "limit", "offset")):
        return
    words = tokens(question)
    proof = leaves(tree, words)
    anti = bool(NEGATIVE.search(question) and len(list(tree.find_all(exp.Select))) > 1)
    if COMPLEX.search(question) and not any(MEASURE.fullmatch(h.lower().strip()) for h in headings):
        return
    if not anti and len(list(tree.find_all(exp.Select))) > 1:
        return
    if not proof and tree.args.get("where"):
        return
    if len(proof) > 3 or (anti and len(proof) != 1):
        return
    # Preserve boolean intent only when the draft has an explicit matching connective.
    ask_or = bool(re.search(r"\bor\b", question, re.I))
    if len(proof) > 1 and not ask_or and tree.args["where"].find(exp.Or):
        return
    if len(proof) > 1 and ask_or and not tree.args["where"].find(exp.Or):
        return
    opts = [field_options(atom, positions, tree, schema, words) for atom, positions in proof]
    if any(not o for o in opts):
        return
    filters = sorted(product(*opts), key=lambda v: -sum(o[0] for o in v))[:10]
    variants = output_options(tree, headings, schema, set(words))
    physical = top_tables(tree, schema)
    for fields in variants:
        outputs = {t for _, t, _ in fields if t}
        for bound in filters:
            targets = outputs | {t for _, t, _ in bound}
            roots = sorted(outputs or set(physical.values()) & targets or targets)
            # With no filter entity or join to repair, leave the parent alone.
            if not roots or (len(targets) == 1 and len(physical) < 2 and not proof):
                continue
            for root in roots[:1]:
                for _, path, merit in relation_plans(root, targets, schema, set(words), depth):
                    query = exp.select().from_(physical_table(schema, root, "r0"))
                    aliases = {root: "r0"}
                    if anti and (outputs - {root} or not path or bound[0][1] == root):
                        continue
                    if anti:
                        origin, left, start, right = path[0]
                        aliases[start] = "r1"
                        sub = exp.select("1").from_(physical_table(schema, start, "r1"))
                        join_path(sub, path[1:], schema, aliases)
                        correlation = exp.EQ(this=schema.column(origin, left, "r0"),
                                             expression=schema.column(start, right, "r1"))
                        sub.where(exp.and_(correlation, bound_atom(proof[0][0], bound[0], schema, aliases)),
                                  copy=False)
                        clause = exp.Not(this=exp.Exists(this=sub))
                    else:
                        join_path(query, path, schema, aliases)
                        clauses = [bound_atom(atom, field, schema, aliases) for (atom, _), field in zip(proof, bound)]
                        if len(clauses) > 1:
                            clause = exp.or_(*clauses) if ask_or else exp.and_(*clauses)
                        else:
                            clause = clauses[0] if clauses else None
                            if (clause is not None and NEGATIVE.search(question) and isinstance(clause, exp.EQ)
                                    and isinstance(clause.expression, exp.Literal)):
                                clause = exp.NEQ(this=clause.this.copy(), expression=clause.expression.copy())
                    output = projection(query, headings, fields, schema, aliases)
                    if output is None:
                        continue
                    query.select(*output, append=False, copy=False)
                    if clause is not None:
                        query.where(clause, copy=False)
                    if tree.args.get("distinct") or re.search(r"\b(?:distinct|unique)\b", question, re.I):
                        query.distinct(copy=False)
                    rank = 75 + sum(p[0] for p in fields) + sum(p[0] for p in bound) + merit
                    evidence = {"heading_bindings": [(t, f) for _, t, f in fields],
                                "predicate_bindings": [(t, f) for _, t, f in bound],
                                "foreign_keys": path, "anti": anti}
                    yield query, rank, "path_binding", evidence


def path_pool(obs, fallback, schema, depth):
    raw = []
    question = obs["task"]["question"]
    for sql in parent_sqls(obs, fallback)[:12]:
        tree = parse(sql)
        raw.extend(compile_query(tree, schema, obs["example"]["columns"], question, depth))
    # Re-aliasing an already tried join can disguise repeat probes with the same semantics.
    seen = set()
    pool = []
    for candidate in package(obs, fallback, raw):
        tree = parse(candidate["sql"])
        key = (tuple(t.name.lower() for t in tree.find_all(exp.Table)),
               tuple((c.table.lower(), c.name.lower()) for c in tree.find_all(exp.Column)),
               tuple((type(n).__name__, n.this) for n in tree.find_all(exp.Literal)),
               tuple(type(n).__name__ for n in tree.walk()))
        if key not in seen:
            seen.add(key)
            pool.append(candidate)
    return pool[:18]
