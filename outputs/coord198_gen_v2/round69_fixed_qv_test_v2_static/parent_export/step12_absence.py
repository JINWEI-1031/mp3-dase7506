"""Recognize relation absence and bind a positive witness within its scope."""
from dataclasses import replace
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step7_semantics import terms, top_tables
from step10_bindings import ATOMS, field_words
from step11_absence import (ABSENCE, NOISE, Plan, add_observed_edges, bind_filters,
                            compile_plan, literal_specs, parent_sqls, root_query)

STRONG_ABSENCE = re.compile(r"\b(?:never|where no|no one|no [\w-]+ (?:has|have)|"
                            r"not (?:\w+ ){0,4}any|without any)\b", re.I)
MEMBERSHIP = re.compile(r"\bnot\s+in\b", re.I)
RANK = re.compile(r"\b(?:largest|smallest|top|bottom|biggest|highest|lowest|most|fewest|"
                  r"least|maximum|minimum)\b", re.I)
SCOPE_NOISE = NOISE | {"in", "on", "of", "by", "at", "an", "a", "is", "be", "for", "to", "did",
                       "do", "does", "and", "or", "been", "number", "type"}
LABEL_NOISE = {"name", "id", "code", "key", "label", "number", "description", "detail"}


def classify(question):
    """Keep membership in an explicitly ranked group outside this branch."""
    for match in ABSENCE.finditer(question):
        if MEMBERSHIP.search(match[0]) and not re.match(r"\s+any\b", question[match.end():], re.I):
            group = question[match.end():].split("?", 1)[0]
            return None, "ranked membership" if RANK.search(group) else "unquantified membership"
        return match, "absence"
    match = STRONG_ABSENCE.search(question)
    return (match, "absence") if match else (None, "outside absence trigger")


def lexical_forms(word):
    forms = {word}
    for suffix in ("ing", "ed", "es", "ation", "ate", "ment"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            stem = word[:-len(suffix)]
            forms.update((stem, stem + "e"))
            if stem[-1] == stem[-2]:
                forms.add(stem[:-1])
    return forms


def matching_words(left, right):
    normalized = set().union(*(lexical_forms(w) for w in right))
    return {word for word in left if lexical_forms(word) & normalized}


def mentioned(value, text):
    return bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I))


def observed_specs(tail, trees):
    """Lift literal IN lists into positive witness predicates, preserving polarity of the outer absence."""
    values = {spec[0].casefold(): spec for spec in literal_specs(tail, trees)}
    lists, spellings = [], {}
    # Scalar comparisons precede lists for an otherwise identical binding.
    for tree in trees:
        for node in tree.find_all(*ATOMS):
            if not isinstance(node.this, exp.Column):
                continue
            if isinstance(node, exp.In):
                operands = node.expressions
                if node.args.get("query") or not operands or not all(
                        isinstance(v, exp.Literal) and mentioned(v.this, tail) for v in operands):
                    continue
                keys = frozenset(v.this.casefold() for v in operands)
                if len(keys) > 1:
                    lists.append(keys)
            else:
                operands = [node.expression] if isinstance(node.expression, exp.Literal) else []
            field = node.this.name.lower()
            for literal in operands:
                value = literal.this.strip("%")
                if not value or not mentioned(value, tail):
                    continue
                key = value.casefold()
                values.setdefault(key, (value, not literal.is_string, exp.EQ, set()))
                values[key][3].add(field)
                spellings.setdefault(key, set()).add(value)
    return list(values.values()), lists, spellings


def disjunction_for(filters, lists, tail):
    if len(filters) < 2:
        return False
    keys = {f.value.casefold() for f in filters}
    same_field = len({(f.table, f.field) for f in filters}) == 1
    if any(keys == group for group in lists):
        return same_field or None
    if any(group & keys for group in lists):
        # A mixed list and scalar needs a nested Boolean expression.
        return None
    return bool(re.search(r"\bor\b", tail, re.I))


def labeled_universe(schema, root, select, prefix):
    if not all(isinstance(col, exp.Column) for col in select.selects):
        return False
    for col in select.selects:
        field = col.name.lower()
        words = set(field_words(field)) - LABEL_NOISE
        if not schema.columns[root][field]["pk"] and words and words <= prefix:
            return True
    return False


def ordinal_plans(obs, schema, fallback, match):
    question = obs["task"]["question"]
    # A following request for projected fields provides no relation-role evidence.
    tail = question[match.start():].split("?", 1)[0].split(". ", 1)[0]
    prefix = terms(question[:match.start()]) - SCOPE_NOISE
    focus = terms(tail) - SCOPE_NOISE
    if not focus:
        return
    headings = obs["example"]["columns"]
    parents = [(sql, parse(sql)) for sql in parent_sqls(obs, fallback)]
    parents = [(sql, tree) for sql, tree in parents if isinstance(tree, exp.Select)]
    trees = [tree for _, tree in parents]
    inferred = add_observed_edges(schema, trees)
    specs, lists, spellings = observed_specs(tail, trees)
    grouped = match[0].lower() == "where no"
    quantified = bool(STRONG_ABSENCE.search(tail))
    for root in schema.tables:
        selected = root_query(schema, root, headings)
        if selected is None:
            continue
        found = [sql for sql, tree in parents if root in top_tables(tree, schema).values()]
        words = terms(root)
        overlap = words & prefix
        if not (found or overlap):
            continue
        parent = found[0] if found else fallback
        root_score = 9 * len(overlap) + 4 * bool(found) + 8 * (words <= prefix)
        if len(words) > 1 and overlap != words:
            root_score -= 3
        label = grouped or quantified and labeled_universe(schema, root, selected, prefix)
        if label and specs and all(isinstance(p, exp.Column) for p in selected.selects):
            filters, strength = bind_filters(schema, [root], specs, tail)
            if filters:
                # A filter on the projected label itself is an attribute comparison.
                output = {c.name.lower() for c in selected.selects}
                if not any(f.field in output for f in filters):
                    disjunction = disjunction_for(filters, lists, tail)
                    if disjunction is not None:
                        plan = Plan(root, (), filters, True, root_score + strength + 10,
                                    parent, False, disjunction)
                        yield plan, spellings, "label"
        for target in schema.tables:
            if target == root:
                continue
            for path in sorted(set(map(tuple, schema.path_options([root], target, 2)))):
                related = [edge[2] for edge in path]
                role_words = set().union(*(terms(t) for t in related)) - words
                edge_words = set().union(*(terms(edge[1]) | terms(edge[3]) for edge in path))
                direct = focus & (role_words | edge_words)
                lexical = matching_words(focus, (role_words | edge_words) - SCOPE_NOISE)
                if not lexical:
                    continue
                filters, strength = bind_filters(schema, related, specs, tail)
                if filters is None or filters and not any(f.table == target for f in filters):
                    continue
                if not filters and not matching_words(focus, terms(target) - words - SCOPE_NOISE):
                    continue
                disjunction = disjunction_for(filters, lists, tail)
                if disjunction is None:
                    continue
                observed = any(edge in inferred for edge in path)
                priority = root_score + 4 * len(lexical) + 2 * len(direct) + strength - 3 * len(path) - 4 * observed
                plan = Plan(root, path, filters, grouped, priority, parent, observed, disjunction)
                yield plan, spellings, "relation"


def case_variants(plan, spellings, question):
    """Keep a small alternative when drafts disagree in their spelling of a value."""
    yield plan
    for i, predicate in enumerate(plan.filters):
        value = predicate.value
        if predicate.numeric or re.search(r"""["']""" + re.escape(value) + r"""["']""", question, re.I):
            continue
        alternatives = spellings.get(value.casefold(), set()) - {value}
        for alternate in sorted(alternatives)[:1]:
            filters = list(plan.filters)
            filters[i] = replace(predicate, value=alternate)
            yield replace(plan, filters=tuple(filters), priority=plan.priority - 1)


def pool(obs, schema, fallback, mode, match):
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e} | {identity(fallback)}
    items = []
    for original, spellings, source in ordinal_plans(obs, schema, fallback, match):
        for plan in case_variants(original, spellings, obs["task"]["question"]):
            query = compile_plan(plan, schema, obs["example"]["columns"], mode)
            if query is None:
                continue
            sql = sql_text(query)
            key = identity(sql)
            if key in seen:
                continue
            try:
                validate_sql(sql)
            except ValueError:
                continue
            seen.add(key)
            items.append({"sql": sql, "mode": mode, "source": source, "root": plan.root,
                          "path": plan.path, "label_scope": plan.label_scope, "priority": plan.priority,
                          "parent": plan.parent, "observed_key_edge": plan.inferred_edge,
                          "filters": [[f.table, f.field, f.operator.key, f.value] for f in plan.filters],
                          "filter_disjunction": plan.disjunction})
            if len(items) == 24:
                return items
    return items
