"""Ground case-only edits to category columns and observations already in the episode."""
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step7_semantics import terms, top_tables
from step10_bindings import existing

CATEGORY = {"category", "class", "color", "division", "gender", "genre", "grade", "industry",
            "level", "membership", "occupation", "position", "race", "rank", "relationship",
            "role", "sector", "sex", "status", "style", "type"}
LEXEME = re.compile(r"[A-Za-z][A-Za-z -]{0,38}[A-Za-z]|[A-Za-z]")
QUOTED = re.compile(r"""(?<!\w)(?:"([^"]+)"|'([^']+)')""")
CLAUSES = (exp.Where, exp.Having)


def question_spelling(question, value):
    if not LEXEME.fullmatch(value) or any(value.casefold() == (a or b).casefold()
                                          for a, b in QUOTED.findall(question)):
        return None
    match = re.search(r"(?<!\w)(" + re.escape(value) + r")(?:s|es)?(?!\w)", question, re.I | re.ASCII)
    return match[1] if match else None


def category_binding(column, select, schema, question):
    binding = existing(column, top_tables(select, schema), schema)
    if not binding:
        return None
    _, table, field = binding
    type_name = schema.columns[table][field]["type"]
    if not re.match(r"(?:text|(?:n?var)?char|nchar|clob)", type_name, re.I):
        return None
    words = terms(field)
    context = terms(question)
    role = terms(table)
    category = words & CATEGORY or field == "name" and role & CATEGORY & context
    if not category or words & {"id", "date", "time", "number", "description", "detail"}:
        return None
    return table, field


def literals(tree, schema, question):
    for predicate in tree.find_all(exp.EQ, exp.NEQ, exp.In):
        column = predicate.this
        if not isinstance(column, exp.Column) or not predicate.find_ancestor(*CLAUSES):
            continue
        select = predicate.find_ancestor(exp.Select)
        if select is None:
            continue
        binding = category_binding(column, select, schema, question)
        if not binding:
            continue
        values = predicate.expressions if isinstance(predicate, exp.In) else [predicate.expression]
        for literal in values:
            if not isinstance(literal, exp.Literal) or not literal.is_string:
                continue
            spelling = question_spelling(question, literal.this)
            if spelling:
                yield predicate, literal, binding, spelling


def count_result(tree, observation):
    if not observation.get("ok"):
        return None
    if observation["row_count"] == 0:
        return False
    if isinstance(tree, exp.Select) and len(tree.selects) == 1 and isinstance(
            tree.selects[0].unalias(), exp.Count) and observation["rows"]:
        return observation["rows"][0][0] > 0
    # Returned rows from set operations or scalar aggregates are not witnesses.
    if not isinstance(tree, exp.Select) or any(isinstance(p.unalias(), exp.AggFunc) for p in tree.selects):
        return None
    return True


def mandatory_positive(predicate):
    return (isinstance(predicate, exp.EQ)
            and not any(predicate.find_ancestor(node) for node in (exp.Or, exp.Not, exp.SetOperation)))


def requested_projection(feedback, headings):
    if not feedback or len(feedback["columns"]) != len(headings):
        return 0
    if [h.lower() for h in feedback["columns"]] == [h.lower() for h in headings]:
        return 2
    return 1


def feedback_evidence(obs, schema, binding, value):
    confirmed, positive, empty = set(), set(), set()
    question = obs["task"]["question"]
    for event in obs["history"]:
        if event["tool"] != "check" or not event["observation"]["ok"]:
            continue
        observation = event["observation"]
        tree = parse(event["sql"])
        if tree is None:
            continue
        if isinstance(tree, exp.Select):
            for index, selected in enumerate(tree.selects):
                column = selected.unalias()
                if not isinstance(column, exp.Column) or category_binding(column, tree, schema, question) != binding:
                    continue
                confirmed.update(row[index] for row in observation["rows"] if len(row) == len(tree.selects)
                                 and isinstance(row[index], str) and row[index].casefold() == value.casefold())
        support = count_result(tree, observation)
        for predicate, literal, target, _ in literals(tree, schema, question):
            if target != binding or literal.this.casefold() != value.casefold() or not mandatory_positive(predicate):
                continue
            if support is True:
                positive.add(literal.this)
            elif support is False:
                empty.add(literal.this)
    return {"projected": sorted(confirmed), "positive": sorted(positive), "empty": sorted(empty - positive)}


def target_in(tree, schema, question):
    items = list(literals(tree, schema, question))
    distinct = {(binding, literal.this.casefold()) for _, literal, binding, _ in items}
    # Edit exactly one grounded occurrence, including one item of a literal list.
    return items[0] if len(items) == len(distinct) == 1 else None


def targets(obs, schema, fallback, planned=None):
    question = obs["task"]["question"]
    if planned is not None:
        parents = [(planned, 3)]
    else:
        parents = []
        for event in obs["history"]:
            if event["tool"] != "check":
                continue
            feedback = event["observation"]
            if not feedback["ok"]:
                continue
            projection = requested_projection(feedback, obs["example"]["columns"])
            if projection:
                parents.append((event["sql"], projection))
    hits = []
    for parent, projection in parents:
        tree = parse(parent)
        item = target_in(tree, schema, question) if tree is not None else None
        if item is None:
            continue
        predicate, literal, binding, spelling = item
        evidence = feedback_evidence(obs, schema, binding, literal.this)
        alternatives = set(evidence["projected"] + evidence["positive"]) - {literal.this}
        empty = literal.this in evidence["empty"]
        lower_ambiguous = planned is not None and literal.this.islower() and literal.this == spelling
        if not (alternatives or empty or lower_ambiguous):
            continue
        rank = (bool(alternatives), projection, parent == fallback, -len(list(tree.find_all(exp.Join))))
        hits.append((rank, parent, tree, predicate, literal, binding, spelling, evidence))
    return max(hits, key=lambda x: x[0]) if hits else None


def compile_candidates(obs, schema, fallback, planned=None):
    target = targets(obs, schema, fallback, planned)
    if not target:
        return None
    _, parent, tree, predicate, literal, binding, spelling, evidence = target
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e}
    value = literal.this
    options = list(dict.fromkeys([value, *evidence["projected"], *evidence["positive"],
                                  spelling, value.lower(), value.title(), value.upper()]))
    ordinal = next(i for i, node in enumerate(tree.walk()) if node is literal)
    pool = []
    for variant in [*options, None]:
        modified = tree.copy()
        copy = list(modified.walk())[ordinal]
        if variant is None:
            comparator = copy.parent
            # Folding multi-item IN would also change other categories.
            if isinstance(comparator, exp.In) and len(comparator.expressions) != 1:
                continue
            comparator.this.replace(exp.Lower(this=comparator.this.copy()))
            copy.replace(exp.Literal.string(value.lower()))
        else:
            copy.replace(exp.Literal.string(variant))
        sql = planned if variant == value and planned is not None else sql_text(modified)
        try:
            validate_sql(sql)
        except ValueError:
            continue
        pool.append({"sql": sql, "parent": parent, "binding": binding, "value": variant,
                     "kind": "normalized" if variant is None else "original" if variant == value else "spelling",
                     "already_checked": identity(sql) in seen})
    return {"pool": pool, "evidence": evidence, "value": value, "question_spelling": spelling}


def schedule(compiled, policy):
    candidates = compiled["pool"]
    evidence = compiled["evidence"]
    value = compiled["value"]
    original = [c for c in candidates if c["kind"] == "original"]
    if policy == "original":
        return original
    if policy == "normalized":
        return [c for c in candidates if c["kind"] == "normalized"] + original
    variants = [c for c in candidates if c["kind"] != "normalized"]
    convention = value.title() if value.islower() else value.lower()
    return sorted(variants, key=lambda c: (
        c["value"] in evidence["projected"], c["value"] in evidence["positive"],
        c["value"] not in evidence["empty"], c["value"] == value,
        c["value"] == convention, c["value"] == compiled["question_spelling"]), reverse=True)
