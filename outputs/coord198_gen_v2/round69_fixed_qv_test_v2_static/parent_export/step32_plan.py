"""Relational slots shared by the CPU and frozen-model plan sources."""
from dataclasses import dataclass
import re

from sqlglot import exp
from sqlglot.errors import SqlglotError

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import error_message, identifier_edits, parse, sql_text
from step4_agent import identity
from step4_edits import Schema, join_repairs

SLOTS = ("output", "relations", "where", "grain", "having", "order", "limit")
CLAUSES = ("SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT")
AST_KEYS = ("expressions", "from", "where", "group", "having", "order", "limit")
FREQUENCY = re.compile(r"\b(?:most|fewest|greatest|largest|highest|least|lowest)\b.{0,30}\b(?:number|count|times|frequently|common)\b|\b(?:most|fewest)\s+\w+s\b", re.I)
COUNT_REQUEST = re.compile(r"\b(?:number of|how many|count|times)\b", re.I)
COMPARISON = re.compile(r"\b(?:any|every|all|average|before|after|both)\b", re.I)
RANK = re.compile(r"\b(?:most|least|fewest|greatest|largest|highest|lowest|top|recently)\b", re.I)


@dataclass(frozen=True)
class Plan:
    output: str
    relations: str
    where: str = ""
    grain: str = ""
    having: str = ""
    order: str = ""
    limit: str = ""
    multiplicity: str = "rows"


def fragment(node, prefix):
    return sql_text(node)[len(prefix) + 1:] if node else ""


def decompose(tree):
    if not isinstance(tree, exp.Select) or any(value for key, value in tree.args.items()
                                              if key not in (*AST_KEYS, "joins", "distinct")):
        return None
    relations = " ".join([fragment(tree.args.get("from"), "FROM"),
                          *(sql_text(j) for j in tree.args.get("joins", []))]).strip()
    if not relations or (tree.args.get("distinct") and tree.args["distinct"].expressions):
        return None
    grain = tree.args.get("group")
    return Plan(", ".join(map(sql_text, tree.expressions)), relations,
                *(fragment(tree.args.get(k), c) for k, c in zip(AST_KEYS[2:], CLAUSES[2:])),
                multiplicity="distinct" if tree.args.get("distinct") else "groups" if grain else "rows")


def tree_plan(tree, basis):
    plan = decompose(tree)
    if plan:
        return {"plan": vars(plan), "basis": basis}


def compile_plan(record, width):
    try:
        p = Plan(**record)
        if not all(isinstance(getattr(p, key), str) for key in (*SLOTS, "multiplicity")):
            return
        if not p.output or not p.relations or p.multiplicity not in ("rows", "distinct", "groups"):
            return
        prefix = "SELECT DISTINCT" if p.multiplicity == "distinct" else "SELECT"
        fragments = [" ".join((prefix if k == "output" else c, getattr(p, k)))
                     for k, c in zip(SLOTS, CLAUSES) if getattr(p, k)]
        tree = validate_sql(" ".join(fragments))
        normalized = decompose(tree)
        if not normalized or len(tree.expressions) != width:
            return
        # A fragment cannot smuggle another top-level clause or change multiplicity.
        if normalized.multiplicity != p.multiplicity or any(
                bool(getattr(normalized, key)) != bool(getattr(p, key)) for key in SLOTS):
            return
        # Isolate each fragment with the minimum scaffold. Reject cross-slot injection.
        for i, (key, clause) in enumerate(zip(SLOTS, CLAUSES)):
            value = getattr(p, key)
            if not value:
                continue
            scaffold = (f"{prefix} {value} FROM __plan" if i == 0 else
                        f"SELECT 1 FROM {value}" if i == 1 else
                        f"SELECT 1 FROM __plan {clause} {value}")
            single = decompose(validate_sql(scaffold))
            if not single or getattr(single, key) != getattr(normalized, key):
                return
            expected = {"output", "relations", key}
            if any(getattr(single, other) for other in SLOTS if other not in expected):
                return
        return sql_text(tree)
    except (TypeError, ValueError, SqlglotError, RecursionError):
        return


def context(obs):
    history = obs["history"]
    return {
        "question": obs["task"]["question"],
        "columns": obs["example"]["columns"],
        "draft": obs["issue_sql"],
        "checks": [(e["sql"], {k: e["observation"].get(k) for k in
                               ("ok", "matches_example", "row_count", "error")})
                   for e in history if e["tool"] == "check"],
        "hints": list(dict.fromkeys(s for e in history if e["tool"] == "repair_hint"
                                   for s in e["observation"])),
        "schema": next((e["observation"] for e in history if e["tool"] == "schema"), None),
    }


def frequency_shape(c, tree):
    return (tree and FREQUENCY.search(c["question"]) and tree.args.get("group")
            and tree.args.get("order") and sum(1 for _ in tree.find_all(exp.Select)) == 1)


def label_group(tree):
    group = tree.args.get("group")
    if not group or len(group.expressions) != 1 or len(tree.expressions) != 1:
        return
    column = group.expressions[0]
    output = tree.expressions[0].unalias()
    if (isinstance(column, exp.Column) and isinstance(output, exp.Column)
            and column.name.casefold() == output.name.casefold()
            and (not column.table or not output.table or column.table.casefold() == output.table.casefold())
            and re.search(r"(?:^|_)(?:name|title|label|description)$", column.name, re.I)):
        return column


def pending_bindings(c, tree, public):
    if c["schema"] is not None:
        return []
    if public:
        label = label_group(tree) if frequency_shape(c, tree) else None
        if label is not None and not re.search(r"\b(?:by|per|each|same)\s+(?:\w+\s+)?(?:name|title|label)\b",
                                               c["question"], re.I):
            return [{"role": "entity grain key", "field": label.sql()}]
        return []
    if tree:
        for sql, feedback in c["checks"]:
            message = error_message(feedback)
            if message.startswith(("no such column:", "no such table:", "ambiguous column name:")):
                return [{"role": "unresolved draft binding", "field": message.split(":", 1)[1].strip(),
                         "source": sql}]
    return []


def entity_key(c, tree, label):
    schema = Schema(c["schema"])
    sources = schema.sources(tree)
    qualifier = label.table or tree.expressions[0].unalias().table
    choices = [t for t, alias in sources.items() if label.name.lower() in schema.columns[t]
               and (not qualifier or alias.lower() == qualifier.lower())]
    if len(choices) != 1:
        return
    table = choices[0]
    # The grouped label must belong to a single referenced entity key.
    keys = [name for name, col in schema.columns[table].items() if col["pk"]]
    if len(keys) == 1 and any(edge[1] == keys[0] for edge in schema.edges[table]):
        return schema.column(table, keys[0], sources[table])


def multiplicity_diff(parent, candidate):
    def shape(tree):
        if not isinstance(tree, exp.Select):
            return {"shape": sql_text(tree) if tree is not None else None}
        return {key: [sql_text(e) for e in value] if isinstance(value, list) else sql_text(value)
                if value is not None else None for key in ("from", "joins", "distinct", "group", "having", "limit")
                for value in [tree.args.get(key)]} | {
                    "existentials": [sql_text(e) for e in tree.find_all(exp.Exists)],
                    "aggregates": [sql_text(e) for slot in (*tree.expressions, tree.args.get("having"))
                                   if slot is not None for e in slot.find_all(exp.AggFunc)]}
    old, new = shape(parent), shape(parse(candidate))
    return {field: {"before": old.get(field), "after": new.get(field)} for field in old.keys() | new.keys()
            if old.get(field) != new.get(field)}


def compiled(c, tree, candidates):
    seen = set()
    for proposal in candidates:
        if not proposal:
            continue
        sql = compile_plan(proposal["plan"], len(c["columns"]))
        if not sql or identity(sql) in seen or (tree is not None and identity(sql_text(tree)) == identity(sql)):
            continue
        seen.add(identity(sql))
        yield proposal | {"sql": sql, "multiplicity_changes": dict(sorted(multiplicity_diff(tree, sql).items()))}


def cpu_plans(c, tree, public):
    if frequency_shape(c, tree):
        label = label_group(tree)
        key = entity_key(c, tree, label) if label is not None and c["schema"] is not None else None
        order = tree.args["order"]
        ranking = order.expressions[0]
        # Aggregated metrics may represent stored totals. A bare sort field
        # in a grouped frequency query has no such aggregation semantics.
        frequency = (isinstance(ranking.this, exp.Column) and bool(tree.args.get("joins"))
                     and not any(e.alias and e.alias.casefold() == ranking.this.name.casefold()
                                 for e in tree.expressions))
        replacements = []
        if key is not None and not re.search(r"\b(?:by|per|each|same)\s+(?:\w+\s+)?(?:name|title|label)\b",
                                             c["question"], re.I):
            replacements.append(("entity key for frequency", tree.copy().group_by(key, append=False)))
        if frequency:
            proposal = tree.copy()
            proposal.args["order"].expressions[0].set("this", exp.Count(this=exp.Star()))
            replacements.extend([("frequency over existing group", proposal)])
            if key is not None and replacements[0][0] == "entity key for frequency":
                replacements.insert(0, ("entity key and frequency", proposal.copy().group_by(key, append=False)))
        for basis, proposal in replacements:
            yield tree_plan(proposal, basis)
    if public:
        return
    if c["schema"] is not None:
        schema = Schema(c["schema"])
        for sql, feedback in c["checks"]:
            draft = parse(sql)
            if isinstance(draft, exp.Select):
                for _, repaired, evidence in join_repairs(draft, schema, feedback):
                    yield tree_plan(parse(repaired), {"declared_path": evidence})
            for role, repaired in identifier_edits(sql, feedback, c["schema"]):
                yield tree_plan(parse(repaired), {"identifier_binding": role, "error": error_message(feedback)})
    # Shared hints supply alternative relations and scopes. Also retain the draft's
    # grouping, predicate and ordering when excess projection is the only edit.
    for sql in [*(s for s, feedback in c["checks"] if feedback["ok"]), *c["hints"]]:
        hint = parse(sql)
        if not isinstance(hint, exp.Select):
            continue
        width = len(c["columns"])
        if len(hint.expressions) > width:
            headings = [re.sub(r"\W|_", "", col.lower()) for col in c["columns"]]
            options = {re.sub(r"\W|_", "", e.alias_or_name.lower()): e for e in hint.expressions}
            if len(set(headings)) == width and all(h in options for h in headings):
                yield tree_plan(hint.copy().select(*(options[h].copy() for h in headings), append=False),
                                "projection matched public headings")
        else:
            yield tree_plan(hint, "visible relation and predicate proposal")


