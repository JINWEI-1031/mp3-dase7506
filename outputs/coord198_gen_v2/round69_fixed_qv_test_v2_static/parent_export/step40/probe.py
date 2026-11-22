"""Tiny projection and parser-feedback plans at the first public check."""
from collections import Counter

from sqlglot import exp
from sqlglot.errors import SqlglotError

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import error_message, parse, sql_text, syntax_edit
from step4_agent import identity, passed
from step32_plan import decompose


def projection(tree, headings, question):
    if not isinstance(tree, exp.Select) or any(p.is_star for p in tree.expressions):
        return None
    if any(isinstance(e.this, exp.Literal) and e.this.is_int
           for order in tree.find_all(exp.Ordered) for e in (order,)):
        return None
    if tree.args.get("group") or tree.args.get("distinct"):
        return None
    available = {}
    for field in tree.expressions:
        available.setdefault(field.alias_or_name.casefold(), []).append(field)
    if any(len(available.get(h, [])) != 1 for h in headings) or len(set(headings)) != len(headings):
        return None
    # Removing a projection may alter the ordering of output rows only when
    # positional references were used, which is excluded above.
    if Counter(headings) != Counter(available.keys()) and not all(
        h.replace("_", " ") in question.casefold() for h in headings
    ):
        return None
    return tree.copy().select(*(available[h][0].copy() for h in headings), append=False)


def propose(obs, settings):
    history = obs["history"]
    if len(history) != settings["trigger_after_charged_calls"] or any(
        e["tool"] == "check" and passed(e["observation"]) for e in history
    ):
        return False, []
    first = history[0]
    draft, feedback = obs["issue_sql"], first["observation"]
    headings = [col.casefold() for col in obs["example"]["columns"]]
    options = [(kind, parse(sql)) for kind, sql in syntax_edit(draft, feedback)]
    edited = projection(parse(draft), headings, obs["task"]["question"])
    if edited is not None and feedback["ok"]:
        options.append(("public_projection", edited))
    candidates = {}
    for kind, tree in options:
        if tree is None:
            continue
        sql = sql_text(tree)
        try:
            validate_sql(sql)
        except (ValueError, SqlglotError):
            continue
        plan = decompose(tree)
        if plan and identity(sql) != identity(draft) and len(tree.expressions) == len(headings):
            candidates.setdefault(identity(sql), {"sql": sql, "plan": vars(plan), "kind": kind,
                                                   "error": error_message(feedback), "public_columns": headings})
    return True, list(candidates.values())[:settings["candidate_cap"]]
