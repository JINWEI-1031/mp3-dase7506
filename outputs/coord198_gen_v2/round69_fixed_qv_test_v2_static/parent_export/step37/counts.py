"""Count positions and row grain are independent selection checks."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step32_plan import tree_plan
from step36.frequency import COUNT_HEADER, selection_evidence as frequency_evidence


def count_positions(tree):
    if not isinstance(tree, exp.Select):
        return ()
    return tuple(i for i, item in enumerate(tree.expressions) if isinstance(item.unalias().unnest(), exp.Count))


def requested_counts(headings):
    return tuple(i for i, heading in enumerate(headings) if COUNT_HEADER.match(heading))


def projection_plan(parent, headings):
    expected = requested_counts(headings)
    old = count_positions(parent)
    if (len(expected) != 1 or len(old) != 1 or old == expected
            or len(parent.expressions) != len(headings) or parent.args.get("distinct")):
        return None
    # SQLite lets ORDER BY refer to projection positions. Those must keep
    # referring to the same expression after the reorder.
    if parent.args.get("order") and any(isinstance(e.this, exp.Literal) and e.this.is_int
                                        for e in parent.args["order"].expressions):
        return None
    if parent.args.get("group") and any(isinstance(e, exp.Literal) and e.is_int
                                        for e in parent.args["group"].expressions):
        return None
    fields = [e.copy() for i, e in enumerate(parent.expressions) if i != old[0]]
    labels = [heading.casefold() for i, heading in enumerate(headings) if i != expected[0]]
    if [e.alias_or_name.casefold() for e in fields] != labels:
        return None
    fields.insert(expected[0], parent.expressions[old[0]].copy())
    return tree_plan(parent.copy().select(*fields, append=False),
                     {"kind": "count_projection_order", "old_count_positions": old,
                      "required_count_positions": expected, "eligible": True,
                      "advantage": "AST confirms misplaced count; row source unchanged"})


def grain_audit(parent, sql):
    after = parse(sql)
    if not isinstance(parent, exp.Select) or not isinstance(after, exp.Select):
        return {"status": "unsupported"}
    def source(tree):
        return {k: sql_text(value) if isinstance(value, exp.Expression)
                else [sql_text(v) for v in value] if isinstance(value, list) else value
                for k in ("from", "joins", "where", "group", "having", "distinct")
                for value in [tree.args.get(k)]}
    old, new = source(parent), source(after)
    keys = [k for k in old if old[k] != new[k]]
    return {"changed_slots": keys, "old_join_count": len(parent.args.get("joins", [])),
            "new_join_count": len(after.args.get("joins", [])),
            "old_count_positions": count_positions(parent), "new_count_positions": count_positions(after),
            "old_count_expressions": [sql_text(e.unalias()) for e in parent.expressions if isinstance(e.unalias().unnest(), exp.Count)],
            "new_count_expressions": [sql_text(e.unalias()) for e in after.expressions if isinstance(e.unalias().unnest(), exp.Count)]}


def selection_evidence(parent, c, candidate, public):
    if public and candidate["basis"]["mode"] == "count per label":
        # A display-position mismatch licenses the projection-only plan.
        # It cannot provide evidence for a different row source or measure.
        return None
    return frequency_evidence(parent, c, candidate, public)

