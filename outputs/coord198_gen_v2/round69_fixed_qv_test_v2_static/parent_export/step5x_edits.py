"""Schema-grounded candidates for the follow-up to step 5."""
import re

from sqlglot import exp

from repair_rules import COUNT_RANK, sql_text
from step4_edits import join_repairs


def physical_table(schema, name, alias):
    return exp.Table(this=exp.to_identifier(schema.tables[name]["name"], quoted=True),
                     alias=exp.TableAlias(this=exp.to_identifier(alias, quoted=True)))


def foreign_key_edits(tree, schema, feedback, question, arity):
    if len(tree.selects) != arity:
        return
    yield from join_repairs(tree, schema, feedback)
    sources = schema.sources(tree)
    base = tree.args.get("from")
    if not sources or not base or not isinstance(base.this, exp.Table):
        return
    connected = {base.this.name.lower(): base.this.alias_or_name}
    for index, join in enumerate(tree.args.get("joins", [])):
        right = join.this.name.lower()
        on = join.args.get("on")
        # Preserve outer joins, composite predicates and explicit cross joins.
        if (not join.side and join.kind in ("", "INNER") and isinstance(on, exp.EQ)
                and all(isinstance(side, exp.Column) for side in (on.this, on.expression))):
            left_aliases = {alias.lower(): name for name, alias in connected.items()}
            sides = (on.this, on.expression)
            if sides[0].table.lower() == sources[right].lower():
                sides = sides[::-1]
            left = left_aliases.get(sides[0].table.lower())
            if left and sides[1].table.lower() == sources[right].lower():
                paths = set(tuple(path) for path in schema.path_options(connected, right, 2)
                            if all(edge[2] not in sources for edge in path[:-1]))
                direct = {path for path in paths if len(path) == 1}
                already_key = any(path[0] == (
                    left, sides[0].name.lower(), right, sides[1].name.lower()) for path in direct)
                preferred = direct or {path for path in paths if path[0][0] == left}
                if not already_key:
                    for path in sorted(preferred)[:2]:
                        result = tree.copy()
                        aliases = dict(connected)
                        used = {alias.lower() for alias in sources.values()}
                        additions = []
                        for origin, a, target, b in path:
                            alias = sources.get(target, "fk_bridge")
                            while target != right and alias.lower() in used:
                                alias += "_"
                            used.add(alias.lower())
                            key = exp.EQ(this=schema.column(origin, a, aliases[origin]),
                                         expression=schema.column(target, b, alias))
                            if target == right:
                                last = result.args["joins"][index].copy()
                                last.set("on", key)
                                additions.append(last)
                            else:
                                additions.append(exp.Join(this=physical_table(schema, target, alias), on=key))
                            aliases[target] = alias
                        joins = result.args["joins"]
                        result.set("joins", [*joins[:index], *additions, *joins[index + 1:]])
                        yield "join/path", sql_text(result), {"from": on.sql(), "foreign_keys": path}
        connected[right] = sources[right]


def aggregate_edits(tree, schema, feedback, question, arity):
    group = tree.args.get("group")
    order = tree.args.get("order")
    having = tree.args.get("having")
    projections = tree.selects
    question = question.lower()
    rank = COUNT_RANK.search(question)

    if (rank and order and len(projections) in (1, 2) and isinstance(projections[0], exp.Column)
            and not having and not tree.args.get("distinct") and arity in (1, 2)):
        entity = projections[0]
        if group and group.expressions != [entity]:
            return
        count = exp.Count(this=exp.Star())
        result = tree.copy()
        result.set("group", exp.Group(expressions=[entity.copy()]))
        desc = rank[0].split()[0] not in {"ascending", "fewest", "lowest", "smallest"}
        result.set("order", exp.Order(expressions=[
            exp.Ordered(this=count.copy(), desc=desc, nulls_first=not desc)]))
        result.set("expressions", [entity.copy()] + ([count] if arity == 2 else []))
        yield "select/group/order", sql_text(result), {"phrase": rank[0]}

    if group and having and len(projections) > arity:
        labels = [item for item in projections if not item.find(exp.AggFunc)]
        if len(labels) == arity and all(item in group.expressions for item in labels):
            result = tree.copy()
            result.set("expressions", [item.copy() for item in labels])
            # Preserve ordering aliases if used after the projection is removed.
            aliases = {item.alias.lower(): item.this for item in projections if item.alias}
            if order:
                for item in result.args["order"].expressions:
                    column = item.this
                    if isinstance(column, exp.Column) and not column.table and column.name.lower() in aliases:
                        column.replace(aliases[column.name.lower()].copy())
            yield "select/having", sql_text(result), {"having": having.sql()}

    extreme = re.search(r"\b(maximum|minimum|longest|shortest|highest|lowest)\b", question)
    if (extreme and arity == len(projections) == 2 and schema.sources(tree)
            and not having and not tree.args.get("distinct")):
        function = exp.Max if extreme[0] in {"maximum", "longest", "highest"} else exp.Min
        values = [p.this if isinstance(p, exp.Alias) else p for p in projections]
        measures = [p for p in values if isinstance(p, function) and isinstance(p.this, exp.Column)]
        labels = [p for p in values if isinstance(p, exp.Column)]
        if len(measures) == len(labels) == 1:
            measure, label = measures[0], labels[0]
            if group and not tree.args.get("limit"):
                return
            label_term = question.find("name")
            measure_first = label_term < 0 or extreme.start() < label_term
            result = tree.copy()
            # Select a row at the requested extremum without bare aggregate columns.
            result.set("group", None)
            result.set("expressions", [p.copy() for p in (
                [measure.this, label] if measure_first else [label, measure.this])])
            result.set("order", exp.Order(expressions=[exp.Ordered(this=measure.this.copy(),
                                                                  desc=function is exp.Max)]))
            result.limit(1, copy=False)
            yield "select/extremum", sql_text(result), {"phrase": extreme[0], "measure": measure.sql()}
