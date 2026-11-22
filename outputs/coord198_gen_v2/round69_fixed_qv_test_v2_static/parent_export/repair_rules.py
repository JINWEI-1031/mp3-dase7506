"""Bounded SQLite edits based on visible schema, AST and tool feedback."""
from collections import Counter
from difflib import SequenceMatcher
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.tokens import Tokenizer


MAX_EDITS_PER_CHECK = 3
MAX_RULE_CANDIDATES = 5
MAX_SQL_BYTES = 16_000


def parse(sql):
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (SqlglotError, RecursionError):
        return None
    if len(statements) == 1 and isinstance(statements[0], (exp.Select, exp.SetOperation)):
        return statements[0]
    return None


def sql_text(tree):
    return tree.sql(dialect="sqlite")


def error_message(feedback):
    return (feedback.get("error") or {}).get("message", "")


def identifier_error(feedback):
    message = error_message(feedback)
    return message.startswith(("no such column: ", "no such table: ",
                               "ambiguous column name: ", "SQL parse error: "))


def normalized(name):
    return name.lower().replace("_", "")


def nearby(name, options):
    """Return one unambiguous spelling match; keep exact and underscore matches first."""
    simplified = normalized(name)
    exact = [option for option in options if normalized(option) == simplified]
    if exact:
        return exact if len(exact) == 1 else []
    ranked = sorted(
        ((SequenceMatcher(None, simplified, normalized(option)).ratio(), option)
         for option in options), reverse=True
    )
    if not ranked or len(simplified) < 4 or ranked[0][0] < .80:
        return []
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < .10:
        return []
    return [ranked[0][1]]


def syntax_edit(sql, feedback):
    """Patch a single unquoted keyword typo only at the parser's reported token."""
    if not error_message(feedback).startswith("SQL parse error: "):
        return []
    try:
        sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as error:
        detail = error.errors[0] if error.errors else {}
        try:
            tokens = Tokenizer(dialect="sqlite").tokenize(sql)
        except (SqlglotError, RecursionError):
            return []
        typos = {"selct": "SELECT", "selec": "SELECT", "form": "FROM",
                 "gruop": "GROUP", "oder": "ORDER", "wher": "WHERE"}
        for index, token in enumerate(tokens):
            name = token.text.lower()
            reported = (token.line, token.col) == (detail.get("line"), detail.get("col"))
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            before_by = (name in {"gruop", "oder"} and next_token is not None
                         and next_token.text.lower() == "by"
                         and (next_token.line, next_token.col) ==
                         (detail.get("line"), detail.get("col")))
            if (not (reported or before_by) or name not in typos
                    or sql[token.start:token.end + 1].lower() != name):
                continue
            if name in {"selct", "selec"} and index != 0:
                continue
            replacement = sql[:token.start] + typos[name] + sql[token.end + 1:]
            if parse(replacement) is not None:
                return [("syntax", replacement)]
    except (SqlglotError, RecursionError):
        return []
    return []


def source_columns(scope, tables):
    sources = {}
    for alias, (_, source) in scope.selected_sources.items():
        if isinstance(source, exp.Table) and not source.db and not source.catalog:
            sources[alias] = tables.get(source.name.lower())
        elif isinstance(source, Scope):
            output = source.outer_columns or source.expression.named_selects
            names = Counter(name.lower() for name in output)
            sources[alias] = (dict(zip((name.lower() for name in output), output))
                              if output and "*" not in output
                              and max(names.values()) == 1 else None)
        else:
            sources[alias] = None
    return sources


def column_targets(column, scope, tables, ambiguous):
    sources = source_columns(scope, tables)
    # Invalid/unknown sources and correlated references can hide other bindings.
    if not sources or any(names is None for names in sources.values()):
        return []
    name, qualifier = column.name.lower(), column.table.lower()
    if not qualifier and isinstance(scope.expression, exp.Select):
        if name in {projection.alias.lower() for projection in scope.expression.selects}:
            return []
    owner = next((alias for alias in sources if alias.lower() == qualifier), None)
    exact = [(alias, names[name]) for alias, names in sources.items() if name in names]
    if ambiguous:
        return exact if not qualifier and len(exact) >= 2 else []
    if owner is None and scope.can_be_correlated:
        parent = scope.parent
        while parent is not None:
            outer = source_columns(parent, tables)
            if any(alias.lower() == qualifier if qualifier else
                   names is None or name in names for alias, names in outer.items()):
                return []
            parent = parent.parent
    if owner and name in sources[owner]:
        return []
    if len(exact) == 1:
        return exact
    if exact:
        return []
    if owner:
        return [(owner, match) for match in nearby(name, list(sources[owner].values()))]
    if not qualifier and len(sources) == 1:
        alias, names = next(iter(sources.items()))
        return [(alias, match) for match in nearby(name, list(names.values()))]
    return []


def identifier_edits(sql, feedback, schema):
    if not identifier_error(feedback):
        return []
    tree = parse(sql)
    if tree is None:
        return syntax_edit(sql, feedback)
    message = error_message(feedback)
    kind, _, token = message.partition(": ")
    token = token.strip().lower()
    table_names = {table["name"].lower(): table["name"] for table in schema}
    tables = {table["name"].lower(): {column["name"].lower(): column["name"]
                                    for column in table["columns"]} for table in schema}
    edits = []

    def record(node, changes, label):
        before = {key: node.args.get(key) for key in changes}
        for key, value in changes.items():
            node.set(key, value)
        edits.append((label, sql_text(tree)))
        for key, value in before.items():
            node.set(key, value)

    try:
        scopes = traverse_scope(tree)
        for scope in scopes:
            if kind == "no such table" and token not in tables:
                for _, source in scope.selected_sources.values():
                    if (isinstance(source, exp.Table) and source.name.lower() == token
                            and not source.db and not source.catalog):
                        for name in nearby(token, list(table_names.values())):
                            changes = {"this": exp.to_identifier(name, quoted=True)}
                            if not source.alias:
                                changes["alias"] = exp.TableAlias(
                                    this=exp.to_identifier(source.name, quoted=True))
                            record(source, changes, "table")
            if kind not in {"no such column", "ambiguous column name"}:
                continue
            for column in scope.find_all(exp.Column):
                if column.is_star or column.db or column.catalog:
                    continue
                key = ".".join(part.name.lower() for part in column.parts)
                if key != token:
                    continue
                for alias, name in column_targets(
                        column, scope, tables, kind == "ambiguous column name"):
                    # Rebinding one side of a join must not turn it into a self-equality.
                    if isinstance(column.parent, exp.EQ):
                        other = column.parent.expression if column.arg_key == "this" else column.parent.this
                        if (isinstance(other, exp.Column) and other.table.lower() == alias.lower()
                                and other.name.lower() == name.lower()):
                            continue
                    record(column, {"this": exp.to_identifier(name, quoted=True),
                                    "table": exp.to_identifier(alias, quoted=True)}, "column")
            if len(edits) >= MAX_EDITS_PER_CHECK:
                break
    except OptimizeError:
        return []
    return list(dict.fromkeys(edits))[:MAX_EDITS_PER_CHECK]


DISTINCT_REQUEST = re.compile(r"\b(?:distinct|unique|different)\s+\w+", re.I)
COUNT_RANK = re.compile(r"\b(?:descending|ascending) order of (?:the )?(?:number|count) of\b"
                        r"|\b(?:most|highest|largest|fewest|lowest|smallest) number of\b", re.I)


def count_rank_edit(tree, question):
    match = COUNT_RANK.search(question)
    order = tree.args.get("order")
    if (match is None or not order or len(order.expressions) != 1 or len(tree.selects) != 1
            or tree.args.get("joins") or tree.args.get("group") or tree.args.get("having")
            or tree.args.get("distinct") or not isinstance(tree.args.get("from"), exp.From)
            or not isinstance(tree.args["from"].this, exp.Table)):
        return None
    ordered = order.expressions[0]
    projection = tree.selects[0]
    if (not isinstance(projection, exp.Column) or not isinstance(ordered.this, exp.Column)
            or ordered.this.name.lower() != projection.name.lower()
            or (ordered.this.table and projection.table
                and ordered.this.table.lower() != projection.table.lower())):
        return None
    desc = match.group().lower().split()[0] not in {"ascending", "fewest", "lowest", "smallest"}
    result = tree.copy()
    result.set("group", exp.Group(expressions=[projection.copy()]))
    result.set("order", exp.Order(expressions=[exp.Ordered(this=exp.Count(this=exp.Star()),
                                                          desc=desc, nulls_first=not desc)]))
    return result


def semantic_edits(sql, question, rules):
    tree = parse(sql)
    if not isinstance(tree, exp.Select):
        return []
    edits = []
    if ("distinct" in rules and DISTINCT_REQUEST.search(question)
            and not tree.args.get("distinct")
            and all(not projection.find(exp.AggFunc) for projection in tree.selects)):
        result = tree.copy()
        result.set("distinct", exp.Distinct())
        edits.append(("distinct", sql_text(result)))
    if "count_rank" in rules:
        result = count_rank_edit(tree, question)
        if result is not None:
            edits.append(("count_rank", sql_text(result)))
    return edits


def intent_conflicts(sql, question):
    tree = parse(sql)
    if not isinstance(tree, exp.Select):
        return None
    conflicts = set()
    if semantic_edits(sql, question, ("distinct",)):
        conflicts.add("distinct")
    order = tree.args.get("order")
    if COUNT_RANK.search(question) and not (
            tree.args.get("group") and order and order.find(exp.Count)):
        conflicts.add("count_rank")
    direction = re.search(r"\b(ascending|descending) order\b", question, re.I)
    if direction and order and len(order.expressions) == 1:
        if bool(order.expressions[0].args.get("desc")) != (direction[1].lower() == "descending"):
            conflicts.add("direction")
    limit_request = re.search(r"\btop ([1-9][0-9]?)\b", question, re.I)
    if limit_request:
        limit = tree.args.get("limit")
        if not order or not limit or limit.expression.name != limit_request[1]:
            conflicts.add("top")
    return conflicts


