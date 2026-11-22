"""Compile public-pass repairs with explicit evidence about relation grain."""
from dataclasses import dataclass
import re

from sqlglot import exp

from repair_rules import parse
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step8_groups import alias_free, output_bindings
from step10_bindings import existing, field_words
from step19_grounding import package
from step26_relations import Filter, membership, edge_is_forward

GROUP = re.compile(r"\b(?:each|per|group(?:ed)? by|most|fewest|highest|lowest|"
                   r"greatest|largest|least|top|maximum|minimum|more than|at least|at most|fewer than)\b", re.I)
FREQUENCY = re.compile(r"\b(?:most|fewest|greatest|largest|highest|lowest|smallest|least)\b"
                       r"[^?.;]{0,42}\b(?:number|count|times|common|frequen\w*)\b|"
                       r"\b(?:most|fewest)\b[^?.;]{0,18}$", re.I)
DISTINCT = re.compile(r"\b(?:distinct|unique|different|kinds? of|types? of)\b", re.I)
MEMBER = re.compile(r"\b(?:who|which|that)\s+(?:have|has|had|made|taken|took|received|attended|placed)\b", re.I)
NUMBER = re.compile(r"\b(?:at least|at most|more than|fewer than|exactly|under|over)\s+(?:\d+|one|two|three)\b", re.I)


@dataclass(frozen=True)
class Grain:
    entity: str
    key: tuple
    labels: tuple
    event: str | None = None
    event_key: tuple = ()
    role: tuple = ()

    def evidence(self, conflict):
        return {"entity": self.entity, "entity_key": self.key, "output_labels": self.labels,
                "event": self.event, "event_key": self.event_key, "relation_role": self.role,
                "resolved_conflict": conflict}


def words(name):
    return {"order" if w == "ordered" else w for w in field_words(name)}


def primary(schema, table):
    return tuple(k for k, meta in schema.columns[table].items() if meta["pk"])


def single_scope(tree):
    return (isinstance(tree, exp.Select) and len(list(tree.find_all(exp.Select))) == 1
            and not any(tree.args.get(k) for k in ("with", "offset", "qualify", "windows"))
            and not tree.find(exp.Window))


def schema_signal(tree, question, headings):
    if not single_scope(tree):
        return False
    joins = tree.args.get("joins") or []
    projections = [p.unalias() for p in tree.selects]
    group = tree.args.get("group")
    if group:
        if any("distinct" in h.lower() for h in headings) and DISTINCT.search(question) and not GROUP.search(question):
            return True
        if not GROUP.search(question):
            return False
        order = tree.args.get("order")
        if (order and len(order.expressions) == 1 and tree.args.get("limit")
                and not (tree.args.get("where") or tree.args.get("having"))
                and (FREQUENCY.search(question) or re.search(r"\b(?:most|fewest)\s+\w+", question, re.I))
                and (not isinstance(order.expressions[0].this, exp.Count) or DISTINCT.search(question) or len(joins) > 1)):
            return True
        labels = {p.name.lower() for p in projections if isinstance(p, exp.Column)}
        return bool(joins and labels and all(isinstance(c, exp.Column) and c.name.lower() in labels
                                            for c in group.expressions))
    match = MEMBER.search(question)
    if not match or NUMBER.search(question) or GROUP.search(question) or len(joins) != 1:
        return False
    if any(tree.args.get(k) for k in ("having", "limit", "order", "distinct")):
        return False
    if not all(isinstance(p, (exp.Column, exp.AggFunc)) for p in projections):
        return False
    tables = [t.name for t in tree.find_all(exp.Table)]
    subject, predicate = words(question[:match.start()]), words(question[match.end():])
    return len(tables) == 2 and any(words(a) <= subject and words(b) <= predicate
                                    for a, b in (tables, tables[::-1]))


def labels_of(tree, sources, schema):
    labels = []
    for p in tree.selects:
        node = p.unalias()
        if isinstance(node, exp.AggFunc):
            continue
        if not isinstance(node, exp.Column):
            return ()
        bound = existing(node, sources, schema)
        if bound is None:
            return ()
        labels.append(bound)
    return tuple(labels)


def join_edges(tree, sources, schema):
    result = []
    for join in tree.args.get("joins") or []:
        on = join.args.get("on")
        if join.side or join.kind not in ("", "INNER") or not isinstance(on, exp.EQ):
            return ()
        if not all(isinstance(c, exp.Column) for c in (on.this, on.expression)):
            return ()
        left = existing(on.this, sources, schema)
        right = existing(on.expression, sources, schema)
        if not left or not right:
            return ()
        _, a, k = left
        _, b, ref = right
        if (a, k, b, ref) not in schema.edges[a]:
            return ()
        result.append((a, k, b, ref))
    return tuple(result)


def group_candidate(tree, schema, question):
    group = tree.args.get("group")
    if group is None or not GROUP.search(question) or tree.args.get("distinct"):
        return
    clean = alias_free(tree)
    sources = top_tables(clean, schema)
    output = labels_of(clean, sources, schema)
    if not output or len({(alias, table) for alias, table, _ in output}) != 1:
        return
    alias, entity, _ = output[0]
    entity_key = primary(schema, entity)
    subject = re.split(r"\b(?:who|whose|which has|that|with|having)\b", question, maxsplit=1, flags=re.I)[0]
    subject_words = words(subject)
    if len(entity_key) != 1 or not words(entity) <= subject_words:
        return
    edges = join_edges(clean, sources, schema)
    if len(sources) < 2 or len(edges) != len(clean.args["joins"]):
        return
    # A category value is a grouping key in its own right. A surrogate ID
    # on the label relation does not establish the requested entity grain.
    if subject_words & {"type", "category", "class", "kind"}:
        return
    if not any(fk["table"].lower() == entity and fk["to"].lower() == entity_key[0]
               for t in sources.values() if t != entity for fk in schema.tables[t]["foreign_keys"]):
        return
    if DISTINCT.search(question) and not any(isinstance(c.this, exp.Distinct) for c in clean.find_all(exp.Count)):
        return
    # Grouping by a name is intentional when the question requests the name as the key.
    named_grain = re.search(r"\b(?:for\s+each|per|group(?:ed)? by)\s+([^?,.;]+)", question, re.I)
    if named_grain and re.search(r"\b(?:name|label|title|category|type)\b", named_grain[1], re.I):
        return
    bound_group = [existing(c, sources, schema) if isinstance(c, exp.Column) else None
                   for c in clean.args["group"].expressions]
    if not all(b and b[:2] == (alias, entity) for b in bound_group):
        return
    fields = tuple(b[2] for b in bound_group)
    if set(entity_key) <= set(fields):
        return
    if not set(fields) <= {f for _, _, f in output}:
        return
    if not (clean.args.get("having") or clean.find(exp.AggFunc)):
        return
    query = clean.group_by(*(schema.column(entity, k, alias) for k in entity_key), append=False)
    plan = Grain(entity, entity_key, tuple(f for _, _, f in output))
    yield query, 120, "entity_group_key", plan.evidence({"old_group": fields})


def named_event(schema, entity, question, sources):
    """Locate an event by its name and the existing relationship evidence."""
    asked = words(question)
    entity_words = words(entity)
    distinct = re.search(r"\b(?:kinds?|types?|number) of (\w+)\b", question, re.I)
    exact_object = words(distinct[1]) if distinct else set()
    matched = []
    for event in sorted(schema.tables):
        if event == entity:
            continue
        event_words = words(event)
        paths = sorted({tuple(p) for p in schema.path_options([entity], event, 2)})
        for path in paths:
            if not path or edge_is_forward(schema, path[0]):
                continue
            explicit = event_words <= asked or (event in sources and (event_words - entity_words) & asked)
            if not explicit:
                continue
            score = (4 * len(event_words & asked) - 3 * len(event_words & entity_words)
                     - 3 * len(event_words - asked) - len(path) + (event in sources))
            score += 5 * bool(exact_object and event_words == exact_object)
            if score > 0:
                matched.append((score, event, path))
    matched.sort(reverse=True)
    if matched and (len(matched) == 1 or matched[0][0] > matched[1][0]):
        return matched[0][1:]
    return None


def event_ranking(tree, schema, question, headings):
    order = tree.args.get("order")
    if (not tree.args.get("group") or not order or len(order.expressions) != 1
            or not tree.args.get("limit") or any(tree.args.get(k) for k in ("distinct", "where", "having"))):
        return
    explicit_frequency = FREQUENCY.search(question) or re.search(r"\b(?:most|fewest)\s+\w+", question, re.I)
    if not explicit_frequency:
        return
    clean = alias_free(tree)
    sources = top_tables(clean, schema)
    labels = labels_of(clean, sources, schema)
    if not labels or len({(a, t) for a, t, _ in labels}) != 1:
        return
    alias, entity, _ = labels[0]
    key = primary(schema, entity)
    if not key or len(words(entity) & words(question)) < min(2, len(words(entity))):
        return
    existing_groups = clean.args["group"].expressions
    if not all(isinstance(c, exp.Column) and (b := existing(c, sources, schema)) and b[:2] == (alias, entity)
               for c in existing_groups):
        return
    relationship = named_event(schema, entity, question, set(sources.values()))
    if relationship is None:
        return
    event, path = relationship
    event_pk = primary(schema, event)
    old_order = clean.args["order"].expressions[0]
    metric = old_order.this
    dropped = set(sources.values()) - {entity, *(edge[2] for edge in path)}
    if any(words(t) <= words(question) for t in dropped):
        return
    counts = list(clean.find_all(exp.Count))
    if any(isinstance(c.this, exp.Distinct) for c in counts) and not DISTINCT.search(question):
        return
    if isinstance(metric, exp.Count) and not dropped:
        if not DISTINCT.search(question) or isinstance(metric.this, exp.Distinct) or len(event_pk) != 1:
            return
    if not isinstance(metric, (exp.Column, exp.Count)):
        return
    if any(not isinstance(p.unalias(), (exp.Column, exp.Count)) for p in clean.selects):
        return
    query = exp.select().from_(physical_table(schema, entity, "r0"))
    aliases = {entity: "r0"}
    join_path(query, path, schema, aliases)
    if DISTINCT.search(question):
        if len(event_pk) != 1:
            return
        measure = exp.Count(this=exp.Distinct(expressions=[schema.column(event, event_pk[0], aliases[event])]))
    elif len(path) == 1 or edge_is_forward(schema, path[-1]):
        measure = exp.Count(this=exp.Star())
    else:
        return
    outputs = [measure.copy() if isinstance(p.unalias(), exp.Count) else
               schema.column(entity, existing(p.unalias(), sources, schema)[2], "r0")
               for p in clean.selects]
    query.select(*outputs, append=False, copy=False)
    query.group_by(*(schema.column(entity, k, "r0") for k in key), copy=False)
    asc = bool(re.search(r"\b(?:fewest|least|smallest|lowest)\b", question, re.I))
    query.order_by(exp.Ordered(this=measure, desc=not asc, nulls_first=asc), copy=False)
    query.set("limit", tree.args["limit"].copy())
    plan = Grain(entity, key, tuple(f for _, _, f in labels), event, event_pk, path)
    yield query, 150, "event_frequency", plan.evidence({"old_metric": metric.sql(), "dropped_tables": sorted(dropped)})


def member_candidate(tree, schema, question, headings):
    match = MEMBER.search(question)
    if (not match or NUMBER.search(question) or GROUP.search(question) or
            any(tree.args.get(k) for k in ("group", "having", "limit", "order", "distinct"))):
        return
    sources = top_tables(tree, schema)
    edges = join_edges(tree, sources, schema)
    if len(edges) != 1:
        return
    a, k, b, ref = edges[0]
    before, after = words(question[:match.start()]), words(question[match.end():])
    directions = [(a, k, b, ref), (b, ref, a, k)]
    for entity, entity_key, event, foreign in directions:
        path = ((entity, entity_key, event, foreign),)
        if (not words(entity) <= before or not words(event) <= after
                or primary(schema, entity) != (entity_key,) or edge_is_forward(schema, path[0])):
            continue
        bound = output_bindings(tree, schema, headings)
        if not bound or any(not (b := existing(c, sources, schema)) or b[1] != entity
                            for p in bound for c in p.find_all(exp.Column)):
            continue
        if any(isinstance(p, exp.Count) and isinstance(p.this, exp.Star) for p in bound):
            continue
        aliases = {entity: "r0"}
        query = exp.select().from_(physical_table(schema, entity, "r0"))
        query.select(*(p.transform(lambda n: schema.column(entity, n.name.lower(), "r0")
                                  if isinstance(n, exp.Column) else n) for p in bound), append=False, copy=False)
        where = tree.args.get("where")
        if where:
            pred = where.this
            if not isinstance(pred, exp.EQ) or not isinstance(pred.this, exp.Column) or not isinstance(pred.expression, exp.Literal):
                continue
            binding = existing(pred.this, sources, schema)
            if not binding:
                continue
            atom = Filter(binding[1], binding[2], pred)
            if atom.table == entity:
                query.where(atom.bind(schema, aliases), copy=False)
                atom = None
        else:
            atom = None
        membership(query, path, schema, aliases, atom)
        plan = Grain(entity, (entity_key,), tuple(headings), event, primary(schema, event), path)
        yield query, 140, "membership_grain", plan.evidence("existence wording and reverse FK multiply entity rows")


def distinct_total(tree, schema, question, headings):
    if (len(headings) != 1 or not re.fullmatch(r"count\(\s*distinct\s+[\w.]+\s*\)", headings[0], re.I)
            or not DISTINCT.search(question) or GROUP.search(question)
            or not tree.args.get("group") or tree.args.get("having") or tree.args.get("limit")):
        return
    bound = output_bindings(tree, schema, headings)
    if not bound or len(tree.selects) != 1 or not tree.find(exp.Count):
        return
    query = tree.copy().select(*bound, append=False)
    query.set("group", None)
    query.set("order", None)
    yield query, 160, "distinct_total", {"output_labels": headings, "event_key": headings[0],
                                         "resolved_conflict": "one distinct aggregate requested across all events"}


def candidate_pool(obs, fallback, tables):
    tree = parse(fallback)
    question = obs["task"]["question"]
    heads = obs["example"]["columns"]
    if not schema_signal(tree, question, heads):
        return []
    schema = Schema(tables)
    raw = [*distinct_total(tree, schema, question, heads),
           *event_ranking(tree, schema, question, heads),
           *group_candidate(tree, schema, question),
           *member_candidate(tree, schema, question, heads)]
    return package({"history": []}, fallback, raw)
