from sqlglot.errors import ParseError, TokenError, OptimizeError
from .errors import _ordinary, _rejection_type, RuleRefusal
'Frozen R39 pure rule text. No research scan, paths, controller or I/O.'
import collections, hashlib, re
from sqlglot import exp
from .binding_core import scope, maps_and_evidence, canonical, affinity

def h(s):
    return hashlib.sha256(s.encode()).hexdigest()

def stem(s):
    if s.endswith('ies') and len(s) > 4:
        return s[:-3] + 'y'
    if s.endswith('s') and (not s.endswith(('ss', 'us', 'is'))) and (len(s) > 3):
        return s[:-1]
    return s

def words(s):
    return {stem(w) for w in re.findall('[a-z]+', s.lower())}
AGG_REQUEST = {'count', 'number', 'total', 'sum', 'average', 'avg', 'mean', 'minimum', 'maximum', 'min', 'max', 'amount', 'quantity', 'percentage', 'percent', 'ratio'}
IDENTITY = {'name', 'title', 'label', 'id', 'code'}
GENERIC = {'the', 'a', 'an', 'what', 'which', 'who', 'are', 'is', 'of', 'for', 'all', 'list', 'show', 'give', 'find', 'return', 'get', 'display', 'top', 'bottom', 'first', 'last', 'please', 'and', 'their', 'each'}

def request_head(question):
    head = re.split('\\b(?:with|whose|who|that|where|having|ordered|sorted|ranked)\\b', question, 1, flags=re.I)[0]
    return (head, words(head))

def column_info(c, bindings):
    candidates = [(actual, t, k) for alias, (actual, t) in bindings.items() for k in t['columns'] if k['name'].lower() == c.name.lower() and (not c.table or alias == c.table.lower())]
    if len(candidates) != 1:
        raise RuleRefusal('UNKNOWN_column_owner_not_unique_after_binding')
    return candidates[0]

def same_bound_column(a, b, bindings):
    if not isinstance(a, exp.Column) or not isinstance(b, exp.Column):
        return False
    aa, _, ac = column_info(a, bindings)
    ba, _, bc = column_info(b, bindings)
    return aa.lower() == ba.lower() and ac['name'].lower() == bc['name'].lower()

def projection_contract(tree, bindings, shape, question):
    """Delete only all excess aggregate projections, never an arbitrary column."""
    width = shape['width']
    extra = len(tree.expressions) - width
    if extra <= 0:
        return (None, 'NO_EXCESS_PROJECTION')
    if width < 1 or tree.args.get('distinct') or (not tree.args.get('group')):
        return (None, 'UNKNOWN_projection_distinct_or_implicit_group')
    head, requested = request_head(question)
    if re.search('\\bhow\\s+(?:many|much)\\b', head, re.I) or requested & AGG_REQUEST:
        return (None, 'UNKNOWN_aggregate_requested_in_output_head')
    aggs = []
    keep = []
    for i, p in enumerate(tree.expressions):
        e = p.this if isinstance(p, exp.Alias) else p
        if isinstance(e, exp.AggFunc):
            aggs.append((i, p, e))
        elif isinstance(e, exp.Column) and (not e.is_star):
            keep.append((i, p, e))
        else:
            return (None, 'UNKNOWN_output_not_plain_column_or_direct_aggregate')
    if len(aggs) != extra or len(keep) != width:
        return (None, 'UNKNOWN_cannot_identify_exactly_excess_aggregates')
    if not aggs:
        return (None, 'NO_EXCESS_AGGREGATE')
    group = tree.args['group']
    groupcols = [e for e in group.expressions if isinstance(e, exp.Column)]
    if len(groupcols) != len(group.expressions):
        return (None, 'UNKNOWN_noncolumn_group_key')
    retained = []
    for output_i, (_, p, c) in enumerate(keep):
        actual, table, col = column_info(c, bindings)
        tokens = words(col['name'])
        entity = words(table['name'])
        meaningful = tokens - GENERIC - IDENTITY
        if not (meaningful and meaningful <= requested or (tokens & IDENTITY and (meaningful | entity) & requested - GENERIC - IDENTITY)):
            return (None, 'UNKNOWN_remaining_output_not_lexically_requested')
        if not any((same_bound_column(c, g, bindings) for g in groupcols)):
            same = [g for g in groupcols if g.name.lower() == c.name.lower()]
            equal = False
            for j in tree.find_all(exp.Join):
                if j.args.get('side') or j.args.get('kind') not in [None, '', 'INNER']:
                    continue

                def terms(n):
                    return terms(n.this) + terms(n.expression) if isinstance(n, exp.And) else [n]
                for eq in terms(j.args.get('on')):
                    if isinstance(eq, exp.EQ) and any((same_bound_column(eq.this, c, bindings) and same_bound_column(eq.expression, g, bindings) or (same_bound_column(eq.expression, c, bindings) and same_bound_column(eq.this, g, bindings)) for g in same)):
                        equal = True
            if not equal:
                return (None, 'UNKNOWN_retained_output_not_group_key_or_local_equality')
        observed = set(shape['types'][output_i]) - {'NULL', 'UNKNOWN'}
        declared = affinity(col['type'])
        if observed and (not (observed <= {declared} or (declared in ['INTEGER', 'REAL', 'NUMERIC'] and observed <= {'INTEGER', 'REAL'}))):
            return (None, 'UNKNOWN_retained_output_example_type_mismatch')
        retained.append(dict(SQL=c.sql(dialect='sqlite'), physical_table=table['name'], column=col['name'], declared_affinity=declared, example_types=shape['types'][output_i]))
    aliases = {p.alias.lower() for _, p, _ in aggs if p.alias}
    for key in ['order', 'having', 'group', 'where']:
        node = tree.args.get(key)
        if node and any((not c.table and c.name.lower() in aliases for c in node.find_all(exp.Column))):
            return (None, 'UNKNOWN_removed_aggregate_alias_still_referenced')
    out = tree.copy()
    out.set('expressions', [out.expressions[i] for i, _, _ in keep])
    assert all((out.args.get(k) == tree.args.get(k) for k in set(tree.args) - {'expressions'}))
    return (out, dict(operation='DELETE_EXACT_EXCESS_UNREQUESTED_AGGREGATE_OUTPUTS', request_head=head, example_width=width, old_width=len(tree.expressions), removed=[p.sql(dialect='sqlite') for _, p, _ in aggs], retained=retained, ORDER_HAVING_aggregate_operations_unchanged=True, assumption='Request-head lexical contract, not an NL truth certificate'))

def normalize_anchors(tree, bindings, question):
    where = tree.args.get('where')
    if where is None:
        return (None, 'NO_WHERE_TEXT_ANCHOR')

    def terms(n):
        if isinstance(n, exp.Paren):
            return terms(n.this)
        if isinstance(n, exp.And):
            return terms(n.this) + terms(n.expression)
        return [n]
    out = tree.copy()
    changed = []
    for n in terms(out.args['where'].this):
        if not isinstance(n, exp.EQ):
            continue
        if isinstance(n.this, exp.Column) and isinstance(n.expression, exp.Literal) and n.expression.is_string:
            c, l = (n.this, n.expression)
        elif isinstance(n.expression, exp.Column) and isinstance(n.this, exp.Literal) and n.this.is_string:
            c, l = (n.expression, n.this)
        else:
            continue
        _, table, col = column_info(c, bindings)
        if affinity(col['type']) != 'TEXT':
            continue
        value = l.this
        if not value.isascii() or not re.search('[A-Za-z]', value) or len(value.strip()) < 2 or any((ord(ch) < 32 for ch in value)):
            continue
        if re.search('[^A-Za-z0-9 _\\-]', value) or re.search(' {2,}', value.strip()):
            continue
        pattern = '(?<!\\w)' + '\\s+'.join((re.escape(w) for w in value.strip().split(' '))) + '(?!\\w)'
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        replacement = exp.EQ(this=exp.Lower(this=exp.Trim(this=c.copy())), expression=exp.Literal.string(value.strip().lower()))
        before = n.sql(dialect='sqlite')
        n.replace(replacement)
        changed.append(dict(source_predicate=before, normalized_predicate=replacement.sql(dialect='sqlite'), literal=value, question_full_anchor=match.group(), question_span=[match.start(), match.end()], column=c.sql(dialect='sqlite'), physical_table=table['name'], TEXT_affinity=True))
    if not changed:
        return (None, 'NO_COMPLETE_QUESTION_TEXT_ANCHOR')
    assert all((out.args.get(k) == tree.args.get(k) for k in set(tree.args) - {'where'}))
    return (out, dict(operation='QUESTION_ANCHORED_ASCII_LOWER_AND_BOUNDARY_SPACE_TRIM', anchors=changed, assumption='Case and boundary ASCII spaces assumed irrelevant only for these full question anchors; never proved from labels', internal_whitespace_not_collapsed=True))

def generate(question, shape, prefix):
    """No id, DB, example values/names, suffix, model or label argument."""
    schema = next((e['observation'] for e in prefix if e['tool'] == 'schema'), None)
    checks = [(i, e['sql']) for i, e in enumerate(prefix, 1) if e['tool'] == 'check']
    if schema is None:
        return ([], ['UNKNOWN_no_paid_schema_within6'], [])
    old = {canonical(sql) for _, sql in checks} - {None}
    seen = set()
    found = []
    rejections = []
    sources = []
    for step, sql in reversed(checks):
        try:
            tree, bindings, ambig, _ = scope(sql, schema)
            if ambig:
                views, _, _ = maps_and_evidence(sql, schema)
                variants = [(scope(v['SQL'], schema)[0], v['mapping'], v['edits']) for v in views]
            else:
                variants = [(tree, {}, [])]
            detail = dict(source_step=step, ambiguous_names=sorted(ambig), binding_views=len(variants), projection_reasons=[], anchor_reasons=[])
            for bound, mapping, edits in variants:
                composed = bound
                ops = []
                if edits:
                    ops.append(dict(operation='ALL_POSITION_COLUMN_BINDING', mapping=mapping, edits=edits))
                projected, pr = projection_contract(composed, bindings, shape, question)
                if projected is not None:
                    composed = projected
                    ops.append(pr)
                else:
                    detail['projection_reasons'].append(pr)
                normalized, nr = normalize_anchors(composed, bindings, question)
                if normalized is not None:
                    composed = normalized
                    ops.append(nr)
                else:
                    detail['anchor_reasons'].append(nr)
                if not ops:
                    continue
                if len(composed.expressions) != shape['width']:
                    detail['projection_reasons'].append('UNKNOWN_complete_candidate_output_width_still_mismatches_example')
                    continue
                final = composed.sql(dialect='sqlite')
                key = canonical(final)
                if key is None or key in old or key in seen:
                    continue
                scope(final, schema)
                seen.add(key)
                if len(found) < 2:
                    found.append(dict(SQL=final, SQL_SHA256=h(final), canonical=key, source_step=step, source_SQL=sql, operations=ops, composition_size=len(ops), output_width=len(composed.expressions), public_outcome='UNKNOWN_NOT_EXECUTED', all3='UNKNOWN_NOT_SCORED', semantic_status='CONDITIONAL_NL_HYPOTHESIS_NOT_CERTIFICATE'))
            sources.append(detail)
        except (ValueError, ParseError, TokenError, OptimizeError) as e:
            _ordinary(e)
            rejections.append(dict(source_step=step, reason=_rejection_type(e) + ':' + str(e)[:180]))
    return (found, rejections, sources)
