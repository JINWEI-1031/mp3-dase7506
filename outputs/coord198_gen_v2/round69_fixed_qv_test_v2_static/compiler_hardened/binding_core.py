from sqlglot.errors import ParseError, TokenError, OptimizeError
from .errors import _ordinary, _rejection_type, RuleRefusal
'Frozen R32 pure physical binding/AST functions only.'
import itertools
from collections import Counter
from sqlglot import exp, parse_one
from sqlglot.optimizer.qualify import qualify

def affinity(t):
    t = t.upper()
    if 'INT' in t:
        return 'INTEGER'
    if any((x in t for x in ['CHAR', 'CLOB', 'TEXT'])):
        return 'TEXT'
    if any((x in t for x in ['REAL', 'FLOA', 'DOUB'])):
        return 'REAL'
    if not t or 'BLOB' in t:
        return 'BLOB'
    return 'NUMERIC'

def scope(sql, schema):
    tree = parse_one(sql, read='sqlite')
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1 or any(tree.find_all(exp.With, exp.CTE, exp.Subquery, exp.Window)):
        raise RuleRefusal('UNKNOWN_complex_scope')
    if any((j.args.get('using') or j.args.get('method') == 'NATURAL' for j in tree.find_all(exp.Join))):
        raise RuleRefusal('UNKNOWN_using_or_natural_join_scope')
    tables = {t['name'].lower(): t for t in schema}
    bindings = {}
    for t in tree.find_all(exp.Table):
        if t.db or t.catalog or t.name.lower() not in tables or (not isinstance(t.this, exp.Identifier)):
            raise RuleRefusal('UNKNOWN_physical_table')
        if t.alias_or_name.lower() in bindings:
            raise RuleRefusal('UNKNOWN_duplicate_alias')
        bindings[t.alias_or_name.lower()] = (t.alias_or_name, tables[t.name.lower()])
    if not bindings:
        raise RuleRefusal('UNKNOWN_no_physical_scope')
    aliases = {p.alias.lower() for p in tree.expressions if p.alias}
    ambiguous = {}
    unqualified = {}
    for c in tree.find_all(exp.Column):
        if c.is_star:
            continue
        if c.db or c.catalog:
            raise RuleRefusal('UNKNOWN_external_column')
        owners = [(alias, actual, t, col) for alias, (actual, t) in bindings.items() for col in t['columns'] if col['name'].lower() == c.name.lower() and (not c.table or alias == c.table.lower())]
        if not owners:
            if not c.table and c.name.lower() in aliases:
                continue
            raise RuleRefusal('UNKNOWN_missing_column_or_DQS_literal')
        if c.table:
            if len(owners) != 1:
                raise RuleRefusal('UNKNOWN_qualified_binding')
            continue
        name = c.name.lower()
        if name in aliases:
            raise RuleRefusal('UNKNOWN_projection_alias_shadowing')
        unqualified[name] = owners
        if len(owners) > 1:
            if len({affinity(o[3]['type']) for o in owners}) != 1:
                raise RuleRefusal('UNKNOWN_homonym_type_mismatch')
            ambiguous[name] = owners
    return (tree, bindings, ambiguous, unqualified)

def position(c):
    p = c.parent
    while p is not None and (not isinstance(p, exp.Select)):
        if isinstance(p, exp.Join):
            return 'JOIN'
        if isinstance(p, exp.Where):
            return 'WHERE'
        if isinstance(p, exp.Group):
            return 'GROUP'
        if isinstance(p, exp.Order):
            return 'ORDER'
        if isinstance(p, exp.Having):
            return 'HAVING'
        p = p.parent
    return 'SELECT'

def qualify_view(tree, mapping):
    out = tree.copy()
    edits = []
    for c in out.find_all(exp.Column):
        if not c.table and c.name.lower() in mapping:
            edits.append(dict(name=c.name, position=position(c), alias=mapping[c.name.lower()]))
            c.set('table', exp.to_identifier(mapping[c.name.lower()]))
    restored = out.copy()
    for c in restored.find_all(exp.Column):
        if c.name.lower() in mapping and c.table == mapping[c.name.lower()]:
            pass
    for old, new in zip(tree.find_all(exp.Column), restored.find_all(exp.Column)):
        if not old.table and old.name.lower() in mapping:
            new.set('table', None)
    assert restored == tree
    for eq in out.find_all(exp.EQ):
        if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column) and (eq.this == eq.expression):
            if not any((e == eq for e in tree.find_all(exp.EQ))):
                raise RuleRefusal('UNKNOWN_join_or_filter_self_equality_created')
    return (out, edits)

def canonical(sql):
    try:
        t = parse_one(sql, read='sqlite')
        if not isinstance(t, exp.Select) or len(list(t.find_all(exp.Select))) != 1 or any(t.find_all(exp.CTE, exp.With, exp.Subquery)):
            return None
        mapping = {}
        for i, tb in enumerate(t.find_all(exp.Table)):
            old = tb.alias_or_name.lower()
            if old in mapping:
                return None
            mapping[old] = 'bind' + str(i)
            tb.set('alias', exp.TableAlias(this=exp.to_identifier(mapping[old])))
        for c in t.find_all(exp.Column):
            if c.table:
                if c.table.lower() not in mapping:
                    return None
                c.set('table', exp.to_identifier(mapping[c.table.lower()]))
        for n in t.find_all(exp.Identifier):
            n.set('this', n.this.lower())
            n.set('quoted', False)
        return t.sql(dialect='sqlite')
    except (ValueError, ParseError, TokenError, OptimizeError) as _rejection:
        _ordinary(_rejection)
        return None

def maps_and_evidence(sql, schema):
    tree, bindings, ambig, unqualified = scope(sql, schema)
    if not ambig:
        raise RuleRefusal('UNKNOWN_no_unqualified_ambiguous_column')
    if len(ambig) > 4 or any((len(v) > 4 for v in ambig.values())):
        raise RuleRefusal('UNKNOWN_static_search_cap')
    evidence = []
    for j in tree.find_all(exp.Join):
        if j.args.get('side') not in [None, ''] or j.args.get('kind') not in [None, '', 'INNER']:
            continue
        on = j.args.get('on')
        if on is None:
            continue

        def conjunction(p):
            if isinstance(p, exp.Paren):
                return conjunction(p.this)
            if isinstance(p, exp.And):
                return conjunction(p.this) + conjunction(p.expression)
            return [p]
        for eq in conjunction(on):
            if isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column) and eq.this.table and eq.expression.table and (eq.this.table.lower() != eq.expression.table.lower()) and (eq.this.name.lower() == eq.expression.name.lower()):
                evidence.append(dict(name=eq.this.name, aliases=[eq.this.table, eq.expression.table], SQL=eq.sql(dialect='sqlite'), scope='positive conjunctive INNER ON equality only; value compatibility not NL ownership proof'))
    names = sorted(ambig)
    assignments = []
    for owners in itertools.islice(itertools.product(*[sorted({o[1] for o in ambig[name]}, key=str.lower) for name in names]), 64):
        assignments.append(dict(zip(names, owners)))
    result = []
    fail = Counter()
    schema_map = {t['name']: {c['name']: c['type'] or 'UNKNOWN' for c in t['columns']} for _, t in bindings.values()}
    for mapping in assignments:
        try:
            t, edits = qualify_view(tree, mapping)
            qualify(t.copy(), dialect='sqlite', schema=schema_map, infer_schema=False, identify=False)
            result.append(dict(SQL=t.sql(dialect='sqlite'), mapping=mapping, edits=edits, ambiguous_names=names, total_positions=len(edits), positions=dict(Counter((e['position'] for e in edits))), INNER_same_name_equality_evidence=evidence, all_nonqualifier_AST_unchanged=True, semantic_certificate=False))
        except (ValueError, ParseError, TokenError, OptimizeError) as e:
            _ordinary(e)
            fail[_rejection_type(e) + ':' + str(e)[:100]] += 1
    global_ambig = []
    global_all = []
    for alias, (actual, table) in sorted(bindings.items()):
        columns = {c['name'].lower() for c in table['columns']}
        for nameset, output in [(set(ambig), global_ambig), (set(unqualified), global_all)]:
            if not nameset <= columns:
                continue
            try:
                t, _ = qualify_view(tree, {n: actual for n in nameset})
                qualify(t.copy(), dialect='sqlite', schema=schema_map, infer_schema=False, identify=False)
                output.append(t.sql(dialect='sqlite'))
            except (ValueError, ParseError, TokenError, OptimizeError) as _rejection:
                _ordinary(_rejection)
                pass
    return (result, dict(fail), dict(all_ambiguous_names_to_one_alias=global_ambig, all_unqualified_columns_to_one_alias=global_all))
