"""Closed question grammar -> Boolean/quantifier IR -> scoped SQL.

Only runtime question/schema/SQL and charged distinct-value feedback are used.
Logical NOT is represented before SQL compilation, never an operator swap.
"""
import copy
import re
from sqlglot import exp
from repair_rules import parse
from step107_scopeflow_sets import request as structural_contract


def node(op, *args):
    return {'op': op, 'args': list(args)}


def _atom(text):
    m = re.fullmatch(r"([a-z][a-z0-9_]*)(?: with ([a-z][a-z0-9_]*) (>=|<=|!=|=|>|<|greater than|less than|at least|at most|equal to) ('[^']+'|-?\d+(?:\.\d+)?))?", text.strip())
    if not m:
        return None
    value, field, operator, literal = m.groups()
    result = {'op': 'atom', 'value': value, 'filters': []}
    if field:
        result['filters'].append({'field': field, 'operator': operator, 'literal': literal})
    return result


def atoms(ir):
    if ir['op'] == 'atom':
        return [ir]
    return [x for arg in ir['args'] for x in atoms(arg)]


def question_ir(question):
    q = ' '.join(question.lower().strip().rstrip('?.').split())
    q = q.replace("don't ", 'do not ').replace("doesn't ", 'does not ')
    m = re.search(r'\b(?:(do not|does not|did not|not) )?(have|has|had|broadcast|broadcasts|occur|occurs)(?: in)? (.+)$', q)
    if not m:
        return None, 'unrecognized_relation_clause'
    prefix = q[:m.start()]
    # These are unbound logical tokens in the parsed prefix, not a NOT blacklist:
    # all supported negations are consumed and represented in the syntax tree.
    if set(re.findall(r'[a-z]+', prefix)) & {'not', 'no', 'neither', 'nor', 'both', 'without', 'never', 'except', 'either', 'only', 'all'}:
        return None, 'unbound_logical_operator'
    outer_not = bool(m[1]); body = m[3]; binding = 'entity'; row_role = None
    single = re.match(r'(?:a single|an individual) ([a-z][a-z0-9_]*) that is (.+)$', body)
    if single:
        row_role, body = single.groups(); binding = 'row'
    common = []
    cm = re.fullmatch(r"(.+), with ([a-z][a-z0-9_]*) (>=|<=|!=|=|>|<|greater than|less than|at least|at most|equal to) ('[^']+'|-?\d+(?:\.\d+)?) for each (?:event|row|relation)", body)
    if cm:
        body, field, operator, literal = cm.groups()
        common = [{'field': field, 'operator': operator, 'literal': literal}]
    form = None
    if body.startswith('not both '):
        parts = body[9:].split(' and '); form = 'not_both'
    elif body.startswith('both '):
        parts = body[5:].split(' and '); form = 'both'
    elif body.startswith('neither '):
        parts = body[8:].split(' nor '); form = 'neither'
    elif body.startswith('not ') and ' but ' in body:
        parts = body[4:].split(' but '); form = 'not_a_b'
    elif ' but not ' in body:
        parts = body.split(' but not '); form = 'a_not_b'
    else:
        return None, 'unrecognized_boolean_clause'
    if len(parts) != 2 or any(_atom(p) is None for p in parts):
        return None, 'ambiguous_or_incomplete_atoms'
    a, b = [_atom(p) for p in parts]
    if a['value'] == b['value']:
        return None, 'identical_categories'
    logic = {'both': node('and', a, b), 'not_both': node('not', node('and', a, b)),
             'neither': node('and', node('not', a), node('not', b)),
             'not_a_b': node('and', node('not', a), b),
             'a_not_b': node('and', a, node('not', b))}[form]
    # Linguistic double negation with neither/not-but is not normalized by guess.
    if outer_not and form != 'both':
        return None, 'ambiguous_nested_linguistic_negation'
    if outer_not:
        logic = node('not', logic); form = 'not_both'
    return {'binding': binding, 'row_role': row_role, 'logic': logic,
            'form': form, 'common_filters': common, 'outer_relation_not': outer_not}, 'explicit_quantifier_ir'


def _flatten(e):
    return _flatten(e.this) + _flatten(e.expression) if isinstance(e, exp.And) else [e]


def _key(e):
    return e.sql(dialect='sqlite').lower()


def _predicate(spec, child, alias):
    field = next((c for c in child['columns'] if c['name'].lower() == spec['field']), None)
    if field is None:
        return None
    operator = {'greater than': '>', 'less than': '<', 'at least': '>=', 'at most': '<=', 'equal to': '='}.get(spec['operator'], spec['operator'])
    raw = spec['literal']; string = raw.startswith("'")
    text_type = any(t in field['type'].lower() for t in ('char', 'text'))
    if string != text_type:
        return None
    if not string and not any(t in field['type'].lower() for t in ('int','real','numeric','number','double','float','decimal')):
        return None
    literal = exp.Literal.string(raw[1:-1]) if string else exp.Literal.number(raw)
    cls = {'>': exp.GT, '<': exp.LT, '>=': exp.GTE, '<=': exp.LTE, '=': exp.EQ, '!=': exp.NEQ}[operator]
    return cls(this=exp.column(field['name'], table=alias), expression=literal)


def request(sql, obs, schema, *, scoped=False):
    ir, reason = question_ir(obs['task']['question'])
    if ir is None:
        return None, reason
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return None, 'nonflat_scope'
    root = tree.args.get('from'); where = tree.args.get('where')
    if not root or not isinstance(root.this, exp.Table) or not where:
        return None, 'missing_root_or_predicate'
    alias = root.this.alias_or_name
    terms = _flatten(where.this)
    categories = [x for x in terms if isinstance(x, exp.In) and isinstance(x.this, exp.Column)
                  and x.this.table and x.this.table.lower() != alias.lower()
                  and len(x.expressions) == 2 and all(isinstance(z, exp.Literal) and z.is_string for z in x.expressions)]
    if len(categories) != 1:
        return None, 'ambiguous_category_predicate'
    cat = categories[0]; root_terms = []; child_terms = []
    for term in terms:
        if term is cat:
            continue
        columns = list(term.find_all(exp.Column))
        if columns and all(c.table.lower() == alias.lower() for c in columns):
            root_terms.append(term)
        else:
            child_terms.append(term)
    stripped = tree.copy()
    stripped.set('where', exp.Where(this=exp.and_(cat.copy(), *[x.copy() for x in root_terms])))
    values = [a['value'] for a in atoms(ir['logic'])]
    canonical = {'task': {'question': 'Relations in both ' + values[0] + ' and ' + values[1]}}
    plan, reason = structural_contract(stripped.sql(dialect='sqlite'), canonical, schema)
    if plan is None:
        return None, reason
    bound = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}
    metadata = {t['name'].lower(): t for t in schema}
    child = metadata[bound[cat.this.table.lower()]]
    if ir['binding'] == 'row' and ir['row_role'].rstrip('s') != child['name'].lower().rstrip('s'):
        return None, 'single_row_relation_role_mismatch'
    all_specs = ir['common_filters'] + [s for a in atoms(ir['logic']) for s in a['filters']]
    if (all_specs or child_terms) and not scoped:
        return None, 'local_scope_extension_disabled'
    all_filters = [_predicate(s, child, cat.this.table) for s in all_specs]
    if any(p is None for p in all_filters):
        return None, 'filter_type_or_owner_mismatch'
    # Each original child predicate must have an explicit common/atom owner in NL.
    if any(_key(t) not in {_key(x) for x in all_filters} for t in child_terms):
        return None, 'unassigned_child_predicate'
    if ir['binding'] == 'row' and any(a['filters'] for a in atoms(ir['logic'])):
        return None, 'ambiguous_filter_inside_row_negation'
    plan.update(ir=ir, original_sql=sql, root_terms=[x.sql(dialect='sqlite') for x in root_terms],
                child_metadata=child, child_alias=cat.this.table, scoped=scoped,
                evidence={'quantifier': ir['form'], 'binding': ir['binding'],
                          'child_predicate_owners': len(child_terms), 'question_filters': len(all_specs)})
    return plan, 'quantifier_and_owned_predicate_contract'


def generate(plan, feedback):
    if not feedback.get('ok') or feedback.get('truncated'):
        return None, 'unavailable_or_truncated_values'
    observed = [r[0] for r in feedback['rows'] if len(r) == 1 and isinstance(r[0], str)]
    values = {}
    for a in atoms(plan['ir']['logic']):
        matches = [v for v in observed if v.lower() == a['value']]
        if len(matches) != 1:
            return None, 'category_not_uniquely_observed'
        values[a['value']] = matches[0]
    tree = parse(plan['sql']); root_alias = plan['root_alias']; out_alias = 'polarity_root'
    if any(t.alias_or_name.lower() == out_alias for t in tree.find_all(exp.Table)):
        return None, 'alias_collision'
    target = parse('SELECT ' + plan['field']).selects[0]
    def predicate(atom):
        conditions = [exp.EQ(this=target.copy(), expression=exp.Literal.string(values[atom['value']]))]
        conditions += [_predicate(s, plan['child_metadata'], plan['child_alias']) for s in atom['filters']]
        return exp.and_(*conditions)
    def boolean(ir, atom_compiler):
        if ir['op'] == 'atom':
            return atom_compiler(ir)
        args = [boolean(x, atom_compiler) for x in ir['args']]
        return exp.Not(this=exp.Paren(this=args[0])) if ir['op'] == 'not' else exp.and_(*args)
    common = [_predicate(s, plan['child_metadata'], plan['child_alias']) for s in plan['ir']['common_filters']]
    def exists(condition):
        inner = tree.copy(); inner.set('expressions', [exp.Literal.number(1)])
        for key in ('group', 'having', 'order', 'limit', 'distinct'):
            inner.set(key, None)
        correlated = exp.EQ(this=exp.column(plan['key'], table=root_alias), expression=exp.column(plan['key'], table=out_alias))
        inner.set('where', exp.Where(this=exp.and_(condition, *[x.copy() for x in common], correlated)))
        return exp.Exists(this=inner)
    ir = plan['ir']; logic = ir['logic']
    if ir['binding'] == 'entity':
        condition = boolean(logic, lambda a: exists(predicate(a)))
    else:
        # An outer relation negation wraps EXISTS; row-property negation stays in it.
        outer = ir['outer_relation_not']
        row_logic = logic['args'][0] if outer else logic
        condition = exists(boolean(row_logic, predicate))
        if outer:
            condition = exp.Not(this=exp.Paren(this=condition))
    def rename(e):
        e = e.copy()
        for c in e.find_all(exp.Column):
            if c.table.lower() == root_alias.lower():
                c.set('table', exp.to_identifier(out_alias))
        return e
    root_terms = [rename(parse('SELECT 1 WHERE ' + s).args['where'].this) for s in plan['root_terms']]
    outer = exp.select(*[rename(x) for x in tree.selects]).from_(exp.to_table(plan['root_table']).as_(out_alias))
    outer.set('distinct', copy.deepcopy(tree.args.get('distinct')))
    outer.set('where', exp.Where(this=exp.and_(*root_terms, condition)))
    return outer.sql(dialect='sqlite'), 'compiled_boolean_quantifier_ir'
