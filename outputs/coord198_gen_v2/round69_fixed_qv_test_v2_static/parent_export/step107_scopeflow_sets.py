"""Evidence-bound two-category existential plans, preserving the original joins."""
import re
from sqlglot import exp
from repair_rules import parse


def request(sql, obs, schema):
    q = obs['task']['question'].lower().strip().rstrip('?.')
    match = re.search(r'\bboth ([a-z]+) and ([a-z]+)$', q)
    negative = re.search(r'\b([a-z]+) but not ([a-z]+)$', q)
    if not (match or negative): return None, 'no_explicit_two_category_scope'
    values = list((match or negative).groups())
    if values[0] == values[1]: return None, 'identical_categories'
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return None, 'nonflat_scope'
    root = tree.args.get('from')
    if not root or not isinstance(root.this, exp.Table): return None, 'non_table_root'
    alias = root.this.alias_or_name
    projections = tree.selects
    if not all(isinstance(x.unalias(), exp.Column) and x.unalias().table.lower() == alias.lower() for x in projections):
        return None, 'non_root_output'
    if tree.args.get('order') or tree.args.get('limit') or tree.args.get('offset'):
        return None, 'ranked_scope'
    where = tree.args.get('where')
    if where is None: return None, 'no_category_predicate'
    def flatten(e): return flatten(e.this)+flatten(e.expression) if isinstance(e, exp.And) else [e]
    terms = flatten(where.this)
    cats = [x for x in terms if isinstance(x, exp.In) and isinstance(x.this, exp.Column)
            and x.this.table and x.this.table.lower()!=alias.lower()
            and len(x.expressions)==2 and all(isinstance(z, exp.Literal) and z.is_string for z in x.expressions)]
    if len(cats)!=1: return None, 'ambiguous_category_predicate'
    cat = cats[0]
    rest = [x for x in terms if x is not cat]
    if any(not x.table or x.table.lower()!=alias.lower() for t in rest for x in t.find_all(exp.Column)):
        return None, 'extra_child_filter_scope'
    tables={t.alias_or_name.lower():t.name.lower() for t in tree.find_all(exp.Table)}
    metadata={t['name'].lower():t for t in schema}
    if any(t not in metadata for t in tables.values()): return None, 'unknown_relation'
    if cat.this.table.lower() not in tables:return None,'unbound_category_role'
    child=metadata[tables[cat.this.table.lower()]]
    field=next((c for c in child['columns'] if c['name'].lower()==cat.this.name.lower()),None)
    if not field or not any(t in field['type'].lower() for t in ('text','char')):return None,'nontext_category'
    rootmeta=metadata[root.this.name.lower()]
    keys=[c['name'] for c in rootmeta['columns'] if c.get('pk')]
    if len(keys)!=1:return None,'non_simple_root_key'
    # Every join must be a declared simple equality; no removed relation or filter.
    for j in tree.args.get('joins',[]):
        on=j.args.get('on')
        if j.args.get('side') or j.args.get('kind') or not isinstance(on,exp.EQ) or not all(isinstance(x,exp.Column) for x in (on.this,on.expression)):
            return None,'non_simple_inner_join'
        a,b=on.this,on.expression
        if not a.table or not b.table or a.table.lower() not in tables or b.table.lower() not in tables:return None,'unbound_join'
        def declared(a,b):
            return any(f['from'].lower()==a.name.lower() and f['to'].lower()==b.name.lower() and f['table'].lower()==tables[b.table.lower()] for f in metadata[tables[a.table.lower()]].get('foreign_keys',[]))
        if not (declared(a,b) or declared(b,a)):return None,'undeclared_join'
    group=tree.args.get('group');having=tree.args.get('having')
    if group and any(not isinstance(x,exp.Column) or x.table.lower()!=alias.lower() for x in group.expressions):return None,'nonroot_group'
    if having:
        h=having.this
        if not (isinstance(h,exp.EQ) and isinstance(h.this,exp.Count) and isinstance(h.this.this,exp.Distinct)
                and len(h.this.this.expressions)==1 and h.this.this.expressions[0]==cat.this
                and isinstance(h.expression,exp.Literal) and h.expression.this=='2' and match):return None,'unsupported_having'
    return {'sql':sql,'values':values,'negative':bool(negative),'root_alias':alias,'root_table':root.this.name,
            'key':keys[0],'field':cat.this.sql(dialect='sqlite'),
            'probe':exp.select(exp.column(field['name'])).distinct().from_(child['name']).limit(9).sql(dialect='sqlite')}, 'two_category_scope'


def generate(plan, feedback):
    if not feedback.get('ok') or feedback.get('truncated'):return None,'unavailable_or_truncated_values'
    observed=[r[0] for r in feedback['rows'] if len(r)==1 and isinstance(r[0],str)]
    values=[]
    for v in plan['values']:
        matches=[z for z in observed if z.lower()==v]
        if len(matches)!=1:return None,'category_not_uniquely_observed'
        values.append(matches[0])
    tree=parse(plan['sql']);alias=plan['root_alias'];outalias='scopeflow_root'
    if any(t.alias_or_name.lower()==outalias for t in tree.find_all(exp.Table)):return None,'alias_collision'
    target=parse('SELECT '+plan['field']).selects[0]
    def flatten(e):return flatten(e.this)+flatten(e.expression) if isinstance(e,exp.And) else [e]
    terms=flatten(tree.args['where'].this)
    rest=[x for x in terms if not (isinstance(x,exp.In) and x.this==target)]
    conditions=[]
    for i,v in enumerate(values):
        inner=tree.copy();inner.set('expressions',[exp.Literal.number(1)])
        for key in ('group','having','order','limit','distinct'):inner.set(key,None)
        predicates=[x.copy() for x in rest]+[exp.EQ(this=target.copy(),expression=exp.Literal.string(v)),
            exp.EQ(this=exp.column(plan['key'],table=alias),expression=exp.column(plan['key'],table=outalias))]
        inner.set('where',exp.Where(this=exp.and_(*predicates)))
        exists=exp.Exists(this=inner)
        conditions.append(exp.Not(this=exists) if i==1 and plan['negative'] else exists)
    def rename(e):
        e=e.copy()
        for c in e.find_all(exp.Column):
            if c.table.lower()==alias.lower():c.set('table',exp.to_identifier(outalias))
        return e
    outer=exp.select(*[rename(x) for x in tree.selects]).from_(exp.to_table(plan['root_table']).as_(outalias))
    outer.set('distinct',tree.args.get('distinct'))
    outer.set('where',exp.Where(this=exp.and_(*[rename(x) for x in rest],*conditions)))
    return outer.sql(dialect='sqlite'),'observed_categories_and_correlated_scope'
