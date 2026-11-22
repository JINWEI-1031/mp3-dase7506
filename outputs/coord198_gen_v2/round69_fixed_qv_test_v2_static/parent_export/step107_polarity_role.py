"""Observed role-prefixed category binding, with current-candidate preservation."""
import re
from sqlglot import exp
from repair_rules import parse
from step107_polarity_clauses import candidates


def norm(s):return ' '.join(s.lower().split())


def request(sql,obs,schema):
    tree=parse(sql)
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return None,'nonflat_role_scope'
    metadata={t['name'].lower():t for t in schema};bound={t.alias_or_name.lower():t.name.lower() for t in tree.find_all(exp.Table)}
    if any(t not in metadata for t in bound.values()):return None,'unknown_role_relation'
    headers=obs['example']['columns']
    if len(headers)!=len(tree.selects):return None,'role_output_width'
    def owners(c):
        return [a for a,t in bound.items() if (not c.table or c.table.lower()==a) and any(z['name'].lower()==c.name.lower() for z in metadata[t]['columns'])]
    # A missing output property may be corrected only by the independent exact
    # header/unknown-column certificate; a valid name/topic column is not replaced.
    if any(not isinstance(e.unalias(),exp.Column) or e.unalias().name.lower()!=h.lower() for e,h in zip(tree.selects,headers)):
        pool,_=candidates(sql,obs,schema);pool=[c for c in pool if c['source']=='unknown_projection_contract']
        if len(pool)!=1:return None,'unresolved_output_role'
        tree=parse(pool[0]['sql']);sql=pool[0]['sql']
    if any(len(owners(c))!=1 for c in tree.find_all(exp.Column)):return None,'ambiguous_role_column'
    where=tree.args.get('where')
    if where is None:return None,'no_role_filter'
    def leaves(e):return leaves(e.this)+leaves(e.expression) if isinstance(e,exp.And) else [e]
    proposals=[];q=obs['task']['question']
    for i,e in enumerate(leaves(where.this)):
        if not (isinstance(e,exp.EQ) and isinstance(e.this,exp.Column) and isinstance(e.expression,exp.Literal) and e.expression.is_string):continue
        c=e.this;alias=owners(c)[0];table=metadata[bound[alias]];field=next(z for z in table['columns'] if z['name'].lower()==c.name.lower())
        role=c.name.lower();value=e.expression.this
        # Single-word categorical role. Names/IDs/time require different evidence.
        if not re.fullmatch(r'[a-z]+',role) or role in {'name','id','date','time','firstname','lastname'} or field.get('pk') or not any(t in field['type'].lower() for t in ('text','char')):continue
        if not value or len(value)>80 or any(x in value for x in ('%','_')):continue
        phrase=role+' '+value
        anchor=re.search(r'\b(?:on|in|for|with|of|from|have|has)\s+'+re.escape(phrase)+r'(?!\w)',q,re.I)
        if anchor is None:continue
        outside=q[:anchor.start()]+q[anchor.end():]
        if set(re.findall(r'[a-z]+',outside.lower())) & {'not','no','neither','nor','without','never','except','exclude','excluding','both','either','or','but'}:
            continue
        proposals.append({'ordinal':i,'column':c.sql(dialect='sqlite'),'field':field['name'],'table':table['name'],'alias':alias,'old_value':value,'question_phrase':phrase})
    if len(proposals)!=1:return None,'no_unique_role_phrase'
    p=proposals[0]
    p.update(sql=sql,probe=exp.select(exp.column(p['field'])).distinct().from_(p['table']).limit(9).sql(dialect='sqlite'))
    return p,'schema_role_plus_question_phrase'


def generate(plan,feedback):
    if not feedback.get('ok') or feedback.get('truncated'):return None,'role_values_unavailable_or_truncated'
    observed=[r[0] for r in feedback['rows'] if len(r)==1 and isinstance(r[0],str)]
    if any(norm(v)==norm(plan['old_value']) for v in observed):return None,'current_value_already_observed'
    choices=[v for v in observed if norm(v)==norm(plan['question_phrase'])]
    if len(choices)!=1:return None,'no_unique_observed_full_role_value'
    tree=parse(plan['sql'])
    def leaves(e):return leaves(e.this)+leaves(e.expression) if isinstance(e,exp.And) else [e]
    target=leaves(tree.args['where'].this)[plan['ordinal']]
    if not isinstance(target,exp.EQ) or target.expression.this!=plan['old_value']:return None,'role_ast_address_mismatch'
    target.set('expression',exp.Literal.string(choices[0]))
    return tree.sql(dialect='sqlite'),'observed_full_question_role_value'
