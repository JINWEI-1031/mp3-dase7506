"""Runtime-evidenced issue skeleton access; no blanket multiseed expansion."""
import re,collections
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_scope441_ir import linked,qi,col
from step_synth441_sketch import predicate_contract
from step_synth441_qualified_sketch import generate as qualified,static_check
def named_seed_population(c,seed,obs,schema):
    if c['family']!='count_membership':return c
    src=sources(parse(seed),schema)
    if len(src)!=1:return c
    base=next(iter(src.values()))
    if base['name']==c['slots']['root'] or not linked(base,obs['task']['question']):return c
    t=parse(c['sql'])
    if any(x.name.lower()==base['name'].lower() for x in t.find_all(exp.Table)):return c
    edge=c['slots']['edge'];base_alias='sk_required_population'
    if edge['child']==base['name']:condition=f'{col(base_alias,edge["child_field"])}={col("sk_root",edge["parent_field"])}'
    else:condition=f'{col("sk_root",edge["child_field"])}={col(base_alias,edge["parent_field"])}'
    sub=parse(f'SELECT 1 FROM {qi(base["name"])} AS {base_alias} WHERE {condition}')
    t.where(exp.Exists(this=sub),append=True,copy=False)
    return {**c,'sql':sql_text(t),'positive':c['positive']+['explicitly_named_seed_population_membership_preserved']}
def current_question_conditions_kept(c,current,obs,schema):
    t=parse(current);candidate=parse(c['sql']);q=obs['task']['question']
    if t is None:return True
    new=predicate_contract(candidate,schema)
    if new is None:return False
    def parts(n):
        while isinstance(n,exp.Paren):n=n.this
        return parts(n.this)+parts(n.expression) if isinstance(n,exp.And) else [n]
    for w in t.find_all(exp.Where):
        for atom in parts(w.this):
            values=[x.this for x in atom.find_all(exp.Literal) if re.search(r'(?<![a-z0-9])'+re.escape(x.this)+r'(?![a-z0-9])',q,re.I)]
            if not values:continue
            # Scope and operator, not only the literal, must survive across seeds.
            shell=w.find_ancestor(exp.Select).copy();shell.set('where',exp.Where(this=atom.copy()));sig=predicate_contract(shell,schema)
            if sig is None or any(new[k]<v for k,v in sig.items()):return False
    return True
def generate(current,obs,schema,family='combined'):
    pool,reasons=qualified(current,obs,schema,family);pool=[named_seed_population(c,current,obs,schema) for c in pool]
    issue=obs.get('issue_sql')
    if family in ['combined','count'] and issue and issue!=current:
        extra,why=qualified(issue,obs,schema,'count')
        # Strong runtime evidence: issue's missing attribute has one typed declared
        # owner/path and another question-supported root slot. No arbitrary hints.
        root=sources(parse(issue),schema);base=next(iter(root.values())) if len(root)==1 else None
        supported=base and any(c['static_admit'] and c['slots']['root']!=base['name'] for c in extra)
        reasons['issue_role_source']=why if not supported else 'positive_missing_owner_and_alternative_entity_role'
        if supported:
            for c in extra:
                c=named_seed_population(c,issue,obs,schema);c['seed_source']='issue_role_evidence'
                if not current_question_conditions_kept(c,current,obs,schema):c['static_admit']=False;c['static_reason']='explicit_question_condition_lost_across_seeds'
                pool.append(c)
    seen=set();out=[]
    for c in sorted(pool,key=lambda c:-c['priority']):
        if c['sql'] in seen:continue
        seen.add(c['sql']);out.append({**c,'seed_source':c.get('seed_source','current')})
    return out[:24],reasons
