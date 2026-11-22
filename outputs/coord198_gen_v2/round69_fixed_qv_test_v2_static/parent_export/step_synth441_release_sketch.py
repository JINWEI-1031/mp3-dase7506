"""Only explicit conflicts across source hypotheses are hard; uncertainty remains soft."""
import re
from sqlglot import exp
from repair_rules import parse
from step124_pool_search import sources
from step_scope441_ir import table
from step_semantic192_rules import tokens
from step_synth441_sketch import predicate_contract,generate as raw_generate,static_check as raw_static
from step_synth441_seeded_sketch import generate as seeded,named_seed_population,current_question_conditions_kept
def count_head(q,schema):
    m=re.search(r'\b(?:how many|number of)\s+(?:(?:distinct|different|the|total)\s+)?(.+)',q,re.I)
    if not m:return set(),None
    stop={'for','with','who','which','that','where','in','on','of','from','by','do','does','did','is','are','was','were','have','has','had','can','could','would','should','must','either','neither','both','without','at','per','each','every','respectively','and'}
    phrase=[]
    for word in re.findall(r'[a-z_]+',m[1].lower()):
        if word in stop or len(phrase)>=4:break
        phrase.append(word)
    entities=set().union(*(tokens(tb['name'])|set().union(*(tokens(c['name']) for c in tb['columns'] if c.get('pk'))) for tb in schema))
    matches=[tokens(w) for w in phrase if tokens(w)&entities]
    return (matches[-1] if matches else tokens(phrase[0]) if phrase else set()),(phrase[0] if phrase else None)
def hard_conflict(c,current,obs,schema):
    q=obs['task']['question'];t=parse(c['sql']);old=parse(current)
    if c['seed_source']=='issue_role_evidence' and old is not None and re.search(r'\bnull\b',q,re.I):
        needed=predicate_contract(old,schema);actual=predicate_contract(t,schema)
        if needed is None or actual is None:return 'explicit_null_scope_unresolved'
        for sig,n in needed.items():
            if 'IS NULL' in sig.upper() and actual[sig]<n:return 'explicit_null_predicate_lost_across_sources'
    seed=sources(parse(obs.get('issue_sql','')),schema)
    if c['family']=='count_membership' and len(seed)==1:
        base=next(iter(seed.values()));words=tokens(base['name'])-{'record','data','list'}
        noun,_=count_head(q,schema)
        if noun and words&noun:
            root=table(schema,c['slots']['root']);root_words=tokens(root['name'])|set().union(*(tokens(x['name']) for x in root['columns'] if x.get('pk')))
            if not root_words&noun:return 'explicit_counted_entity_cannot_be_replaced_by_attribute_owner'
        negative=any(re.search(r'\b(?:without|with\s+no)\s+(?:any\s+|the\s+)?'+re.escape(w)+r's?\b',q,re.I) for w in words)
        if negative:
            # This grammar's named-population qualifier is positive membership.
            # Do not mistake it for the requested anti-membership.
            return 'positive_seed_population_conflicts_with_explicit_without_entity'
    if re.search(r'\bnot\s+both\b|\bneither\b',q,re.I) and not t.find(exp.Not) and not t.find(exp.Or):
        attrs={e.this.name.lower() for w in t.find_all(exp.Where) for e in w.find_all(exp.EQ) if isinstance(e.this,exp.Column) and isinstance(e.expression,exp.Literal) and e.expression.this.lower() in ['1','yes','true'] and tokens(e.this.name)<=tokens(q)}
        if len(attrs)>=2:return 'positive_attribute_conjunction_conflicts_with_explicit_negation'
    return None
def generate(current,obs,schema,family='combined'):
    pool,why=seeded(current,obs,schema,family)
    noun,first=count_head(obs['task']['question'],schema)
    # A compound counted noun (e.g. phone models) must not be hard rejected
    # merely because the old single-word guard read the modifier as the object.
    if first and noun!=tokens(first) and family in ['count','combined']:
        for source in dict.fromkeys([current,obs.get('issue_sql',current)]):
            candidates,_=raw_generate(source,obs,schema,'count')
            for c in candidates:
                if not c['static_admit']:continue
                root=table(schema,c['slots']['root']);ks=set().union(*(tokens(x['name']) for x in root['columns'] if x.get('pk')))
                if not ks&noun:continue
                c=named_seed_population(c,source,obs,schema);c['seed_source']='current' if source==current else 'issue_role_evidence'
                if source!=current and not current_question_conditions_kept(c,current,obs,schema):continue
                existing=next((x for x in pool if x['sql']==c['sql']),None)
                if existing and existing.get('static_reason')=='explicit_counted_entity_changed_to_other_key_role':existing.update(static_admit=True,static_reason=None)
                elif not existing:pool.append(c)
    for c in pool:
        c['current_parent']=current
        if c['static_admit']:
            reason=hard_conflict(c,current,obs,schema)
            if reason:c['static_admit']=False;c['static_reason']=reason
    return sorted(pool,key=lambda c:-c['priority'])[:24],why
def static_check(c,obs,schema):
    ok,reason=raw_static(c,obs,schema)
    if not ok:return ok,reason
    reason=hard_conflict(c,c.get('current_parent',c['parent']),obs,schema)
    return reason is None,reason
