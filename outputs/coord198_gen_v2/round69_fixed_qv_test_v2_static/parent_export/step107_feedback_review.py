"""Fixed signal ranking plus a separate, mandatory semantic proof gate."""
import re
from sqlglot import exp
from repair_rules import parse
from step107_rank_features import features,NAMES,key
from step107_rank_agent import certificate
from step107_frontier_agent import eligible
from step107_independent_agent import contradictions
from step107_polarity_clauses import candidates as local_candidates
from step107_slots_fields import semantic_column
from step107_slots_fields import words

def explicit_role(parent,candidate,obs):
    q=obs['task']['question'].lower().strip()
    m=re.match(r'^(?:what (?:is|are)|return|show|list|give)(?: me)?(?: the)?(?: distinct| unique)? ([a-z_]+)(?: of| for| from|[?.]|$)',q)
    role={'names':'name','topics':'topic','categories':'category','titles':'title','codes':'code','dates':'date','times':'time'}.get(m[1],m[1]) if m else None
    if role not in ('name','topic','category','title','code','date','time'):return False
    a,b=parse(parent),parse(candidate);changed=False
    for x,y in zip(a.selects,b.selects):
        if x==y:continue
        x,y=x.unalias(),y.unalias()
        if not isinstance(x,exp.Column) or not isinstance(y,exp.Column) or x.table.lower()!=y.table.lower():return False
        if role not in words(y.name) or role in words(x.name):return False
        changed=True
    return changed

def null_policy_preserved(parent,candidate):
    a,b=parse(parent),parse(candidate)
    if len(list(a.find_all(exp.Select)))!=1:return False
    match=re.search(r'\bORDER\s+BY\s+(.+?)(?:\bLIMIT\b|\bOFFSET\b|;?$)',parent,re.I|re.S)
    if not match:return False
    parts=match[1].split(',');old=a.args['order'].expressions;new=b.args['order'].expressions
    if len(parts)!=len(old):return False
    for text,x,y in zip(parts,old,new):
        explicit=re.search(r'\bNULLS\s+(FIRST|LAST)\b',text,re.I)
        desired=(explicit[1].lower()=='first') if explicit else not bool(y.args.get('desc'))
        if bool(y.args.get('nulls_first'))!=desired:return False
    return True

def feedback(sql,obs):
    return next((e['observation'] for e in reversed(obs['history']) if e['tool'] in ('check','execute') and key(e['sql'])==key(sql)),None)

def diagnose(sql,obs):
    fb=feedback(sql,obs);reasons=contradictions(sql,obs)
    if fb is not None:
        if not fb.get('ok'):reasons.append('actual_execution_failure')
        elif 'matches_example' in fb and not fb['matches_example']:reasons.append('actual_public_mismatch')
    v=dict(zip(NAMES,features(sql,obs)))
    if v['direction_conflict']:reasons.append('explicit_order_direction_conflict')
    return sorted(set(reasons))

def signal(sql,obs):
    x=dict(zip(NAMES,features(sql,obs)))
    return 4*x['public_match']+x['header_role_overlap']+x['question_output_overlap']-x['direction_conflict']-x['unknown_columns']-x['ambiguous_columns']

def proof(parent,candidate,obs):
    fb=feedback(candidate,obs)
    if not fb or not fb.get('matches_example'):return False,'candidate_not_observed_public_pass'
    if not eligible(candidate,obs):return False,'legacy_parse_width_named_value_guard'
    reasons=diagnose(parent,obs)
    if not reasons:return False,'no_concrete_parent_contradiction'
    ok,guard=certificate(parent,candidate,obs)
    a,b=parse(parent),parse(candidate)
    if ok and 'explicit_order_direction_conflict' in reasons and not dict(zip(NAMES,features(candidate,obs)))['direction_conflict'] and a.selects==b.selects and null_policy_preserved(parent,candidate):
        return True,'explicit_direction_same_order_object'
    schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
    if schema is not None:
        pool,_=local_candidates(parent,obs,schema)
        # This is scoped leaf-address verification, not unrestricted SQL equivalence.
        target=semantic_column(candidate,schema)
        literal_signature=lambda sql:[(v.is_string,v.this) for v in parse(sql).find_all(exp.Literal)]
        if target and any(c['source']=='unknown_projection_contract' and literal_signature(c['sql'])==literal_signature(candidate) and semantic_column(c['sql'],schema)==target for c in pool):
            return True,'unknown_projection_exact_local_contract'
    if ok and any(r in reasons for r in ('actual_execution_failure','actual_public_mismatch')) and explicit_role(parent,candidate,obs):
        return True,'failed_parent_same_scope_same_type_local_candidate'
    return False,'unproved_local_context_or_semantics:'+guard

def select(parent,obs,mode='review'):
    sqls=[];seen={key(parent)}
    for e in obs['history']:
        if e['tool']=='check' and e['observation'].get('matches_example') and key(e['sql']) not in seen:
            seen.add(key(e['sql']));sqls.append(e['sql'])
    if mode=='public':
        fb=feedback(parent,obs)
        return parent if fb and fb.get('matches_example') else next((s for s in reversed(sqls) if eligible(s,obs)),parent)
    current=signal(parent,obs);best=parent
    for sql in sqls:
        if signal(sql,obs)<=current:continue
        if mode=='signal' and (not eligible(sql,obs) or certificate(parent,sql,obs)[0] is not True):continue
        if mode=='signal' or proof(parent,sql,obs)[0]:current=signal(sql,obs);best=sql
    return best
