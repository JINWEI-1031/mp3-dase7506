"""Observable numeric evidence only; no identifier or answer feature encoding."""
import math
import re
from sqlglot import exp
from repair_rules import parse
from step107_slots_fields import words,output_focus
from step107_frontier_search import sources

NAMES=('parse_ok','public_observed','public_match','execution_observed','execution_ok','schema_observed',
 'output_width_error','header_role_overlap','question_output_overlap','named_value_keep','named_values_observed',
 'unknown_columns','ambiguous_columns','declared_join_fraction','join_evidence_observed','left_star_count',
 'root_pk_group','pk_group_observed','distinct_requested','distinct_present','negation_requested','not_present',
 'both_requested','exists_present','order_requested','order_present','direction_conflict','scope_count',
 'projection_count','join_count','filter_count','aggregate_count','where_edit','projection_edit','order_edit')

def key(sql):
    tree=parse(sql)
    return tree.sql(dialect='sqlite') if tree is not None else sql

def features(sql,obs):
    t=parse(sql);q=obs['task']['question'];ql=q.lower();history=obs.get('history',[])
    schema=next((e['observation'] for e in history if e['tool']=='schema'),None)
    feedback=next((e['observation'] for e in reversed(history) if e['tool'] in ('check','execute') and key(e['sql'])==key(sql)),None)
    v={n:0. for n in NAMES};v['parse_ok']=float(t is not None);v['schema_observed']=float(schema is not None)
    if feedback is not None:
        v.update(execution_observed=1.,execution_ok=float(feedback.get('ok',False)),public_observed=float('matches_example' in feedback),public_match=float(feedback.get('matches_example',False)))
    if t is None:return [v[n] for n in NAMES]
    selects=list(t.find_all(exp.Select));v['scope_count']=min(len(selects),4)/4
    v['projection_count']=min(len(t.selects),8)/8
    v['output_width_error']=float(len(t.selects)!=len(obs['example']['columns']))
    overlaps=[];qo=[]
    for e,h in zip(t.selects,obs['example']['columns']):
        cols=list(e.find_all(exp.Column));names=set().union(*(words(c.name) for c in cols)) if cols else set()
        hw=words(h);qw=words(output_focus(q))
        overlaps.append(len(names&hw)/max(1,len(hw)));qo.append(len(names&qw)/max(1,len(names)))
    v['header_role_overlap']=sum(overlaps)/max(1,len(overlaps));v['question_output_overlap']=sum(qo)/max(1,len(qo))
    original=parse(obs['issue_sql']);required=[]
    if original:
        required=[l.this.strip('%').lower() for l in original.find_all(exp.Literal) if l.is_string and len(l.this.strip('%'))>1 and l.this.strip('%').lower() in ql]
    actual=' '.join(l.this.lower() for l in t.find_all(exp.Literal) if l.is_string)
    v['named_values_observed']=float(bool(required));v['named_value_keep']=sum(x in actual for x in required)/max(1,len(required)) if required else 0.
    v['distinct_requested']=float(bool(re.search(r'\b(?:distinct|unique|different)\b',ql)));v['distinct_present']=float(bool(t.args.get('distinct')) or bool(t.find(exp.Distinct)))
    v['negation_requested']=float(bool(re.search(r'\b(?:not|without|neither|no)\b',ql)));v['not_present']=float(bool(t.find(exp.Not)))
    v['both_requested']=float('both' in words(q));v['exists_present']=float(bool(t.find(exp.Exists)))
    v['order_requested']=float(bool(re.search(r'\border|highest|lowest|largest|smallest|first|last\b',ql)));v['order_present']=float(bool(t.args.get('order')))
    if t.args.get('order'):
        directions=[bool(e.args.get('desc')) for e in t.args['order'].expressions]
        if re.search(r'\bdescending|decreasing\b',ql):v['direction_conflict']=float(not all(directions))
        elif re.search(r'\bascending|increasing\b',ql):v['direction_conflict']=float(any(directions))
    joins=list(t.find_all(exp.Join));v['join_count']=min(len(joins),6)/6
    v['filter_count']=min(sum(1 for x in t.find_all(exp.Predicate)),12)/12;v['aggregate_count']=min(sum(1 for x in t.find_all(exp.AggFunc)),6)/6
    if schema is not None:
        unknown=ambiguous=total=declared=0;group_proofs=[]
        for s in selects:
            src=sources(s,schema)
            for c in s.find_all(exp.Column):
                if c.find_ancestor(exp.Select) is not s:continue
                owners=[a for a,tb in src.items() if (not c.table or c.table.lower()==a.lower()) and any(f['name'].lower()==c.name.lower() for f in tb['columns'])]
                unknown+=not owners;ambiguous+=len(owners)>1;total+=1
            for j in s.args.get('joins',[]):
                on=j.args.get('on')
                if isinstance(on,exp.EQ) and all(isinstance(x,exp.Column) for x in (on.this,on.expression)):
                    a,b=on.this,on.expression;by={alias.lower():tb for alias,tb in src.items()}
                    if a.table.lower() in by and b.table.lower() in by:
                        def link(x,y):return any(f['from'].lower()==x.name.lower() and f['table'].lower()==by[y.table.lower()]['name'].lower() and f['to'].lower()==y.name.lower() for f in by[x.table.lower()].get('foreign_keys',[]))
                        declared+=link(a,b) or link(b,a)
                if j.args.get('side')=='LEFT' and any(isinstance(c.this,exp.Star) for c in s.find_all(exp.Count)):v['left_star_count']=1.
            root=s.args.get('from');group=s.args.get('group')
            if root and isinstance(root.this,exp.Table) and root.this.alias_or_name in src and group:
                alias=root.this.alias_or_name;pk=[c['name'].lower() for c in src[alias]['columns'] if c.get('pk')]
                group_proofs.append(bool(pk) and all(any(isinstance(c,exp.Column) and c.name.lower()==name and (not c.table or c.table.lower()==alias.lower()) for c in group.expressions) for name in pk))
        v['unknown_columns']=unknown/max(1,total);v['ambiguous_columns']=ambiguous/max(1,total)
        v['join_evidence_observed']=float(bool(joins));v['declared_join_fraction']=declared/max(1,len(joins))
        v['pk_group_observed']=float(bool(group_proofs));v['root_pk_group']=sum(group_proofs)/max(1,len(group_proofs))
    if original and isinstance(original,exp.Select) and isinstance(t,exp.Select):
        for clause,feature in (('where','where_edit'),('order','order_edit')):v[feature]=float(original.args.get(clause)!=t.args.get(clause))
        v['projection_edit']=float(original.selects!=t.selects)
    result=[float(v[n]) for n in NAMES];assert all(math.isfinite(x) for x in result)
    return result

def score(vector,model):return sum(x*w for x,w in zip(vector,model['weights']))
