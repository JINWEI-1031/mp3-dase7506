"""Question-grounded edits for residual generation; no task/database identifiers."""
import re
from collections import deque
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_rolebind192_terminal_rules import generate as parent_generate
from step_rolebind192_rules import role_words
from step_semantic192_rules import typed_literals
from step_pool192_rules import schema_errors
from step4_agent import identity
from step_rolebind192_rules import signals as previous_signals

def signals(sql,obs,schema=()):
    result=previous_signals(sql,obs,schema);tree=parse(sql)
    if isinstance(tree,exp.Select) and len(tree.expressions)==len(obs['example']['columns']) and any(e.alias_or_name.lower()!=h.lower() for e,h in zip(tree.expressions,obs['example']['columns'])):result.append('output_contract')
    return result

def item(tree,kind,evidence,confidence=8):
    return {'sql':sql_text(tree),'family':'evidence28','path':[kind],'evidence':evidence,'confidence':confidence,'source':'evidence28_192'}

def includes_zero(question):
    q=question.lower()
    if re.search(r'\b(?:who|that|which)\s+(?:have|has|had)|\bwith\s+(?:some|at least|a |an )',q):return False
    return bool(re.search(r'\b(?:for each|per|every|each)\b',q) or re.search(r'\ball\b.+\b(?:number|count|times|how many)\b',q))

def zero_counts(tree,obs,schema):
    if not includes_zero(obs['task']['question']) or len(list(tree.find_all(exp.Select)))!=1 or not tree.args.get('group'):return []
    root=tree.args.get('from');joins=tree.args.get('joins',[]);src=sources(tree,schema)
    if not root or len(joins)!=1 or not isinstance(root.this,exp.Table) or not isinstance(joins[0].this,exp.Table):return []
    ra=root.this.alias_or_name;ca=joins[0].this.alias_or_name
    if ra not in src or ca not in src:return []
    if not role_words(src[ca]['name'])<=role_words(obs['task']['question']):return []
    if joins[0].args.get('side') not in (None,'','LEFT'):return []
    if not tree.find(exp.Count) or not all(not c.table or c.table.lower()==ra.lower() for e in tree.expressions if not e.find(exp.Count) for c in e.find_all(exp.Column)):return []
    if tree.args.get('where') and any(c.table and c.table.lower()!=ra.lower() for c in tree.args['where'].find_all(exp.Column)):return []
    on=joins[0].args.get('on')
    if not isinstance(on,exp.EQ):return []
    child_cols=[c for c in (on.this,on.expression) if isinstance(c,exp.Column) and c.table.lower()==ca.lower()]
    if len(child_cols)!=1:return []
    child_col=child_cols[0]
    links=[f for f in src[ca]['foreign_keys'] if f['table'].lower()==src[ra]['name'].lower() and f['from'].lower()==child_col.name.lower()]
    if len(links)!=1:return []
    t=tree.copy();t.args['joins'][0].set('side','LEFT')
    for count in t.find_all(exp.Count):
        if isinstance(count.this,exp.Star):count.set('this',child_col.copy())
    return [item(t,'zero_event_left',{'distributive_question':True,'declared_fk':links[0],'count_nullable_child':True},11)]

def quantified_scope(tree,obs,schema):
    q=obs['task']['question'];request=re.search(r'\b(?:than|before|after)\s+(every|all|any)\s+(?:the\s+)?([a-z][a-z -]*?)[?.]?$',q,re.I)
    if not request:return []
    role=request[2].strip()
    if len(role.split())>3:return []
    out=[]
    for cmp in tree.find_all((exp.GT,exp.GTE,exp.LT,exp.LTE)):
        sub=cmp.expression
        if not isinstance(sub,exp.Subquery) or not isinstance(sub.this,exp.Select):continue
        inner=sub.this;src=sources(inner,schema)
        if len(src)!=1 or len(inner.expressions)!=1 or not isinstance(inner.expressions[0],(exp.Max,exp.Min)):continue
        if role_words(next(iter(src.values()))['name'])<=role_words(role):continue
        fields=[(a,c['name']) for a,t in src.items() for c in t['columns'] if re.search(r'CHAR|TEXT|CLOB',c['type'],re.I) and role_words(c['name'])&{'job','occupation','profession','role'}]
        if len(fields)!=1:continue
        if any(v.is_string and v.this.lower()==role.lower() for v in inner.find_all(exp.Literal)):continue
        a,field=fields[0];new=tree.copy();target=list(new.find_all(type(cmp)))[list(tree.find_all(type(cmp))).index(cmp)].expression.this
        gt=isinstance(cmp,(exp.GT,exp.GTE));all_=request[1].lower()!='any';agg=exp.Max if gt==all_ else exp.Min
        target.set('expressions',[agg(this=inner.expressions[0].this.copy())]);predicate=exp.EQ(this=exp.Lower(this=exp.column(field,table=a)),expression=exp.Lower(this=exp.Literal.string(role)))
        target.where(predicate,append=True,copy=False)
        out.append(item(new,'quantified_role_scope',{'quantifier':request[1].lower(),'role_literal_from_question':role,'schema_role_field':field},10))
    return out

def graph_paths(schema,start,end):
    known={t['name'].lower():t for t in schema};edges={}
    def add(a,af,b,bf,inferred):
        edge=(a,af,b,bf,inferred)
        for x,y in ((a,b),(b,a)):
            values=edges.setdefault(x.lower(),[])
            if (y,edge) not in values:values.append((y,edge))
    for t in schema:
        for f in t['foreign_keys']:
            if f['table'].lower() in known:add(t['name'],f['from'],known[f['table'].lower()]['name'],f['to'],False)
        for c in t['columns']:
            if not re.search(r'.+(?:_id|Id|ID)$',c['name']) or any(f['from'].lower()==c['name'].lower() for f in t['foreign_keys']):continue
            targets=[u for u in schema if u['name']!=t['name'] and len([x for x in u['columns'] if x.get('pk')])==1 and any(x.get('pk') and x['name'].lower()==c['name'].lower() for x in u['columns'])]
            if len(targets)==1:add(t['name'],c['name'],targets[0]['name'],c['name'],True)
    pending=deque([(start,[],{start.lower()})]);paths=[]
    while pending:
        name,path,seen=pending.popleft()
        if name.lower()==end.lower():paths.append(path);continue
        if len(path)>=3:continue
        for nxt,edge in edges.get(name.lower(),[]):
            if nxt.lower() not in seen:pending.append((nxt,[*path,(nxt,edge)],seen|{nxt.lower()}))
    if not paths:return []
    minimum=min(map(len,paths));return [p for p in paths if len(p)==minimum]

def bridge(tree,obs,schema):
    if len(list(tree.find_all(exp.Select)))!=1 or len(tree.args.get('joins',[]))!=1:return []
    join=tree.args['joins'][0];root=tree.args.get('from')
    if not root or not isinstance(root.this,exp.Table) or not isinstance(join.this,exp.Table) or join.args.get('side'):return []
    # Only repair a structurally invalid current relation, not a valid alternative role.
    if not schema_errors(sql_text(tree),obs,schema):return []
    src=sources(tree,schema)
    if len(src)!=2:return []
    a,b=root.this,join.this;options=graph_paths(schema,a.name,b.name)
    if len(options)!=1 or len(options[0])<2:return []
    new=tree.copy();new.set('joins',[]);aliases={a.name.lower():a.alias_or_name,b.name.lower():b.alias_or_name}
    for nxt,(l,lf,r,rf,inferred) in options[0]:
        if nxt.lower() not in aliases:aliases[nxt.lower()]='e28_'+str(len(aliases))
        target=b.copy() if nxt.lower()==b.name.lower() else exp.to_table(nxt).as_(aliases[nxt.lower()])
        new.append('joins',exp.Join(this=target,on=exp.EQ(this=exp.column(lf,table=aliases[l.lower()]),expression=exp.column(rf,table=aliases[r.lower()]))))
    return [item(new,'evidenced_bridge',{'unique_shortest_path':options[0],'inferred_edges':sum(e[-1] for _,e in options[0])},10)]

def counted_entities(tree,obs,schema):
    q=obs['task']['question'];request=re.search(r'\b(?:how many|number of)\s+(?:distinct\s+)?([a-z]+)',q,re.I)
    if not request or tree.args.get('group') or len(tree.expressions)!=1 or not isinstance(tree.expressions[0].unalias(),exp.Count):return []
    target=[t for t in schema if role_words(t['name'])==role_words(request[1])]
    if len(target)!=1:return []
    src=sources(tree,schema);choices=[]
    for a,t in src.items():
        choices.extend((a,f['from']) for f in t['foreign_keys'] if f['table'].lower()==target[0]['name'].lower())
    if len(choices)!=1:return []
    a,field=choices[0];new=tree.copy();count=new.expressions[0].unalias();count.set('this',exp.Distinct(expressions=[exp.column(field,table=a)]))
    return [item(new,'counted_entity_fk',{'requested_entity':target[0]['name'],'counted_fk':[a,field],'distinct_entities':True},9)]

def projection(tree,obs,schema):
    src=sources(tree,schema);heads=obs['example']['columns'];old=tree.expressions
    if not src or len(heads)!=len(old):return []
    q=obs['task']['question'];chosen=[];used=set();evidence=[]
    for i,h in enumerate(heads):
        matches=[j for j,e in enumerate(old) if j not in used and (e.alias_or_name.lower()==h.lower() or h.lower().startswith('count(') and isinstance(e.unalias(),exp.Count))]
        if len(matches)==1:
            j=matches[0];value=old[j].copy();used.add(j)
            # A descriptive alias must not conceal a different key field.
            c=value.unalias()
            if isinstance(c,exp.Column) and c.name.lower()!=h.lower() and c.table in src and any(x['name'].lower()==h.lower() for x in src[c.table]['columns']):
                value=exp.column(h,table=c.table);evidence.append('alias_vs_field')
            chosen.append(value);continue
        if not re.fullmatch(r'\w+',h):return []
        owners=[(a,c['name']) for a,t in src.items() for c in t['columns'] if c['name'].lower()==h.lower()]
        current=old[i].unalias()
        if isinstance(current,exp.Count) and not re.search(r'\b(?:count|how many|number of)\b',q,re.I):
            root=tree.args.get('from');preferred=[x for x in owners if root and x[0]==root.this.alias_or_name]
            if len(preferred)==1:owners=preferred
        if len(owners)!=1:return []
        a,name=owners[0];chosen.append(exp.column(name,table=a));evidence.append('public_output_field')
    new=tree.copy();new.set('expressions',chosen)
    if not any(e.find(exp.AggFunc) for e in chosen) and any(e.find(exp.Count) for e in old) and re.search(r'\bdistinct\b',q,re.I):new.set('distinct',exp.Distinct())
    out=[item(new,'output_field_binding',{'public_headers':heads,'evidence':evidence},9)]
    # Same-named output fields can be bound by a unique co-projected entity.
    for i,e in enumerate(old):
        c=e.unalias()
        if not isinstance(c,exp.Column) or not re.search(r'name$',c.name,re.I):continue
        anchors={k.table for j,v in enumerate(old) if j!=i for k in v.find_all(exp.Column) if k.table in src}
        targets=[a for a in anchors if a!=c.table and any(x['name'].lower()==c.name.lower() for x in src[a]['columns']) and any(re.search(r'\bfor\s+(?:the\s+)?'+re.escape(word),q,re.I) for word in sorted(role_words(src[a]['name'])))]
        if len(targets)==1:
            variant=tree.copy();expressions=list(variant.expressions);expressions[i]=exp.column(c.name,table=targets[0]);variant.set('expressions',expressions);out.append(item(variant,'co_projected_entity_role',{'owner':src[targets[0]]['name'],'question_for_entity':True},10))
    return out

def generate(seeds,obs,schema,*,new=True,cap=24,mode='combined'):
    if not seeds:return []
    result=parent_generate(seeds[:1],obs,schema,cap=cap);tree=parse(seeds[0])
    if not isinstance(tree,exp.Select):return result
    if mode in ('semantic','combined'):
        for provider in (quantified_scope,bridge,counted_entities,projection):result+=provider(tree,obs,schema)
    if mode in ('zero','combined'):
        result+=zero_counts(tree,obs,schema)
        if includes_zero(obs['task']['question']):
            for c in result:
                if c['path'][0] in ('count_event_role','zero_event_left'):
                    t=parse(c['sql']);left=any(j.args.get('side')=='LEFT' for j in t.args.get('joins',[]));c['confidence']+=4 if left else -2;c['evidence']={**c.get('evidence',{}),'include_zero_entities':True}
    unique={};current=identity(seeds[0])
    for c in result:
        key=identity(c['sql'])
        if key==current or not typed_literals(c['sql'],schema) or schema_errors(c['sql'],obs,schema):continue
        if key not in unique or unique[key]['confidence']<c['confidence']:unique[key]=c
    return sorted(unique.values(),key=lambda c:-c['confidence'])[:cap]
