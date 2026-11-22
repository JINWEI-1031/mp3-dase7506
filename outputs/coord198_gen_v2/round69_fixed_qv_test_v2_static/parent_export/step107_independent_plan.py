"""Small independent FK plans from output ownership and question-backed names."""
import re
from collections import deque
from sqlglot import exp
from repair_rules import sql_text

def plan(obs,schema):
    q=obs['task']['question'];low=q.lower()
    if re.search(r'\b(?:not|without|except|either|or|most|least|first|last|average|total|count|many|female|male)\b|\d',low):return []
    heads=obs['example']['columns']
    if not heads or any(not re.fullmatch(r'\w+',h) for h in heads):return []
    roots=[t for t in schema if all(any(c['name'].lower()==h.lower() for c in t['columns']) for h in heads)]
    if len(roots)!=1:return []
    root=roots[0];known={t['name'].lower():t for t in schema};graph={}
    for t in schema:
        for fk in t['foreign_keys']:
            edge=(t['name'],fk['from'],fk['table'],fk['to'])
            graph.setdefault(t['name'].lower(),[]).append((fk['table'].lower(),edge))
            graph.setdefault(fk['table'].lower(),[]).append((t['name'].lower(),edge))
    def paths(target):
        queue=deque([(root['name'].lower(),[],{root['name'].lower()})]);out=[]
        while queue:
            name,p,seen=queue.popleft()
            if name==target:out.append(p);continue
            if len(p)>=2:continue
            for nxt,e in graph.get(name,[]):
                if nxt not in seen:queue.append((nxt,p+[e],seen|{nxt}))
        if not out:return []
        n=min(map(len,out));return [p for p in out if len(p)==n]
    values=[]
    for match in re.finditer(r"'((?:''|[^'])*)'",obs['issue_sql']):
        value=match[1].replace("''", "'")
        if len(value)>1 and value.lower() in low and value not in values:values.append(value)
    if not values:return []
    anchors=[]
    for value in values:
        choices=[]
        observed=[x.lower() for x in re.findall(r"(\w+)\s*=\s*'"+re.escape(value.replace("'","''"))+r"'",obs['issue_sql'],re.I)]
        explicit_root=bool(re.search(r'\b'+re.escape(' '.join(re.findall('[a-z]+',re.sub(r'([a-z])([A-Z])',r'\1 \2',root['name']).lower())))+r'\s+'+re.escape(value.lower())+r'\b',low))
        if not any(x.endswith('name') or x=='surname' for x in observed) and not explicit_root:return []
        for t in schema:
            if not any(x.endswith('name') or x=='surname' for x in observed) and t is not root:continue
            fields={c['name'].lower():c['name'] for c in t['columns']}
            first=[v for k,v in fields.items() if k in ('fname','firstname','first_name')]
            last=[v for k,v in fields.items() if k in ('lname','lastname','last_name','surname')]
            parts=value.split()
            if len(parts)==2 and len(first)==len(last)==1:
                bound=[(first[0],parts[0]),(last[0],parts[1])]
            else:
                names=[v for k,v in fields.items() if k.endswith('name')]
                if len(names)!=1:continue
                bound=[(names[0],value)]
            p=paths(t['name'].lower())
            if len(p)==1:
                evidence=0 if any(field.lower() in observed for field,_ in bound) else 1
                choices.append((evidence,len(p[0]),t,p[0],bound))
        if not choices:return []
        best=min((c[0],c[1]) for c in choices);choices=[c for c in choices if (c[0],c[1])==best]
        if len(choices)!=1:return []
        anchors.append(choices[0][1:])
    # Independent planning is confined to one entity or a two-anchor event.
    if len(anchors)>2 or len(anchors)==2 and anchors[0][1]['name']==anchors[1][1]['name']:return []
    aliases={root['name'].lower():'p0'};edges=[]
    for _,_,path,_ in anchors:
        for edge in path:
            if edge not in edges:edges.append(edge)
    query=exp.select(*[exp.column(h,table='p0',quoted=True) for h in heads]).from_(exp.to_table(root['name']).as_('p0'))
    for a,af,b,bf in edges:
        x,y=a.lower(),b.lower()
        if x in aliases and y in aliases:continue
        if x not in aliases and y not in aliases:return []
        nxt=y if x in aliases else x;aliases[nxt]='p'+str(len(aliases))
        query.append('joins',exp.Join(this=exp.to_table(known[nxt]['name']).as_(aliases[nxt]),on=exp.EQ(this=exp.column(af,table=aliases[x],quoted=True),expression=exp.column(bf,table=aliases[y],quoted=True))))
    conditions=[exp.EQ(this=exp.column(field,table=aliases[t['name'].lower()],quoted=True),expression=exp.Literal.string(value)) for _,t,_,fields in anchors for field,value in fields]
    query.where(exp.and_(*conditions),copy=False)
    return [{'sql':sql_text(query),'evidence':{'output_owner':root['name'],'name_anchors':len(anchors),'paths':edges},'kind':'independent_name_fk_plan'}]
