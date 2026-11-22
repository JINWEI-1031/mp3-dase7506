"""Bounded compositional SQL proposals and label-free ranking from public observations."""
import re
from collections import deque
from sqlglot import exp
from repair_rules import parse,sql_text
from step4_agent import identity
from step123_agent import words,singular
from step127_agent import flags
from step127_structured import propose


def materialized(controller):
    output=[];seen=set()
    def scan(value):
        if isinstance(value,dict):
            if isinstance(value.get('sql'),str):output.append(value['sql'])
            if isinstance(value.get('generated'),dict):output.extend(s for s in value['generated'] if isinstance(s,str) and re.match(r'\s*(SELECT|WITH)',s,re.I))
            for k,v in value.items():
                if k not in ('sql','generated','retrieved'):scan(v)
        elif isinstance(value,(list,tuple)):
            for v in value:scan(v)
    def visit(obj):
        if id(obj) in seen:return
        seen.add(id(obj))
        if hasattr(obj,'audit'):scan(obj.audit)
        for v in vars(obj).values():
            if hasattr(v,'__dict__') and type(v).__module__.startswith('step'):visit(v)
    visit(controller);return output


def sources(tree,schema):
    known={t['name'].lower():t for t in schema};result={}
    if not isinstance(tree,exp.Select):return result
    root=tree.args.get('from');ts=[root.this] if root and isinstance(root.this,exp.Table) else []
    ts += [j.this for j in tree.args.get('joins',[]) if isinstance(j.this,exp.Table)]
    for t in ts:
        if t.name.lower() in known:result[t.alias_or_name]=known[t.name.lower()]
    return result


def score(sql,obs,schema):
    t=parse(sql)
    if t is None:return -1000
    q=obs['task']['question'].lower();qw={singular(w) for w in words(q)};value=0.0
    selects=list(t.find_all(exp.Select))
    for s in selects:
        src=sources(s,schema)
        value-=.15*len(s.args.get('joins',[]))
        if s.args.get('distinct') and not re.search(r'\b(?:distinct|different|unique)\b',q):value-=1.2
        for alias,table in src.items():
            value+=.25*sum(singular(w) in qw for w in words(table['name']))
        group=s.args.get('group')
        if group:
            for alias,table in src.items():
                keys={c['name'].lower() for c in table['columns'] if c.get('pk')}
                cols={c.name.lower() for c in group.expressions if isinstance(c,exp.Column) and (not c.table or c.table.lower()==alias.lower())}
                if keys and keys<=cols:value-=1.0*len(cols-keys)
        where=s.args.get('where')
        if where:
            for literal in where.find_all(exp.Literal):
                if not literal.is_string and literal.this not in ('0',) and not re.search(r'\b'+re.escape(literal.this)+r'\b',q):value-=2.5
        for col in s.find_all(exp.Column):
            if not src:continue
            owners=[a for a,table in src.items() if any(c['name'].lower()==col.name.lower() for c in table['columns']) and (not col.table or a.lower()==col.table.lower())]
            if not owners and col.name.lower() not in {e.alias.lower() for e in s.expressions if e.alias}:value-=2.0
    if isinstance(t,exp.Select):
        order=t.args.get('order');limit=t.args.get('limit');top=re.search(r'\btop\s+(\d+)\b',q)
        first=bool(re.search(r'\bfirst\s+(?!name\b|and\b)[a-z]',q))
        expected=top[1] if top else '1' if first else None
        if expected:value+=4 if order and limit and limit.expression.name==expected else -4
        if re.search(r'\b(?:total|sum)\b',q):value+=2 if t.find(exp.Sum) else -2
        if re.search(r'\baverage\b',q):value+=2 if t.find(exp.Avg) else -2
        value-=2*len(flags(sql,obs['task']['question'],schema))
    return value


def normalize(sql,obs,schema):
    original=parse(sql)
    if original is None:return []
    outputs=[]
    for preference in (0,-1):
        t=original.copy()
        for select in t.find_all(exp.Select):
            src=sources(select,schema)
            if not src:continue
            heads=obs['example']['columns'];old=list(select.expressions)
            if len(old)!=len(heads):
                chosen=[]
                for h in heads:
                    matches=[e for e in old if e.alias_or_name.lower()==h.lower()]
                    if 'count(' in h.lower():matches=[e for e in old if isinstance(e.unalias(),exp.Count)] or [exp.Count(this=exp.Star())]
                    if len(matches)==1:chosen.append(matches[0].copy());continue
                    owners=[(a,c['name']) for a,table in src.items() for c in table['columns'] if c['name'].lower()==h.lower()]
                    if owners:
                        a,name=owners[preference];chosen.append(exp.column(name,table=a,quoted=True))
                    else:chosen=[];break
                if chosen:select.set('expressions',chosen)
            for col in select.find_all(exp.Column):
                if col.table:continue
                owners=[(a,c['name']) for a,table in src.items() for c in table['columns'] if c['name'].lower()==col.name.lower()]
                if owners:
                    a,name=owners[preference];col.set('table',exp.to_identifier(a,quoted=True));col.set('this',exp.to_identifier(name,quoted=True))
        outputs.append(sql_text(t))
    return list(dict.fromkeys(outputs))


def atomics(sql,obs,schema):
    t=parse(sql)
    if t is None:return []
    out=[('shape',s) for s in normalize(sql,obs,schema)]
    if not isinstance(t,exp.Select) or len(list(t.find_all(exp.Select)))!=1:return out
    src=sources(t,schema);q=obs['task']['question'].lower()
    fixed=propose(sql,obs['task']['question'],schema,flags(sql,obs['task']['question'],schema))
    if fixed:out.append(('schema_link',fixed))
    # Joint qualification/projection followed by case-normalization is a two-edit path.
    mismatch=obs['example']['row_count']>0 and any(e['tool']=='check' and e['observation']['ok'] and e['observation']['row_count']==0 for e in obs['history'])
    if mismatch:
        new=t.copy();changed=False
        for eq in list(new.find_all(exp.EQ)):
            if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Literal) and eq.expression.is_string:
                eq.set('this',exp.Lower(this=eq.this.copy()));eq.set('expression',exp.Lower(this=eq.expression.copy()));changed=True
        if changed:out.append(('casefold',sql_text(new)))
    if t.args.get('group'):
        new=t.copy();group=new.args['group'];remove=set()
        for alias,table in src.items():
            keys={c['name'].lower() for c in table['columns'] if c.get('pk')}
            cols=[c for c in group.expressions if isinstance(c,exp.Column) and (not c.table or c.table.lower()==alias.lower())]
            if keys and keys<={c.name.lower() for c in cols}:
                remove|={sql_text(c) for c in cols if c.name.lower() not in keys}
        if remove:
            group.set('expressions',[c for c in group.expressions if sql_text(c) not in remove]);out.append(('group_key',sql_text(new)))
    # Missing association table: keep original table aliases and all filters/grouping.
    tables=list(t.find_all(exp.Table));joins=t.args.get('joins',[])
    if len(tables)==2 and len(joins)==1 and not joins[0].args.get('side'):
        graph={}
        for table in schema:
            for fk in table['foreign_keys']:
                a,b=table['name'],fk['table'];edge=(a,fk['from'],b,fk['to'])
                graph.setdefault(a.lower(),[]).append((b,edge));graph.setdefault(b.lower(),[]).append((a,edge))
        start,end=tables[0].name,tables[1].name;paths=[];queue=deque([(start,[],{start.lower()})])
        while queue:
            name,path,seen=queue.popleft()
            if name.lower()==end.lower():paths.append(path);continue
            if len(path)==2:continue
            for nxt,edge in graph.get(name.lower(),[]):
                if nxt.lower() not in seen:queue.append((nxt,[*path,(nxt,edge)],seen|{nxt.lower()}))
        if paths and min(map(len,paths))==2 and len(paths)==1:
            new=t.copy();new.set('joins',[]);aliases={tables[0].name.lower():tables[0].alias_or_name,tables[1].name.lower():tables[1].alias_or_name}
            for nxt,(a,af,b,bf) in paths[0]:
                if nxt.lower() not in aliases:aliases[nxt.lower()]='search_mid'
                target=tables[1].copy() if nxt.lower()==end.lower() else exp.to_table(nxt).as_(aliases[nxt.lower()])
                new.append('joins',exp.Join(this=target,on=exp.EQ(this=exp.column(af,table=aliases[a.lower()]),expression=exp.column(bf,table=aliases[b.lower()]))))
            out.append(('bridge',sql_text(new)))
    # Count the entity explicitly named after 'how many', including an omitted child table.
    request=re.search(r'\b(?:how many|number of)\s+(.+?)(?:\s+(?:are|is|were|related|for|per|each)\b|[?.]|$)',q)
    if request and t.find(exp.Count) and t.args.get('group'):
        terms={singular(w) for w in words(request[1])};targets=[table for table in schema if words(table['name']) and all(singular(w) in terms for w in words(table['name']))]
        if len(targets)==1 and targets[0]['name'].lower() not in {x['name'].lower() for x in src.values()}:
            target=targets[0];keys=[c['name'] for c in target['columns'] if c.get('pk')];links=[(a,fk) for a,table in src.items() for fk in target['foreign_keys'] if fk['table'].lower()==table['name'].lower()]
            if len(keys)==len(links)==1:
                alias,fk=links[0]
                for side in ('','LEFT'):
                    new=t.copy();new.append('joins',exp.Join(this=exp.to_table(target['name']).as_('counted_entity'),side=side,on=exp.EQ(this=exp.column(fk['to'],table=alias),expression=exp.column(fk['from'],table='counted_entity'))))
                    for count in new.find_all(exp.Count):count.set('this',exp.column(keys[0],table='counted_entity'))
                    out.append(('count_entity_'+(side or 'INNER'),sql_text(new)))
    return out


def expand(seeds,obs,schema,depth=2,cap=48):
    result={}
    def add(sql,path):
        if not isinstance(sql,str) or not sql.strip() or len(sql.encode())>16000:return
        key=identity(sql)
        if key not in result:result[key]={'sql':sql,'path':path,'score':score(sql,obs,schema)}
    for sql in seeds:add(sql,[])
    beam=sorted(result.values(),key=lambda r:r['score'],reverse=True)[:6]
    for level in range(depth):
        for item in beam:
            for kind,sql in atomics(item['sql'],obs,schema):add(sql,[*item['path'],kind])
        beam=sorted(result.values(),key=lambda r:r['score'],reverse=True)[:6]
        if len(result)>=cap:break
    return sorted(result.values(),key=lambda r:r['score'],reverse=True)[:cap]
