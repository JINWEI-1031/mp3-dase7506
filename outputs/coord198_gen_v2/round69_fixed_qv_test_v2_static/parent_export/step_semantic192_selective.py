"""Integrate only supported semantic edits and abstain on ambiguous relation roles."""
import re
from sqlglot import exp
from repair_rules import parse
from step_semantic192_rules import tokens
from step_semantic192_scoped import Agent as Scoped,generate as scoped_generate,overlap
from step124_pool_search import sources

def temporal_order(tree,schema):
    if not isinstance(tree,exp.Select) or not tree.args.get('order'):return False
    src=sources(tree,schema)
    for col in tree.args['order'].find_all(exp.Column):
        for alias,table in src.items():
            if col.table and col.table.lower()!=alias.lower():continue
            for field in table['columns']:
                if field['name'].lower()!=col.name.lower() or field.get('pk') or tokens(field['name'])&{'id','code'}:continue
                typed=bool(re.search(r'DATE|TIME',field['type'],re.I))
                foreign=any(f['from'].lower()==field['name'].lower() for f in table['foreign_keys'])
                named=bool(tokens(field['name'])&{'date','time','year','birth','birthday','born','opened','founded','created','released','launched','updated'})
                if typed or named and not foreign:return True
    return False

def generate(seeds,obs,schema,families):
    out=[];known={t['name'].lower():t for t in schema};q=tokens(obs['task']['question']);current=parse(seeds[0]) if seeds else None
    for item in scoped_generate(seeds,obs,schema,families-{'count'}):
        if item['path'][0]=='relation_minimal':continue
        tree=parse(item['sql'])
        if item['path'][0].startswith('event_') and (not temporal_order(tree,schema) or not re.search(r'earliest|latest|recent|oldest|newest|\bdate\b|\btime\b|\byear\b',obs['task']['question'],re.I)):continue
        required=set(re.findall(r'\b(?:than|least|most|over|under|above|below)\s+(-?\d+(?:\.\d+)?)\b',obs['task']['question'],re.I))
        present={v.this for v in tree.find_all(exp.Literal)}
        if not required<=present:continue
        if item['path'][0]=='negative_relation' and isinstance(tree,exp.Select):
            root=tree.args.get('from');exists=tree.find(exp.Exists)
            inner=exists.this if exists else None
            source=inner.args.get('from') if isinstance(inner,exp.Select) else None
            parent=known.get(root.this.name.lower()) if root and isinstance(root.this,exp.Table) else None
            child=known.get(source.this.name.lower()) if source and isinstance(source.this,exp.Table) else None
            if parent and child:
                links=[f for f in child['foreign_keys'] if f['table'].lower()==parent['name'].lower()]
                if len(links)>1:
                    scores=[sum(overlap({w},q-tokens(parent['name'])-{'id','code'}) for w in tokens(f['from'])-{'id','code'}) for f in links]
                    if scores.count(max(scores))>1:continue
                covered=False
                if current is not None:
                    for neg in current.find_all(exp.Not):
                        if not isinstance(neg.this,exp.In):continue
                        sub=neg.this.args.get('query')
                        if isinstance(sub,exp.Subquery):sub=sub.this
                        if not isinstance(sub,exp.Select):continue
                        names={t.name.lower() for t in sub.find_all(exp.Table)}
                        fields={c.name.lower() for e in sub.expressions for c in e.find_all(exp.Column)}
                        if child['name'].lower() in names and any(f['from'].lower() in fields for f in links):covered=True
                if covered:continue
        out.append(item)
    return out
class Agent(Scoped):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint);self.semantic_families={'event','relation','negative'}
        import step_semantic192_agent as api
        api.generate=generate
        previous=api.ready
        if not getattr(previous,'typed_time',False):
            def ready(sql,obs,schema,families):
                result=previous(sql,obs,schema,families)
                if 'event' in families and not temporal_order(parse(sql),schema) and re.search(r'earliest|latest|recent|oldest|newest',obs['task']['question'],re.I):result.add('event')
                return result
            ready.typed_time=True;ready.fresh_output=True;ready.scoped_output=True;api.ready=ready
def create_agent(checkpoint=None):return Agent(checkpoint)
