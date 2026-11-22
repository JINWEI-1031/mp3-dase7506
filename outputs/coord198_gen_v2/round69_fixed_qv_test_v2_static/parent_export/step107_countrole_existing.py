"""Preserve zero-event parents when the evidenced event is already joined."""
from sqlglot import exp
from step107_countrole_portable import parse,sql_text,sources,role_words

def generate(sql,obs,schema):
    t=parse(sql)
    if not isinstance(t,exp.Select) or len(list(t.find_all(exp.Select)))!=1 or not t.args.get('group') or t.args.get('having') or t.args.get('distinct'):return []
    joins=t.args.get('joins',[]);root=t.args.get('from')
    if len(joins)!=1 or not root or not isinstance(root.this,exp.Table) or not isinstance(joins[0].this,exp.Table):return []
    j=joins[0]
    if j.args.get('side') or j.args.get('kind') not in (None,'','INNER'):return []
    src=sources(t,schema);pa=root.this.alias_or_name;ca=j.this.alias_or_name
    if pa not in src or ca not in src:return []
    parent,child=src[pa],src[ca];q=role_words(obs['task']['question'])
    if not role_words(child['name']) or not role_words(child['name'])<=q:return []
    links=[f for f in child['foreign_keys'] if f['table'].lower()==parent['name'].lower()]
    if len(links)!=1:return []
    fk=links[0];on=j.args.get('on')
    if not isinstance(on,exp.EQ) or not isinstance(on.this,exp.Column) or not isinstance(on.expression,exp.Column):return []
    if {(on.this.table.lower(),on.this.name.lower()),(on.expression.table.lower(),on.expression.name.lower())}!={(pa.lower(),fk['to'].lower()),(ca.lower(),fk['from'].lower())}:return []
    def root_column(c):
        if c.table:return c.table.lower()==pa.lower()
        owners=[a for a,x in src.items() if any(f['name'].lower()==c.name.lower() for f in x['columns'])]
        return owners==[pa]
    for part in ('where','group','order'):
        if t.args.get(part) and any(not root_column(c) for c in t.args[part].find_all(exp.Column)):return []
    counts=list(t.find_all(exp.Count))
    if not counts:return []
    if any(not isinstance(c.this,(exp.Star,exp.Column)) or isinstance(c.this,exp.Column) and c.this.table.lower()!=ca.lower() for c in counts):return []
    for e in t.expressions:
        if not e.find(exp.Count) and any(not root_column(c) for c in e.find_all(exp.Column)):return []
    new=t.copy();new.args['joins'][0].set('side','LEFT')
    for c in new.find_all(exp.Count):c.set('this',exp.column(fk['from'],table=ca))
    return [{'sql':sql_text(new),'family':'count','path':['retain_zero_existing_event'],'confidence':10,'evidence':{'declared_fk':fk,'explicit_zero_parent_policy':True}}]
