"""Schema-constrained minimal candidate synthesis; no task/table-specific answers."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step123_agent import words,singular


def propose(sql,question,schema,risks):
    # This provider deliberately handles a single query scope, not correlated rewrites.
    tree=parse(sql)
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return None
    if any(j.args.get('side') for j in tree.args.get('joins',[])):return None
    known={t['name'].lower():t for t in schema};sources={}
    for t in tree.find_all(exp.Table):
        if t.name.lower() not in known:return None
        sources[t.alias_or_name.lower()]=known[t.name.lower()]
    def bind(col):
        if not isinstance(col,exp.Column):return None
        choices=[(a,t,c) for a,t in sources.items() for c in t['columns'] if c['name'].lower()==col.name.lower() and (not col.table or col.table.lower()==a)]
        return choices[0] if len(choices)==1 else None
    new=tree.copy();changed=False
    if risks & {'identifier_joined_to_descriptive_text','join_conflicts_with_unique_declared_fk'}:
        for join in new.args.get('joins',[]):
            on=join.args.get('on')
            if on is None:continue
            for eq in on.find_all(exp.EQ):
                a,b=bind(eq.this),bind(eq.expression)
                if not a or not b or a[0]==b[0]:continue
                links=[]
                for left,right in ((a,b),(b,a)):
                    links += [(left[0],fk['from'],right[0],fk['to']) for fk in left[1]['foreign_keys'] if fk['table'].lower()==right[1]['name'].lower()]
                if len(links)!=1:continue
                x,xf,y,yf=links[0]
                actual={(a[0],a[2]['name'].lower(),b[0],b[2]['name'].lower()),(b[0],b[2]['name'].lower(),a[0],a[2]['name'].lower())}
                if (x,xf.lower(),y,yf.lower()) not in actual:
                    eq.set('this',exp.column(xf,table=x));eq.set('expression',exp.column(yf,table=y));changed=True
    if 'alphabetical_order_uses_identifier' in risks:
        order=new.args.get('order')
        if order and len(order.expressions)==1:
            item=order.expressions[0];owner=bind(item.this)
            if owner and not new.args.get('group'):
                alias,table,column=owner
                fks=[f for f in table['foreign_keys'] if f['from'].lower()==column['name'].lower()]
                if len(fks)==1 and fks[0]['table'].lower() in known:
                    fk=fks[0];target=known[fk['table'].lower()]
                    key=next((c for c in target['columns'] if c['name'].lower()==fk['to'].lower()),None)
                    texts=[c for c in target['columns'] if re.search(r'CHAR|TEXT|CLOB',c.get('type',''),re.I) and not c.get('pk')]
                    hints={singular(w) for w in words(question)+words(column['name']) if w!='id'}
                    scores=[(sum(singular(w) in hints for w in words(c['name']))+int(bool(re.search(r'name|detail|description',c['name'],re.I))),c) for c in texts]
                    if key and key.get('pk') and sum(bool(c.get('pk')) for c in target['columns'])==1 and column.get('not_null') and scores:
                        best=max(s for s,c in scores);choices=[c for s,c in scores if s==best]
                        if len(choices)==1 and (best>0 or len(texts)==1):
                            lookup='review_lookup'
                            while lookup in sources:lookup+='x'
                            new.append('joins',exp.Join(this=exp.to_table(target['name']).as_(lookup),on=exp.EQ(this=exp.column(fk['from'],table=alias),expression=exp.column(fk['to'],table=lookup))))
                            item.set('this',exp.column(choices[0]['name'],table=lookup));changed=True
    return sql_text(new) if changed else None
