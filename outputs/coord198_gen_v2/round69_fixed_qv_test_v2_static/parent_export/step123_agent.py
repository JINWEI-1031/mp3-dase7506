"""E1 schema/alias linking and E2 entity-grain/negation plans over E0."""
import os
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step4_agent import identity,passed
from step122_agent import create_agent as baseline


def words(value):return re.findall(r'[a-z]+',re.sub(r'([a-z])([A-Z])',r'\1 \2',value).lower())


def singular(word):
    if word.endswith('ies'):return word[:-3]+'y'
    if word.endswith('s') and not word.endswith('ss'):return word[:-1]
    return word


def mentions(q,name):
    tokens=[singular(w) for w in words(name)]
    return bool(tokens) and all(t in {singular(w) for w in words(q)} for t in tokens)


def context(tree,schema):
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return None
    known={t['name'].lower():t for t in schema};sources={}
    for t in tree.find_all(exp.Table):
        if t.name.lower() not in known:return None
        sources[t.alias_or_name.lower()]=known[t.name.lower()]
    def owner(col):
        if not isinstance(col,exp.Column):return None
        if col.table:return (col.table,sources[col.table.lower()]) if col.table.lower() in sources else None
        choices=[(a,t) for a,t in sources.items() if any(c['name'].lower()==col.name.lower() for c in t['columns'])]
        return choices[0] if len(choices)==1 else None
    return sources,owner


def entity_plans(obs,sql,schema):
    tree=parse(sql);ctx=context(tree,schema)
    if ctx is None or any(tree.args.get(k) for k in ('group','having','order','limit','distinct')):return [],None
    sources,owner=ctx;q=obs['task']['question'];out=[];probe=None
    # Rewrite only the mistaken predicate column, preserving every join and other predicate.
    projected=[owner(e.unalias()) for e in tree.expressions]
    target=projected[0] if projected and all(x is not None and x[0].lower()==projected[0][0].lower() for x in projected) else None
    for eq in tree.find_all(exp.EQ):
        if not isinstance(eq.this,exp.Column) or not isinstance(eq.expression,exp.Literal) or not eq.expression.is_string:continue
        location=owner(eq.this)
        if location is None:continue
        alias,table=location;literal=eq.expression.this
        if not re.search(r'\b'+re.escape(literal.lower())+r'\b',q.lower()):continue
        # Split a full person name only when both name fields are unambiguous.
        first=[c['name'] for c in table['columns'] if c['name'].lower() in ('fname','first_name','firstname')]
        last=[c['name'] for c in table['columns'] if c['name'].lower() in ('lname','last_name','lastname')]
        parts=literal.split()
        if len(first)==len(last)==1 and len(parts)==2 and eq.this.name.lower() in {first[0].lower(),last[0].lower()}:
            new=tree.copy();matching=next(x for x in new.find_all(exp.EQ) if sql_text(x)==sql_text(eq))
            matching.replace(exp.and_(exp.EQ(this=exp.column(first[0],table=alias),expression=exp.Literal.string(parts[0])),exp.EQ(this=exp.column(last[0],table=alias),expression=exp.Literal.string(parts[1]))))
            out.append(sql_text(new))
        if target and mentions(q,target[1]['name']) and location[0].lower()!=target[0].lower():
            names=[c['name'] for c in target[1]['columns'] if 'name' in words(c['name']) or c['name'].lower().endswith('name')]
            # Require explicit entity-name adjacency or 'named/called' wording, rather than any noun overlap.
            noun=' '.join(words(target[1]['name']))
            if len(names)==1 and re.search(r'\b'+re.escape(noun)+r'\s+(?:(?:named|called)\s+)?'+re.escape(literal.lower())+r'\b',q.lower()):
                new=tree.copy();matching=next(x for x in new.find_all(exp.EQ) if sql_text(x)==sql_text(eq))
                matching.set('this',exp.column(names[0],table=target[0]));out.append(sql_text(new))
        # Category values are grounded by a charged tool call, with no Team-specific column.
        fieldtokens=[w for w in words(eq.this.name) if w not in ('name','code','id')]
        if fieldtokens and all(w in words(q) for w in fieldtokens) and re.search(r'\b'+re.escape(' '.join(fieldtokens))+r'\s+'+re.escape(literal.lower())+r'\b',q.lower()):
            query=exp.select(exp.column(eq.this.name,quoted=True)).distinct().from_(exp.to_table(table['name'])).where(exp.Like(this=exp.column(eq.this.name,quoted=True),expression=exp.Literal.string('%'+literal.replace('%','').replace('_','')+'%'))).limit(9)
            probe=(sql_text(query),sql_text(eq),sql)
    return list(dict.fromkeys(out)),probe


def grain_plans(obs,sql,schema):
    tree=parse(sql);ctx=context(tree,schema);q=obs['task']['question'].lower()
    if ctx is None or len(tree.expressions)!=1 or not isinstance(tree.expressions[0].unalias(),exp.Count) or any(tree.args.get(k) for k in ('group','having','limit','offset')):return []
    phrase=re.search(r'\b(?:how many|number of)\s+([a-z_]+)',q)
    if not phrase:return []
    targets=[t for t in schema if singular(phrase[1]) in {singular(w) for w in words(t['name'])}]
    if len(targets)!=1:return []
    parent=targets[0];sources,owner=ctx;negative=bool(re.search(r'\b(?:do not have|does not have|have no|has no|without)\b',q));out=[]
    if negative:
        comparator=re.search(r'\b(more than|greater than|above|over|longer than|older than|higher than|less than|below|under|shorter than|younger than|lower than)\s+(\d+(?:\.\d+)?)',q)
        where=tree.args.get('where');pred=where.this if where else None
        if not comparator or not isinstance(pred,(exp.GT,exp.GTE,exp.LT,exp.LTE)) or not isinstance(pred.this,exp.Column) or not isinstance(pred.expression,exp.Literal) or pred.expression.this!=comparator[2]:return []
        link=owner(pred.this)
        if link is None:return []
        child=link[1];fks=[fk for fk in child['foreign_keys'] if fk['table'].lower()==parent['name'].lower()]
        if len(fks)!=1:return []
        # Do not discard an unrelated third entity or additional joins/predicates.
        if any(t['name'].lower() not in {child['name'].lower(),parent['name'].lower()} for t in sources.values()):return []
        fk=fks[0];op=exp.LT if comparator[1] in ('less than','below','under','shorter than','younger than','lower than') else exp.GT
        qualify=op(this=exp.column(pred.this.name,table='c'),expression=pred.expression.copy())
        sub=exp.select('1').from_(exp.to_table(child['name']).as_('c')).where(exp.and_(exp.EQ(this=exp.column(fk['from'],table='c'),expression=exp.column(fk['to'],table='p')),qualify))
        new=exp.select(exp.Count(this=exp.Star())).from_(exp.to_table(parent['name']).as_('p')).where(exp.Not(this=exp.Exists(this=sub)))
        out.append(sql_text(new))
    else:
        aliases=[a for a,t in sources.items() if t['name'].lower()==parent['name'].lower()]
        keys=[c['name'] for c in parent['columns'] if c.get('pk')]
        count=tree.expressions[0].unalias()
        if len(aliases)==len(keys)==1 and tree.args.get('joins') and isinstance(count.this,exp.Star) and not re.search(r'\b(?:each|per|every)\b',q):
            # Counting parents across joins needs a parent key, not joined-row cardinality.
            new=tree.copy();new.set('expressions',[exp.Count(this=exp.Distinct(expressions=[exp.column(keys[0],table=aliases[0])]))]);out.append(sql_text(new))
    return out


class Agent:
    def __init__(self,checkpoint=None):
        self.base=baseline(checkpoint);self.arms=set(os.environ.get('MP3_STEP123_ARMS','e1,e2').split(','))
        self.fallback=None;self.pending=None;self.probe=None;self.tried=set()

    def act(self,obs):
        history=obs['history'];remaining=obs['remaining']
        if self.pending:
            if passed(history[-1]['observation']):return {'tool':'submit','sql':self.pending}
            self.pending=None
        if self.probe:
            query,comparison,original=self.probe;self.probe=None;sample=history[-1]['observation']
            values=[r[0] for r in sample.get('rows',[]) if len(r)==1 and isinstance(r[0],str) and re.search(r'\b'+re.escape(r[0].lower())+r'\b',obs['task']['question'].lower())]
            if sample.get('ok') and not sample.get('truncated') and len(values)==1 and remaining:
                tree=parse(original);eq=next((e for e in tree.find_all(exp.EQ) if sql_text(e)==comparison),None)
                if eq is not None:
                    eq.set('expression',exp.Literal.string(values[0]));sql=sql_text(tree)
                    if identity(sql) not in {identity(e['sql']) for e in history if e['tool']=='check'}:self.pending=sql;return {'tool':'check','sql':sql}
        if self.fallback is None:
            action=self.base.act(obs)
            if action['tool']!='submit':return action
            self.fallback=action
        tree=parse(self.fallback['sql']);q=obs['task']['question'].lower()
        if not remaining or not isinstance(tree,exp.Select):return self.fallback
        e1='e1' in self.arms and len(list(tree.find_all(exp.Select)))==1 and any(v.is_string for v in tree.find_all(exp.Literal)) and bool(re.search(r'\b(?:named|called)\b',q) or any(t.alias for t in tree.find_all(exp.Table)) or any(' ' in v.this for v in tree.find_all(exp.Literal) if v.is_string))
        e2='e2' in self.arms and any(isinstance(e.unalias(),exp.Count) for e in tree.expressions) and bool(re.search(r'\b(?:how many|number of)\b',q))
        if not e1 and not e2:return self.fallback
        schema=next((e['observation'] for e in history if e['tool']=='schema'),None)
        if schema is None:return {'tool':'schema'} if remaining>=2 else self.fallback
        candidates=[];probe=None
        if e1:
            sqls,probe=entity_plans(obs,self.fallback['sql'],schema);candidates += [('e1',s) for s in sqls]
        if e2:candidates += [('e2',s) for s in grain_plans(obs,self.fallback['sql'],schema)]
        seen={identity(e['sql']) for e in history if e['tool']=='check'}
        for kind,sql in candidates:
            if kind not in self.tried and identity(sql) not in seen and sum(k==kind for k,s in candidates)==1:
                self.tried.add(kind);self.pending=sql;return {'tool':'check','sql':sql}
        if probe and 'probe' not in self.tried and remaining>=2:
            self.tried.add('probe');self.probe=probe;return {'tool':'execute','sql':probe[0]}
        return self.fallback


def create_agent(checkpoint=None):return Agent(checkpoint)
