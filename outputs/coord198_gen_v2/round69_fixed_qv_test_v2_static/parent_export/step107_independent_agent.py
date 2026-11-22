import re
from sqlglot import exp
from repair_rules import parse
from step4_agent import passed,identity
from step107_frontier_selected import create_agent as baseline
from step107_independent_plan import plan
from step107_frontier_agent import eligible

ARMS=('baseline','trigger','plan','gated','combined')

def contradictions(sql,obs):
    tree=parse(sql);out=[]
    if tree is None:return ['invalid_sql']
    q=obs['task']['question'].lower()
    actual=' '.join(x.this.lower() for x in tree.find_all(exp.Literal) if x.is_string)
    for m in re.finditer(r"'((?:''|[^'])*)'",obs['issue_sql']):
        value=m[1].replace("''", "'").lower()
        if len(value)>1 and value in q and not all(w in actual for w in value.split()):out.append('missing_question_named_value')
    for eq in tree.find_all(exp.EQ):
        if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Literal) and eq.expression.is_string:
            v=eq.expression.this
            if len(v.split())==2 and v.lower() in q and eq.this.name.lower() in ('lname','last_name','lastname','surname'):
                out.append('full_name_bound_to_surname')
    if len(tree.selects)!=len(obs['example']['columns']):out.append('output_width')
    return sorted(set(out))

class Agent:
    def __init__(self,checkpoint=None,*,arm='gated'):
        self.base=baseline(checkpoint);self.arm=arm;self.fallback=None;self.pending=None;self.done=False;self.audit=[];self.pool=[]
    def act(self,obs):
        if self.arm=='baseline':return self.base.act(obs)
        if self.pending:
            self.done=True
            return {'tool':'submit','sql':self.pending} if passed(obs['history'][-1]['observation']) else self.fallback
        if self.fallback is None:
            a=self.base.act(obs)
            if a['tool']!='submit':return a
            self.fallback=a
        if self.done or not obs['remaining']:return self.fallback
        evidence=contradictions(self.fallback['sql'],obs)
        if self.arm=='combined':
            from step107_chainnext_entity_rules import entity_risk
            evidence+=sorted(entity_risk(self.fallback['sql'],obs))
        if self.arm in ('gated','trigger','combined') and not evidence:return self.fallback
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if schema is None:return {'tool':'schema'} if obs['remaining']>=2 else self.fallback
        self.pool=plan(obs,schema);self.audit.append({'trigger':evidence,'generated':len(self.pool)})
        if self.arm=='trigger':self.done=True;return self.fallback
        seen={identity(e['sql']) for e in obs['history'] if e['tool']=='check'}
        for item in self.pool:
            if identity(item['sql']) in seen or not eligible(item['sql'],obs):continue
            self.pending=item['sql'];self.audit.append(item);return {'tool':'check','sql':self.pending}
        self.done=True;return self.fallback

def create_agent(checkpoint=None):return Agent(checkpoint)
