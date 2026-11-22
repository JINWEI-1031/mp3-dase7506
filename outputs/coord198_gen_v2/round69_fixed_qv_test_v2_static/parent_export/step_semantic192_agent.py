"""Novel semantic candidates over the causal recovery branch of the 192 parent."""
import os,re
from sqlglot import exp
from repair_rules import parse
from step_recovery192_balanced import Agent as Recovery
from step_recovery192 import signature
from step_semantic192_rules import generate,tokens
from step_pool192_rules import score as old_score,schema_errors
from step4_agent import identity,passed
from mp3_sql.sqlite_tool import validate_sql

def ready(sql,obs,schema,families):
    t=parse(sql)
    if not isinstance(t,exp.Select):return set()
    allowed=set(families);q=obs['task']['question'].lower()
    if 'event' in allowed:
        order=t.args.get('order')
        if order and any(re.search(r'date|time|year|opened|release|launch',c.name,re.I) for c in order.find_all(exp.Column)):allowed.discard('event')
    if any(isinstance(n.parent,exp.Not) for n in t.find_all(exp.Exists)):allowed.discard('negative')
    return allowed

def semantic_score(item,current,obs,schema):
    value=old_score(item['sql'],obs,schema)-4*schema_errors(item['sql'],obs,schema)
    q=tokens(obs['task']['question']);t=parse(item['sql'])
    if item['path'][0]=='measure_field':value+=4
    elif item['path'][0]=='relation_semijoin':value+=3
    elif item['family']=='event':
        order=t.args.get('order');value+=2+sum(len(tokens(c.name)&q) for c in order.find_all(exp.Column)) if order else 0
    elif item['family']=='negative':value+=3
    elif item['family']=='count':value+=1 if 'distinct' in item['path'][0] else .5
    elif item['family']=='relation':value+=1
    return value

class Agent(Recovery):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint);arm=os.environ.get('MP3_SEMANTIC192_ARM','combined')
        self.semantic_families={'event','relation','count','negative'} if arm=='combined' else {arm}
        self.sem_pending=None;self.sem_failed=False;self.sem_used=False;self.post=None;self.post_checked=False
    def choose(self,obs,schema,current,seeds,early=False):
        families=ready(current,obs,schema,self.semantic_families)
        self.semantic_pool=generate(seeds,obs,schema,families)
        checked={signature(e['sql'],schema) for e in obs['history'] if e['tool'] in ('check','execute')}
        for item in self.semantic_pool:item['score']=semantic_score(item,current,obs,schema)
        matched=any(e['tool']=='check' and passed(e['observation']) for e in obs['history'])
        for item in sorted(self.semantic_pool,key=lambda x:x['score'],reverse=True):
            if signature(item['sql'],schema) in checked or identity(item['sql'])==identity(current) or schema_errors(item['sql'],obs,schema):continue
            if matched and item['score']<old_score(current,obs,schema)+.5:continue
            try:validate_sql(item['sql'])
            except ValueError:continue
            self.audit.append({**item,'early':early,'remaining':obs['remaining']});self.sem_pending=item['sql'];self.sem_used=True
            return {'tool':'check','sql':item['sql']}
        return None
    def candidate(self,obs,schema,seeds,early=False):
        if early and self.sem_failed:
            self.reasons.append('semantic_failure_resume_parent');return None
        if not self.sem_used and (not early or obs['remaining']>=3):
            current=self.queue['sql'] if early else self.fallback['sql']
            action=self.choose(obs,schema,current,[current,*seeds[:8]],early)
            if action:
                self.pending=action['sql'];return action
        return super().candidate(obs,schema,seeds,early)
    def act(self,obs):
        if self.sem_pending is not None:
            success=obs['history'][-1]['tool']=='check' and passed(obs['history'][-1]['observation']);sql=self.sem_pending;self.sem_pending=None
            if success:return {'tool':'submit','sql':sql}
            self.sem_failed=True
            if self.post is not None:return self.post
        if self.post is None:
            action=super().act(obs)
            if action['tool']!='submit':return action
            self.post=action
        if self.sem_used or not obs['remaining']:return self.post
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if schema is None:
            q=obs['task']['question']
            if obs['remaining']>=2 and re.search(r"earliest|latest|recent|first|last|average|total|how many|number of|without|don't|\bnot\b|\bno\b",q,re.I):return {'tool':'schema'}
            return self.post
        seeds=[self.post['sql'],obs['issue_sql']]
        for e in obs['history']:
            if e['tool']=='repair_hint':seeds.extend(e['observation'])
        action=self.choose(obs,schema,self.post['sql'],seeds)
        return action if action else self.post

def create_agent(checkpoint=None):return Agent(checkpoint)
