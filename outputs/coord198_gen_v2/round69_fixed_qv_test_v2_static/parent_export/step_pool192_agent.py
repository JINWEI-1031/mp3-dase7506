"""Targeted generation over the frozen 192/240 baseline, using real observations only."""
import os,re
from step124_pool_frozen import create_agent as baseline
from step124_pool_search import materialized
from step4_agent import identity,passed
from mp3_sql.sqlite_tool import validate_sql
from step_pool192_rules import FAMILIES,generate,schema_errors,score,pair_terms

class Agent:
    def __init__(self,checkpoint=None):
        self.base=baseline(checkpoint);arm=os.environ.get('MP3_POOL192_ARM','combined')
        self.families=set(FAMILIES) if arm=='combined' else set(arm.split(',')) if arm!='baseline' else set()
        self.queue=None;self.pending=None;self.fallback=None;self.phase='base';self.saved=[];self.audit=[];self.pool=[]

    def candidate(self,obs,schema,seeds,early=False):
        self.pool=generate(seeds,obs,schema,self.families)
        checked={identity(e['sql']) for e in obs['history'] if e['tool'] in ('check','execute')}
        current=self.queue['sql'] if early else self.fallback['sql'];threshold=score(current,obs,schema)
        matched=any(e['tool']=='check' and passed(e['observation']) for e in obs['history'])
        for item in self.pool:
            if identity(item['sql']) in checked or identity(item['sql'])==identity(current):continue
            if schema_errors(item['sql'],obs,schema)>0:continue
            if not early and matched and item['score']<threshold+.5:continue
            try:validate_sql(item['sql'])
            except ValueError:continue
            self.pending=item['sql'];self.audit.append({**item,'early':early,'remaining':obs['remaining']})
            return {'tool':'check','sql':item['sql']}
        return None

    def act(self,obs):
        if not self.families:return self.base.act(obs)
        history=obs['history'];schema=next((e['observation'] for e in history if e['tool']=='schema'),None)
        if self.pending is not None:
            sql=self.pending;self.pending=None
            if history[-1]['tool']=='check' and passed(history[-1]['observation']):return {'tool':'submit','sql':sql}
        if self.queue is not None:
            if obs['remaining'] and schema is not None:
                action=self.candidate(obs,schema,[self.queue['sql'],*self.saved,*[e['sql'] for e in history if e['tool']=='check']],True)
                if action:return action
            action=self.queue;self.queue=None;self.saved=[]
            if obs['remaining']:return action
            return {'tool':'submit','sql':action['sql']}
        if self.phase=='base':
            action=self.base.act(obs)
            if action['tool']=='check' and self.families & {'shape','entity'} and obs['remaining']>=2:
                errors=[e['observation'].get('error') for e in history if e['tool']=='check' and e['observation'].get('error')]
                evidence=any(re.search(r'ambiguous column|no such (?:table|column)',e.get('message',''),re.I) for e in errors)
                if evidence and (schema is None or schema_errors(action['sql'],obs,schema)>0):
                    self.queue=action;self.saved=materialized(self.base)
                    if schema is None:return {'tool':'schema'}
                    candidate=self.candidate(obs,schema,[action['sql'],*self.saved],True)
                    if candidate:return candidate
                    self.queue=None
            if action['tool']!='submit':return action
            self.fallback=action;self.phase='review';self.saved=materialized(self.base)
        if not obs['remaining']:return self.fallback
        q=obs['task']['question'];current=self.fallback['sql']
        risk=('ordinal' in self.families and bool(re.search(r'\b(?:first|last|earliest|latest)\s+(?!name\b|and\b)[a-z]',q,re.I))) or ('categories' in self.families and pair_terms(q))
        risk=risk or any(e['tool']=='check' and (e['observation'].get('error') or obs['example']['row_count']>0 and e['observation']['row_count']==0) for e in history)
        if not risk:return self.fallback
        if schema is None:
            if obs['remaining']>=2:return {'tool':'schema'}
            return self.fallback
        seeds=[current,obs['issue_sql'],*self.saved,*[e['sql'] for e in history if e['tool']=='check']]
        for e in history:
            if e['tool']=='repair_hint':seeds.extend(e['observation'])
        candidate=self.candidate(obs,schema,seeds)
        return candidate if candidate else self.fallback

def create_agent(checkpoint=None):return Agent(checkpoint)
