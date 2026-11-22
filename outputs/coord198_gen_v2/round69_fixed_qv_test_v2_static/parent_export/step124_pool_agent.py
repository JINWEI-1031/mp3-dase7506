"""Step124 candidate-pool search; independent namespace for this experiment."""
import os
from step124_agent import create_agent as baseline
from step4_agent import identity,passed
from step124_pool_search import materialized,expand,score
from mp3_sql.sqlite_tool import validate_sql

class Agent:
    def __init__(self,checkpoint=None):
        self.base=baseline(checkpoint);self.arm=os.environ.get('MP3_POOL_ARM','combined')
        self.fallback=None;self.searching=False;self.pool=[];self.audit=[];self.seed=[]
    def act(self,obs):
        history=obs['history'];checks=[e for e in history if e['tool']=='check']
        matched=[e for e in checks if passed(e['observation'])]
        schema=next((e['observation'] for e in history if e['tool']=='schema'),None)
        if not self.searching:
            action=self.base.act(obs)
            if action.get('sql'):self.seed.append(action['sql'])
            reserve=self.arm in ('reserve','combined') and not matched and len(checks)>=2 and obs['remaining']<=3
            if action['tool']!='submit' and not reserve:return action
            self.searching=True
            self.fallback=action if action['tool']=='submit' else {'tool':'submit','sql':checks[-1]['sql'] if checks else obs['issue_sql']}
        if self.arm=='baseline':return self.fallback
        if schema is None:
            if obs['remaining']>=2:return {'tool':'schema'}
            return self.fallback
        seeds=[obs['issue_sql'],self.fallback['sql'],*self.seed,*materialized(self.base),*[e['sql'] for e in checks]]
        for e in history:
            if e['tool']=='repair_hint':seeds.extend(e['observation'])
        depth=0 if self.arm in ('pool','reserve') else 1 if self.arm=='search1' else 2
        candidates=expand(seeds,obs,schema,depth=depth);self.pool=candidates
        bykey={identity(e['sql']):e for e in checks}
        base_score=score(self.fallback['sql'],obs,schema)
        best=max((e['sql'] for e in matched),key=lambda sql:score(sql,obs,schema),default=None)
        bestscore=score(best,obs,schema) if best else -10000
        gap=.5
        for item in candidates:
            if obs['remaining']<=0:break
            if identity(item['sql']) in bykey:continue
            if best is not None and item['score']<bestscore+gap:continue
            try:validate_sql(item['sql'])
            except ValueError:continue
            self.audit.append({'sql':item['sql'],'score':item['score'],'path':item['path'],'remaining':obs['remaining']})
            return {'tool':'check','sql':item['sql']}
        if best is not None and (identity(best)==identity(self.fallback['sql']) or bestscore>=base_score+gap):
            return {'tool':'submit','sql':best}
        return self.fallback
def create_agent(checkpoint=None):return Agent(checkpoint)
