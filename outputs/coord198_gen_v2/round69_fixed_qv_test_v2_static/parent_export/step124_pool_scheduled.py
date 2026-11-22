"""Spend one low-budget public-mismatch probe on search, then resume the baseline."""
from step124_pool_scoped import create_agent as scoped,score
from step124_pool_search import materialized,expand
from step4_agent import passed,identity
from mp3_sql.sqlite_tool import validate_sql

class BudgetScheduler:
    def __init__(self,base):self.base=base;self.used=False;self.audit=[]
    def act(self,obs):
        action=self.base.act(obs)
        checks=[e for e in obs['history'] if e['tool']=='check']
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.used or schema is None or not 0<obs['remaining']<=3 or len(checks)<2 or any(passed(e['observation']) for e in checks) or action['tool'] not in ('execute','retrieve'):return action
        seeds=[obs['issue_sql'],*materialized(self.base),*[e['sql'] for e in checks]]
        if action.get('sql'):seeds.append(action['sql'])
        for e in obs['history']:
            if e['tool']=='repair_hint':seeds.extend(e['observation'])
        seen={identity(e['sql']) for e in obs['history'] if e['tool'] in ('execute','check')}
        for item in expand(seeds,obs,schema,depth=2):
            if identity(item['sql']) in seen:continue
            try:validate_sql(item['sql'])
            except ValueError:continue
            self.used=True;self.audit.append({**item,'remaining':obs['remaining'],'replaced_tool':action['tool']})
            return {'tool':'check','sql':item['sql']}
        return action

def create_agent(checkpoint=None):
    agent=scoped(checkpoint);agent.base=BudgetScheduler(agent.base);return agent
