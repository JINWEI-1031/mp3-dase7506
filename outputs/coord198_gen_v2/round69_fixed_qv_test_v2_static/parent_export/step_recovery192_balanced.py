"""Preserve the parent while allowing one newly justified feedback edit."""
from step_recovery192 import Agent as Parent
class Agent(Parent):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint);self.features={'exit','priority','dedup'}
    def candidate(self,obs,schema,seeds,early=False):
        checks=[e for e in obs['history'] if e['tool']=='check']
        progressed=bool(early and self.failures==1 and obs['remaining']>=3 and checks and self.audit and self.audit[-1]['path'][-1]=='shape' and checks[-1]['observation']['ok'] and checks[-1]['observation']['row_count']==0 and obs['example']['row_count']>0)
        if not progressed:return super().candidate(obs,schema,seeds,early)
        previous=self.features;self.features=previous-{'exit'}
        try:return super().candidate(obs,schema,seeds,early)
        finally:self.features=previous
def create_agent(checkpoint=None):return Agent(checkpoint)
