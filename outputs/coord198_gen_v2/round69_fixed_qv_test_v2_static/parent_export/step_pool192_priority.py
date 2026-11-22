"""Repair the pending proposal before replacing it with a different hypothesis."""
import os
from step_pool192_agent import Agent as Parent
from step_pool192_rules import FAMILIES,schema_errors

class Agent(Parent):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint)
        self.families={'shape'} if os.environ.get('MP3_POOL192_ARM')=='shape_priority' else set(FAMILIES)
    def act(self,obs):
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.queue is not None and self.pending is None and schema is not None and schema_errors(self.queue['sql'],obs,schema)==0:
            action=self.queue;self.queue=None;self.saved=[]
            return action if obs['remaining'] else {'tool':'submit','sql':action['sql']}
        return super().act(obs)

def create_agent(checkpoint=None):return Agent(checkpoint)
