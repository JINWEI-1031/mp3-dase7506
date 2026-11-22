"""Selective ordinal interpretation plus preservation of valid pending proposals."""
import re
from step_pool192_priority import Agent as Priority
import step_pool192_rules as rules

original_ordinal=rules.ordinal
def ordinal(sql,obs,schema):
    question=re.sub(r'\bfirst\s+names?\b','given identifiers',obs['task']['question'],flags=re.I)
    question=re.sub(r'\blast\s+names?\b','family identifiers',question,flags=re.I)
    grounded={**obs,'task':{**obs['task'],'question':question}}
    return original_ordinal(sql,grounded,schema)

class Agent(Priority):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint);rules.ordinal=ordinal
    def act(self,obs):
        question=re.sub(r'\b(?:first|last)\s+names?\b','name fields',obs['task']['question'],flags=re.I)
        if not re.search(r'\b(?:first|last|earliest|latest)\s+(?!and\b)[a-z]',question,re.I):self.families.discard('ordinal')
        return super().act(obs)

def create_agent(checkpoint=None):return Agent(checkpoint)
