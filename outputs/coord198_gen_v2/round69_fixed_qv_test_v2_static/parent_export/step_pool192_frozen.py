"""Frozen candidate generation over the 192/240 D release."""
from step_pool192_selective import Agent
def create_agent(checkpoint=None):
    agent=Agent(checkpoint)
    agent.families=set(['categories', 'entity', 'ordinal', 'shape', 'value'])
    return agent
