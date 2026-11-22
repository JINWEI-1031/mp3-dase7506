"""Frozen Step124 pool search with existing D."""
from step124_pool_neutral import create_agent as factory
def create_agent(checkpoint=None):
    agent=factory(checkpoint)
    agent.arm='search1'
    return agent
