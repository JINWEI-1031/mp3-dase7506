"""Frozen complete chain; unchanged D; development-selected scopeflow arm."""
from step107_scopeflow_agent import Agent

def create_agent(checkpoint=None):
    return Agent(checkpoint, arm='sets')
