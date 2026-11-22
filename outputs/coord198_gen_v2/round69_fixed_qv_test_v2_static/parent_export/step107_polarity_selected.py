"""Frozen full-chain polarity/role research; D unchanged."""
from step107_polarity_agent import Agent

def create_agent(checkpoint=None):
    return Agent(checkpoint, arm='role_early')
