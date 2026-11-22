"""Frozen full-chain frontier selected on train/validation only."""
from step107_frontier_agent import Agent

def create_agent(checkpoint=None):
    return Agent(checkpoint, arm='search1')
