"""Frozen full-chain frontier selected on train/validation only."""
from step107_chainnext_entity import Agent

def create_agent(checkpoint=None):
    return Agent(checkpoint, arm='entity')
