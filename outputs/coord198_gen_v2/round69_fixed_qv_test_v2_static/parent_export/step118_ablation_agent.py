"""Separate Step117 controller gains from live base-model and LoRA contributions."""
import os
from step117_agent import Agent
from step116_agent import Agent as EarlierAgent
from step61_agent import create_agent as rules


def create_agent(checkpoint=None):
    agent = (EarlierAgent if os.environ.get('MP3_STEP118_STAGE') == 'earlier' else Agent)(checkpoint)
    if os.environ.get('MP3_STEP118_MODEL') == 'off':
        agent.base = rules(checkpoint)
    return agent
