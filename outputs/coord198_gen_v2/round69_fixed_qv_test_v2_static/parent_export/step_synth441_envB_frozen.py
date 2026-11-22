"""Fixed count release qualified in environment B; source algorithm unchanged."""
from step_synth441_release_agent import Agent

def create_agent(checkpoint=None):
    return Agent(checkpoint, arm='count')
