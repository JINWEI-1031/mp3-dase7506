"""Keep semantic review after the early structural repair stage."""
from step116_agent import Agent as EarlierAgent


class Agent(EarlierAgent):
    def act(self, observation):
        action = super().act(observation)
        if self.early and action['tool'] == 'submit':
            self.early = False
            self.finished = False
            self.pending = None
            self.fallback = action
            return super().act(observation)
        return action


def create_agent(checkpoint=None):
    return Agent(checkpoint)
