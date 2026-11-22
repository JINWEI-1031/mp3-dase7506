"""Restore historical Step107 structural repairs displaced by the final model check."""
from step107_agent import create_agent as original
from step71_agent import WitnessSuffix
from step100_agent import MeasureSuffix
from step4_agent import passed


def proposal(obs):
    if (obs['remaining'] == 1
            and any(e['tool'] == 'schema' for e in obs['history'])
            and not any(e['tool'] == 'check' and passed(e['observation']) for e in obs['history'])):
        fallback = {'tool': 'submit', 'sql': obs['issue_sql']}
        for factory in (WitnessSuffix, MeasureSuffix):
            patch = factory(fallback)
            action = patch.act(obs)
            if action['tool'] == 'check':
                return patch, action
    return None


class Agent:
    def __init__(self, checkpoint=None):
        self.base = original(checkpoint)
        self.replacement = None
        self.audit = []

    def act(self, obs):
        if self.replacement is not None:
            return self.replacement.act(obs)
        # Existing rules only; preserve all public-matching trajectories and use
        # one remaining check, with the ordinary failure fallback.
        choice = proposal(obs)
        if choice:
            self.replacement, action = choice
            self.audit.append({'kind': 'restore_historical_last_check', 'rule': type(self.replacement).__name__, 'sql': action['sql']})
            return action
        return self.base.act(obs)


def create_agent(checkpoint=None): return Agent(checkpoint)
