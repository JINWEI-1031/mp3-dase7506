"""Integrated repairs with precedence for successful event-specific grouping."""
from step63_agent import create_agent as create_baseline
from step72_agent import Agent as LastCheckAgent
from step76_agent import Agent as RuleAgent
from step77_agent import Agent as LoginAgent
from step81_agent import RULES


class Agent(RuleAgent):
    def __init__(self, baseline):
        super().__init__(baseline, rules=RULES, superseded_rules={50: {27}, 52: {27}})


def create_agent(checkpoint=None):
    return Agent(LoginAgent(LastCheckAgent(create_baseline(checkpoint))))
