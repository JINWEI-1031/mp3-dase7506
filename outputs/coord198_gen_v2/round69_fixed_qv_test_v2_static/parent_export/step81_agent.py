"""Integrated SQL repairs with reserved checks and bounded tool feedback."""
from step63_agent import create_agent as create_baseline
from step72_agent import Agent as LastCheckAgent
from step76_agent import Agent as RuleAgent, RULES as PREVIOUS_RULES
from step77_agent import Agent as LoginAgent
from step78_agent import BoundarySuffix
from step79_agent import IntersectionSuffix
from step80_agent import PurchaseSuffix


RULES = {**PREVIOUS_RULES, 78: (BoundarySuffix, False),
         79: (IntersectionSuffix, False), 80: (PurchaseSuffix, False)}


class Agent(RuleAgent):
    def __init__(self, baseline):
        super().__init__(baseline, rules=RULES)


def create_agent(checkpoint=None):
    return Agent(LoginAgent(LastCheckAgent(create_baseline(checkpoint))))
