"""Compose the rule repairs under the original eight-call tool budget."""
from step27_agent import Step27Agent
from step49_agent import MeasureSuffix
from step50_counts import CountSuffix
from step51_agent import Agent as EventAgent
from step53_agent import PopulationSuffix
from step55_agent import Agent as AverageAgent
from step56_agent import Agent as RoleAgent
from step57_agent import PathSuffix
from step59_agent import Agent as ValueAgent
from step60_agent import BranchSuffix
from step62_agent import ScheduledAction, event_threshold
from step69_agent import ConflictSuffix
from step70_agent import CategorySuffix
from step71_agent import WitnessSuffix
from step72_agent import Agent as LastCheckAgent
from step73_agent import BridgeSuffix
from step74_agent import DedupSuffix
from step75_agent import CourseSuffix
from step63_agent import create_agent as create_baseline


RULES = {
    27: (Step27Agent, True), 49: (MeasureSuffix, False), 50: (CountSuffix, False),
    51: (EventAgent, True), 52: (event_threshold, False), 53: (PopulationSuffix, False),
    55: (AverageAgent, True), 56: (RoleAgent, True), 57: (PathSuffix, False),
    59: (ValueAgent, True), 60: (BranchSuffix, False), 69: (ConflictSuffix, False),
    70: (CategorySuffix, False), 71: (WitnessSuffix, False), 73: (BridgeSuffix, False),
    74: (DedupSuffix, False), 75: (CourseSuffix, False),
}
PRIORITY = {"submit": 0, "check": 1, "execute": 2, "schema": 3}


class Agent:
    def __init__(self, baseline, rules=RULES, superseded_rules=None):
        self.baseline = baseline
        self.rules = rules
        self.superseded_rules = superseded_rules or {}
        self.anchor = None
        self.active = None
        self.tried = set()
        self.audit = []

    def act(self, observation):
        if self.anchor is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.anchor = action
        if self.active is not None:
            number, patch, entry = self.active
            action = patch.act(observation)
            if action["tool"] != "submit":
                entry["actions"].append(action)
                return action
            if action != self.anchor:
                self.tried.update(self.superseded_rules.get(number, ()))
            self.anchor = action
            self.tried.add(number)
            self.active = None
        if not observation["remaining"]:
            return self.anchor
        choices = []
        for number, (factory, wraps_agent) in self.rules.items():
            if number in self.tried:
                continue
            patch = factory(ScheduledAction(self.anchor) if wraps_agent else self.anchor)
            action = patch.act(observation)
            if action != self.anchor:
                choices.append((number, patch, action))
        if not choices:
            return self.anchor
        number, patch, action = min(choices, key=lambda item: (PRIORITY[item[2]["tool"]], item[0]))
        entry = {"round": number, "before": self.anchor, "actions": [action]}
        self.audit.append(entry)
        if action["tool"] == "submit":
            self.anchor = action
            self.tried.add(number)
            return self.act(observation)
        self.active = number, patch, entry
        return action


def create_agent(checkpoint=None):
    return Agent(LastCheckAgent(create_baseline(checkpoint)))
