"""Compose the training-positive repairs under the round-61 call budget."""
from step49_agent import MeasureSuffix
from step50_counts import CountSuffix
from step51_agent import Agent as EventAgent
from step52_agent import ThresholdSuffix
from step53_agent import PopulationSuffix
from step55_agent import Agent as AverageAgent
from step56_agent import Agent as RoleAgent
from step57_agent import PathSuffix
from step59_agent import Agent as ValueAgent
from step60_agent import BranchSuffix
from step61_agent import create_agent as create_baseline


class ScheduledAction:
    def __init__(self, action):
        self.action = action

    def act(self, _):
        return self.action


def event_threshold(fallback):
    return ThresholdSuffix(fallback, "matched_shared")


# Try direct checks before plans that need a schema read or value sample.
# Equal-cost ties use round order, independently of grading outcomes.
RULES = {
    49: (MeasureSuffix, False), 50: (CountSuffix, False),
    51: (EventAgent, True), 52: (event_threshold, False),
    53: (PopulationSuffix, False), 55: (AverageAgent, True),
    56: (RoleAgent, True), 57: (PathSuffix, False),
    59: (ValueAgent, True), 60: (BranchSuffix, False),
}
ACTION_PRIORITY = {"check": 0, "execute": 1, "schema": 2}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.active = None
        self.anchor = None
        self.tried = set()
        self.audit = []

    def act(self, observation):
        if self.active is not None:
            number, patch, scheduled, entry = self.active
            if scheduled.action["tool"] == "check" and not observation["remaining"]:
                scheduled.action = self.baseline.act(observation)
            action = patch.act(observation)
            if action != scheduled.action:
                entry["actions"].append(action)
                return action
            self.active = None
            self.tried.add(number)
            original = self.anchor
        else:
            original = self.baseline.act(observation)

        last = observation["remaining"]
        if not last or (original["tool"] != "submit" and (original["tool"] != "check" or last > 2)):
            return original
        output = self.baseline.suffix
        if (output is not None and output.candidate
                and original["sql"] == output.candidate["sql"]):
            return original

        proposals = []
        for number, (factory, wraps_agent) in RULES.items():
            if number in self.tried or (original["tool"] != "submit" and not wraps_agent):
                continue
            scheduled = ScheduledAction(original)
            patch = factory(scheduled if wraps_agent else original)
            action = patch.act(observation)
            if action != original:
                proposals.append((number, patch, scheduled, action))
        if not proposals:
            return original
        winner = min(proposals, key=lambda item: (ACTION_PRIORITY[item[3]["tool"]], item[0]))
        number, patch, scheduled, action = winner
        entry = {"round": number, "remaining": last, "baseline_action": original,
                 "actions": [action], "alternatives": [{"round": n, "action": alternative}
                                                       for n, _, _, alternative in proposals if n != number]}
        self.anchor = original
        self.audit.append(entry)
        self.active = (number, patch, scheduled, entry)
        return action


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
