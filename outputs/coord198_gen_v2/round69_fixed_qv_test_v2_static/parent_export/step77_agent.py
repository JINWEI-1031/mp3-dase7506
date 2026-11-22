"""Reserve a value probe and final check for an unresolved login-state predicate."""
from step4_agent import passed
from step59_agent import Agent as ValueAgent, STATE
from step62_agent import ScheduledAction
from step72_agent import Agent as LastCheckAgent
from step63_agent import create_agent as create_baseline


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.active = None
        self.scheduled = None
        self.audit = None

    def act(self, observation):
        if self.active is not None:
            action = self.active.act(observation)
            if action == self.scheduled:
                self.active = None
            return action
        action = self.baseline.act(observation)
        history = observation["history"]
        if (observation["remaining"] != 2 or action["tool"] != "check"
                or not STATE.search(observation["task"]["question"])
                or not any(event["tool"] == "schema" for event in history)
                or any(event["tool"] == "check" and passed(event["observation"]) for event in history)):
            return action
        patch = ValueAgent(ScheduledAction(action))
        proposal = patch.act(observation)
        if proposal["tool"] != "execute":
            return action
        self.active = patch
        self.scheduled = action
        self.audit = {"displaced": action, "probe": proposal}
        return proposal


def create_agent(checkpoint=None):
    return Agent(LastCheckAgent(create_baseline(checkpoint)))
