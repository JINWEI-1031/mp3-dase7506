"""Try one evidence-backed edit after the original draft fails its check."""
from pathlib import Path

from step4_agent import passed
from step25_agent import create_agent as create_baseline
from step40.probe import propose


PROBE_SETTINGS = {"trigger_after_charged_calls": 1, "candidate_cap": 3}


class EarlyRepairAgent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.probe = None
        self.reuse_at = None

    def act(self, observation):
        history = observation["history"]
        if self.probe is None:
            if (len(history) == 1 and observation["remaining"]
                    and history[0]["tool"] == "check"
                    and history[0]["sql"] == observation["issue_sql"]):
                _, pool = propose(observation, PROBE_SETTINGS)
                if pool:
                    self.probe = pool[0]
                    return {"tool": "check", "sql": self.probe["sql"]}
            return self.baseline.act(observation)

        feedback = history[1]
        if passed(feedback["observation"]):
            return {"tool": "submit", "sql": feedback["sql"]}

        # Preserve the baseline's original tool inputs while charging the
        # probe against the real remaining budget. If the baseline asks for
        # the same check later, give it that already observed result there.
        position = self.reuse_at
        resumed = [history[0], *history[2:]] if position is None else [
            history[0], *history[2:position], feedback, *history[position:]]
        action = self.baseline.act({**observation, "history": resumed})
        if position is None and action == {"tool": "check", "sql": feedback["sql"]}:
            self.reuse_at = len(history)
            action = self.baseline.act({**observation, "history": [*resumed, feedback]})
        if action["tool"] == "submit" or observation["remaining"]:
            return action
        fallback = next((event["sql"] for event in reversed(resumed)
                         if event["tool"] == "check" and passed(event["observation"])), observation["issue_sql"])
        return {"tool": "submit", "sql": fallback}


def create_agent(checkpoint=None):
    config = checkpoint or Path(__file__).with_name("configs") / "step25-exists.json"
    return EarlyRepairAgent(create_baseline(config))
