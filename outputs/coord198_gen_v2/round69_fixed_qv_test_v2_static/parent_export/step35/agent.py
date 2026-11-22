"""Compare predicate sources under the same checking and slot-eight policy."""
from itertools import islice
import json
from pathlib import Path

from repair_rules import parse
from step4_agent import identity, passed
from step25_agent import create_agent as step25
from step32_plan import compiled, context
from step34.quantities import schedule_pool
from step35.sources import POOL_LIMIT, proposal_inputs, proposals

ARMS = ("submit", "executed")


class Step35Agent:
    def __init__(self, baseline, arm, replace_slot=True):
        self.baseline, self.arm = baseline, arm
        self.replace_slot = replace_slot
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "already_checked": [], "eligible": False,
                      "trigger": False, "schema_requested": False, "schema_only_sql": [],
                      "fixed_slot_pool": []}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback
        action = self.fallback or self.baseline.act(obs)
        public = any(e["tool"] == "check" and passed(e["observation"]) for e in history)
        if remaining == 1 and action["tool"] == "check" and not public:
            fixed = schedule_pool(obs)
            self.audit["fixed_slot_pool"] = fixed
            if fixed and self.replace_slot and identity(fixed[0]["sql"]) != identity(action["sql"]):
                choice = fixed[0]
                self.fallback = {"tool": "submit", "sql": obs["issue_sql"]}
                self.audit.update(pool=fixed, trigger=True, selected=choice, replaced=action["sql"],
                                  checked=[choice["sql"]])
                return {"tool": "check", "sql": choice["sql"]}
        if action["tool"] != "submit":
            return action
        self.audit["eligible"] = public
        c = context(obs)
        c["baseline"] = action["sql"]
        parent = parse(action["sql"])
        inputs = proposal_inputs(obs, c, parent, public, self.arm == "executed")
        generated = compiled(c, parent, proposals(obs, c, parent, public, inputs))
        bound = list(islice(generated, 32))
        # Use the same eligibility and priority in both arms. Keep an
        # equally bounded diagnostic pool after sorting selectable plans.
        bound.sort(key=lambda p: not p["basis"]["eligible"])
        bound = bound[:POOL_LIMIT]
        requested = inputs[-1]
        self.audit.update(binding_requests=requested, donor_input_count=len(inputs[1]),
                          submit_proofs=len(inputs[0]))
        if (c["schema"] is None and not any(p["basis"]["eligible"] for p in bound)
                and remaining >= 2 and requested):
            self.fallback = action
            self.audit["schema_requested"] = True
            return {"tool": "schema"}
        self.audit.update(pool=[*bound, *self.audit["fixed_slot_pool"]], trigger=bool(bound or requested or self.audit["fixed_slot_pool"]),
                          schema_only_sql=[p["sql"] for p in bound] if self.audit["schema_requested"] else [])
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        self.audit["already_checked"] = [p["sql"] for p in self.audit["pool"] if identity(p["sql"]) in previous]
        choice = next((p for p in bound if p["basis"]["eligible"] and identity(p["sql"]) not in previous), None)
        if choice and remaining:
            self.fallback = action
            self.audit.update(checked=[choice["sql"]], selected=choice)
            return {"tool": "check", "sql": choice["sql"]}
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError("Unknown step35 arm")
    return Step35Agent(step25(path.parent.parent / "configs/step25-exists.json"), arm)

