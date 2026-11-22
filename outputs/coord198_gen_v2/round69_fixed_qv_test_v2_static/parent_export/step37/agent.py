"""Compare the repaired AST guard with an expanded question grammar."""
from itertools import islice
import json
from pathlib import Path

from sqlglot import exp

from repair_rules import parse
from step4_agent import identity, passed
from step25_agent import create_agent as step25
from step32_plan import compiled, context, tree_plan
from step34.quantities import schedule_pool
from step35.sources import POOL_LIMIT, proposal_inputs, proposals
from step36.frequency import infer, needs_observation, question_plans
from step37.counts import grain_audit, projection_plan, selection_evidence
from step37 import membership

ARMS = ("ast", "membership")


class Step37Agent:
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
        inputs = proposal_inputs(obs, c, parent, public, False)
        base_pool = list(islice(compiled(c, parent, proposals(obs, c, parent, public, inputs)), 20))
        base_pool.sort(key=lambda p: not p["basis"]["eligible"])
        direct = []
        intent = infer(c["question"], c["columns"])
        relation_intent = membership.infer(c["question"], c["columns"])
        distinct = isinstance(parent, exp.Select) and bool(parent.args.get("distinct"))
        can_preserve_distinct = distinct and not any(parent.args.get(k) for k in
                                                     ("group", "having", "order", "limit", "with"))
        projection = projection_plan(parent, c["columns"])
        if projection:
            repaired = list(compiled(c, parent, [projection]))
            for p in repaired:
                p["basis"]["grain_review"] = grain_audit(parent, p["sql"])
            direct.extend(repaired)
        if intent and c["schema"] is not None:
            frequency = list(islice(compiled(c, parent, question_plans(c["question"], c["columns"], c["schema"])), 6))
            for option in frequency:
                evidence = selection_evidence(parent, c, option, public)
                option["basis"].update(source="question_schema", eligible=bool(evidence),
                                       advantage=evidence, grain_review=grain_audit(parent, option["sql"]))
            direct.extend(frequency)
        if relation_intent and c["schema"] is not None and self.arm == "membership":
            candidates = list(islice(compiled(c, parent, membership.plans(c["question"], c["columns"], c["schema"])), 6))
            for option in candidates:
                if public and distinct:
                    tree = parse(option["sql"]).distinct()
                    proposal = tree_plan(tree, option["basis"])
                    option = next(compiled(c, parent, [proposal]), None)
                    if option is None:
                        continue
                # For an existing public match, changing the row-source
                # bag remains diagnostic. Existing DISTINCT can be kept
                # for a positively stated existence relation.
                supported = not public or can_preserve_distinct and option["basis"]["mode"] == "entity exists" and (
                    not parent.args.get("where") or relation_intent.value is not None and
                    [v.this for v in parent.args["where"].find_all(exp.Literal)] == [relation_intent.value])
                option["basis"].update(source="question_membership", eligible=bool(supported))
                direct.append(option)
        requested = list(inputs[-1])
        if needs_observation(intent, parent, public) and not projection:
            requested.append("question event and output roles")
        if relation_intent and (not public or can_preserve_distinct):
            requested.append("named entity and event relationship")
        if c["schema"] is None and not projection and not any(p["basis"]["eligible"] for p in base_pool) and remaining >= 2 and requested:
            self.fallback = action
            self.audit["schema_requested"] = True
            self.audit["binding_requests"] = requested
            return {"tool": "schema"}
        known = {identity(p["sql"]) for p in base_pool}
        bound = [*base_pool, *(p for p in direct if identity(p["sql"]) not in known)]
        bound.sort(key=lambda p: not p["basis"]["eligible"])
        bound = bound[:POOL_LIMIT]
        self.audit.update(binding_requests=requested, question_intent=intent.mode if intent else None, 
                          question_candidates=len(direct), membership_intent=bool(relation_intent), submit_proofs=len(inputs[0]))
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
        raise ValueError("Unknown step37 arm")
    return Step37Agent(step25(path.parent.parent / "configs/step25-exists.json"), arm)





