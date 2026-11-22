"""Keep round-47 checks and verify the row unit at submission."""
from sqlglot import exp

from repair_rules import parse
from step30_multiplicity import one_row_evidence
from step38.event_grain import terms
from step45_agent import multiplicity_evidence
from step47_agent import create_agent as create_baseline


def row_unit_evidence(observation, original_sql, candidate):
    before, after = parse(original_sql), parse(candidate["sql"])
    basis = candidate["basis"]
    if not isinstance(before, exp.Select) or not isinstance(after, exp.Select):
        return False, "unsupported row source"
    if basis["row_multiplicity"] != "one row per qualifying output key":
        return multiplicity_evidence(observation, before, after, basis)

    root, key = basis["output_entity"], basis["output_key"]
    words = terms(root)
    question = observation["task"]["question"]
    explicit_unit = set(one_row_evidence(question))
    if explicit_unit:
        if words and words <= explicit_unit:
            return True, "explicit one row per named entity"
        return False, "explicit row unit names an association or another entity"

    paths = basis["key_paths"]
    witnesses = basis["witness_roles"]
    # Round 38 already matched the output to the named root and proved
    # branching via declared foreign keys. A single direct witness can
    # qualify that root. Intermediate relations can themselves carry rows.
    if (not before.args.get("distinct") and len(paths) == 2 and len(witnesses) == 1
            and len(paths[witnesses[0]]) == 1 and tuple(paths[witnesses[0]][0][:2]) == (root, key)
            and all(table == root for table, _ in basis["output_columns"])):
        return True, "projected root qualified by a single direct keyed witness"
    return False, "preserve association rows or existing DISTINCT without explicit entity grain"


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.decision = None

    def act(self, observation):
        action = self.baseline.act(observation)
        suffix = self.baseline.suffix
        if action["tool"] != "submit" or suffix is None or action == suffix.fallback:
            return action
        candidate = suffix.audit["selected"]
        if candidate["basis"].get("source") != "event_grain":
            return action
        assert action["sql"] == candidate["sql"]
        accepted, reason = row_unit_evidence(observation, suffix.fallback["sql"], candidate)
        self.decision = {"accepted": accepted, "reason": reason, "mode": candidate["basis"]["row_multiplicity"]}
        return action if accepted else suffix.fallback


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
