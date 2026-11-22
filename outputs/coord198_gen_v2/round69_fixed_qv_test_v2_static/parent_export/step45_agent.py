"""Keep round 42's checks fixed and audit evidence before replacing its fallback."""
from collections import Counter
from pathlib import Path
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity
from step25_agent import create_agent as create_baseline
from step30_multiplicity import one_row_evidence
from step32_plan import context
from step36.frequency import infer as frequency_intent, order_count
from step37.counts import count_positions, requested_counts
from step38.agent import Suffix
from step38.event_grain import terms


ENTITY_HAS = re.compile(r"^\s*(?:which|what)\s+(.+?)\s+(?:have|has|had)\s+([^?!.]+)", re.I)
DEDUP = re.compile(r"\b(?:distinct|unique|different|each|once)\b", re.I)


def same_except(left, right, slot):
    left, right = left.copy(), right.copy()
    left.set(slot, None)
    right.set(slot, None)
    return left == right


def multiplicity_evidence(observation, before, after, basis):
    question = observation["task"]["question"]
    mode = basis["row_multiplicity"]
    root = terms(basis["output_entity"])
    if mode == "one row per qualifying output key":
        explicit = root and root <= set(one_row_evidence(question))
        if explicit:
            return True, "explicit entity row unit"
        mention = ENTITY_HAS.match(question)
        event = set().union(*(terms(role) for role in basis["witness_roles"])) - root
        if mention and root and root <= terms(mention[1]) and event & terms(mention[2]):
            return True, "named output entities with an explicitly possessed relation"
        return False, "no independent evidence for collapsing joined records to entity keys"
    draft = parse(observation["issue_sql"])
    if (mode == "one row per joined witness tuple" and isinstance(draft, exp.Select)
            and not draft.args.get("distinct") and not DEDUP.search(question)
            and before.args.get("distinct") and not after.args.get("distinct")
            and same_except(before, after, "distinct")
            and [col.alias_or_name.casefold() for col in draft.expressions]
            == [col.alias_or_name.casefold() for col in after.expressions]):
        return True, "restore draft row multiplicity without changing the repaired join or predicates"
    if (mode == "one row per projected value tuple" and after.args.get("distinct")
            and re.search(r"\b(?:distinct|unique|different)\b", question, re.I)
            and same_except(before, after, "distinct")):
        return True, "explicit distinct projected values"
    return False, "no evidence for changing projected row multiplicity"


def replacement_evidence(observation, original_sql, candidate):
    visible = context(observation)
    before, after = parse(original_sql), parse(candidate["sql"])
    basis = candidate["basis"]
    kind = basis.get("kind")
    if not isinstance(after, exp.Select) or not basis.get("eligible"):
        return False, "unsupported proposal"
    if basis.get("source") == "event_grain":
        return multiplicity_evidence(observation, before, after, basis)
    if kind == "count_projection_order":
        expected = requested_counts(visible["columns"])
        supported = (isinstance(before, exp.Select) and len(expected) == 1
                     and count_positions(before) != expected == count_positions(after)
                     and same_except(before, after, "expressions")
                     and Counter(map(sql_text, before.expressions)) == Counter(map(sql_text, after.expressions)))
        return supported, "count positions and unchanged non-projection SQL"
    if kind == "question_schema_frequency":
        intent = frequency_intent(visible["question"], visible["columns"])
        checked_count = order_count(after)
        supported = bool(intent and intent.mode == basis["mode"] and basis["event_terms"] and basis["path"]
                         and after.args.get("group") and (checked_count if intent.mode == "rank by frequency"
                                                          else count_positions(after) == requested_counts(visible["columns"])))
        return supported, "named event, declared path and count at the requested grain"
    if kind == "declared relation graph":
        return bool(basis.get("advantage") and basis.get("bindings")), "question-bound roles and graph evidence"
    if kind == "question_schema_membership":
        return bool(basis.get("root") and basis.get("event") and basis.get("path")), "bound relationship roles and key path"
    if kind in {"witnesses", "counted_group_witnesses", "population"}:
        return bool(basis.get("evidence")), "grounded predicate scope or comparison population"
    return False, "unrecognized rewrite type"


class EvidenceSuffix(Suffix):
    def __init__(self, fallback):
        super().__init__(fallback, use_new=True, allow_schema=True)

    def act(self, observation):
        action = super().act(observation)
        if action["tool"] != "submit" or action == self.fallback:
            return action
        candidate = self.audit["selected"]
        assert identity(candidate["sql"]) == identity(action["sql"])
        accepted, reason = replacement_evidence(observation, self.fallback["sql"], candidate)
        self.audit["replacement"] = {"accepted": accepted, "reason": reason, "kind": candidate["basis"].get("kind"),
                                     "row_multiplicity": candidate["basis"].get("row_multiplicity")}
        return action if accepted else self.fallback


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = EvidenceSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    config = checkpoint or Path(__file__).with_name("configs") / "step25-exists.json"
    return Agent(create_baseline(config))
