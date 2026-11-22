"""Check deduplication of an observed entity list when its public rows prove the gap."""
from collections import Counter

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step38.event_grain import intent, terms
from step63_agent import create_agent as create_baseline


def deduplicated_rows(feedback, example, ordered):
    if (not feedback["ok"] or feedback["truncated"] or example["truncated"]
            or not example["rows"] or feedback["row_count"] != len(feedback["rows"])
            or example["row_count"] != len(example["rows"])):
        return False
    # Rows arrive as JSON scalar arrays, so tuple equality preserves their SQL values.
    rows = list(dict.fromkeys(tuple(row) for row in feedback["rows"]))
    expected = [tuple(row) for row in example["rows"]]
    return len(rows) < len(feedback["rows"]) and (
        rows == expected if ordered else Counter(rows) == Counter(expected))


def dedup_candidate(observation):
    roles = intent(observation["task"]["question"])
    if roles is None or observation["ordered"]:
        return None
    history = observation["history"]
    if any(event["tool"] == "check" and passed(event["observation"]) for event in history):
        return None
    candidates = {}
    checked = {identity(event["sql"]) for event in history if event["tool"] == "check"}
    for event in history:
        if event["tool"] != "check" or not deduplicated_rows(
                event["observation"], observation["example"], False):
            continue
        tree = parse(event["sql"])
        if (not isinstance(tree, exp.Select) or tree.args.get("distinct")
                or any(tree.args.get(key) for key in ("group", "having", "order", "limit", "offset", "with"))
                or len(list(tree.find_all(exp.Select))) != 1
                or len(tree.expressions) != len(observation["example"]["columns"])):
            continue
        root = tree.args.get("from")
        joins = tree.args.get("joins", [])
        if (root is None or not isinstance(root.this, exp.Table) or not joins
                or root.this.db or root.this.catalog or not terms(root.this.name) & roles[0]):
            continue
        if any(join.side or join.kind not in ("", "INNER") or not isinstance(join.this, exp.Table)
               or join.this.db or join.this.catalog or not isinstance(join.args.get("on"), exp.EQ)
               for join in joins):
            continue
        projection = [item.unalias() for item in tree.expressions]
        if (any(not isinstance(item, exp.Column) or item.is_star
                or item.table.casefold() != root.this.alias_or_name.casefold() for item in projection)
                or [item.alias_or_name.casefold() for item in tree.expressions]
                != [head.casefold() for head in observation["example"]["columns"]]):
            continue
        if not any(terms(join.this.name) & roles[1] for join in joins):
            # The event may be a field such as Grade rather than a named relation.
            where = tree.args.get("where")
            if where is None or not any(terms(col.name) & roles[1] for col in where.find_all(exp.Column)):
                continue
        tree.set("distinct", exp.Distinct())
        sql = sql_text(tree)
        if usable(sql) and identity(sql) not in checked:
            candidates[identity(sql)] = {"sql": sql, "source": event["sql"],
                                        "rows_before": event["observation"]["row_count"],
                                        "rows_after": observation["example"]["row_count"]}
    return next(iter(candidates.values())) if len(candidates) == 1 else None


class DedupSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        if self.candidate is not None:
            return {"tool": "submit", "sql": self.candidate} if passed(
                observation["history"][-1]["observation"]) else self.fallback
        if not observation["remaining"]:
            return self.fallback
        self.evidence = dedup_candidate(observation)
        if self.evidence is None:
            return self.fallback
        self.candidate = self.evidence["sql"]
        return {"tool": "check", "sql": self.candidate}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = DedupSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
