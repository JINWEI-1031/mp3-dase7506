"""Follow-up policies using only official public observations."""
from pathlib import Path
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import COUNT_RANK, sql_text
from step4_agent import continuation_conflicts, identity, passed
from step4_edits import simple, threshold_edits
from step5_agent import load_policy
from step5x_agent import ExpansionAgent
from step5x_edits import foreign_key_edits


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def mentioned(value, question):
    tokens = words(value)
    query = words(question)
    return bool(tokens) and any(query[i:i + len(tokens)] == tokens for i in range(len(query)))


def top_terms(node):
    if isinstance(node, exp.And):
        return [*top_terms(node.this), *top_terms(node.expression)]
    return [node]


def contradiction(tree):
    where = tree.args.get("where")
    if not where:
        return False
    bounds = {}
    for node in top_terms(where.this):
        if (type(node) not in (exp.GT, exp.GTE, exp.LT, exp.LTE)
                or not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Literal)):
            continue
        try:
            value = float(node.expression.name)
        except ValueError:
            continue
        lower, upper = bounds.setdefault(node.this.sql().lower(), ([], []))
        if isinstance(node, (exp.GT, exp.GTE)):
            lower.append((value, type(node) is exp.GT))
        else:
            upper.append((value, type(node) is exp.LT))
    return any(low > high or (low == high and (ls or hs))
               for lower, upper in bounds.values() for low, ls in lower for high, hs in upper)


def evidence(tree, question):
    if not isinstance(tree, exp.Select):
        return set()
    result = set()
    if contradiction(tree):
        result.add("contradiction")
    if COUNT_RANK.search(question):
        order = tree.args.get("order")
        if not (tree.args.get("group") and order and order.find(exp.Count)):
            result.add("count_rank")
    result.update(continuation_conflicts(sql_text(tree), question) or ())
    return result


def projection_matches(tree, question):
    hits = set()
    for index, col in enumerate(tree.selects):
        if isinstance(col, exp.Column) and mentioned(col.name.replace("_", " "), question):
            hits.add(index)
    return hits


def safe_switch(draft, hint, question, arity, policy):
    old = simple(draft)
    new = validate_sql(hint)
    if len(new.selects) != arity:
        return False
    if policy == "first":
        return True
    if old is None:
        return False
    if "contradiction" in evidence(old, question):
        if isinstance(new, exp.Intersect) and re.search(r"\bboth\b", question, re.I):
            return True
    if not isinstance(new, exp.Select):
        return False
    if evidence(new, question) < evidence(old, question):
        return True
    if policy == "structure":
        return False
    # A projection-only change must add explicit question-to-column evidence.
    return (all(old.args.get(key) == new.args.get(key)
                for key in (old.args.keys() | new.args.keys()) - {"expressions"})
            and projection_matches(old, question) < projection_matches(new, question))


def path_and_filter(tree, schema, feedback, question, arity):
    where = tree.args.get("where")
    if not where:
        return
    filters = top_terms(where.this)
    anchors = {
        index for index, node in enumerate(filters)
        if isinstance(node, (exp.EQ, exp.Like)) and isinstance(node.this, exp.Column)
        and isinstance(node.expression, exp.Literal) and node.expression.is_string
        and mentioned(node.expression.name.strip("%"), question)
    }
    for _, sql, keys in foreign_key_edits(tree, schema, feedback, question, arity):
        repaired = simple(sql)
        if repaired is None:
            continue
        for location, candidate, proof in threshold_edits(repaired, question):
            yield "join/" + location, candidate, {**keys, **proof}
        # Preserve an explicit literal anchor while removing at most one
        # unsupported string predicate from the destination of the repaired join.
        if not anchors or "from" not in keys:
            continue
        target = keys["foreign_keys"][-1][2]
        alias = schema.sources(repaired).get(target, "").lower()
        for index, predicate in enumerate(top_terms(repaired.args["where"].this)):
            if (index not in anchors and isinstance(predicate, exp.EQ)
                    and isinstance(predicate.this, exp.Column) and predicate.this.table.lower() == alias
                    and isinstance(predicate.expression, exp.Literal) and predicate.expression.is_string
                    and not mentioned(predicate.expression.name, question)):
                remaining = [item.copy() for i, item in enumerate(top_terms(repaired.args["where"].this))
                             if i != index]
                repaired_copy = repaired.copy()
                repaired_copy.set("where", exp.Where(this=exp.and_(*remaining)))
                yield "join/where", sql_text(repaired_copy), {**keys, "removed": predicate.sql()}


class FollowupAgent(ExpansionAgent):
    def __init__(self, base, config, mode):
        super().__init__(base, config, "aggregate")
        self.mode = mode
        self.review_empty = False
        if mode in {"path_filter", "combined"}:
            self.generators += (path_and_filter,)

    def act(self, observation):
        draft = observation["issue_sql"]
        history = observation["history"]
        if not self.review_empty:
            action = super().act(observation)
            if (self.mode == "path_filter" or len(history) != 1 or action != {"tool": "submit", "sql": draft}
                    or observation["remaining"] < 2):
                return action
            feedback = history[0]["observation"]
            if not passed(feedback) or feedback["row_count"] != 0:
                return action
            self.review_empty = True
            return {"tool": "repair_hint"}

        checked = {e["sql"]: e["observation"] for e in history if e["tool"] == "check"}
        raw = next(e["observation"] for e in history if e["tool"] == "repair_hint")
        candidates = []
        for sql in raw:
            candidate = self.add(sql, "empty_hint")
            if candidate and candidate != draft and candidate not in candidates:
                candidates.append(candidate)
        question = observation["task"]["question"]
        arity = len(observation["example"]["columns"])
        policy = "conservative" if self.mode == "combined" else self.mode
        for sql in candidates:
            try:
                validate_sql(sql)
                eligible = policy != "keep" and safe_switch(draft, sql, question, arity, policy)
            except ValueError:
                continue
            if sql in checked:
                if eligible and passed(checked[sql]):
                    return {"tool": "submit", "sql": sql}
            elif observation["remaining"]:
                return {"tool": "check", "sql": sql}
        return {"tool": "submit", "sql": draft}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    mode = path.read_text(encoding="utf-8").strip()
    if mode not in {"first", "keep", "structure", "conservative", "path_filter", "combined"}:
        raise ValueError("Unknown follow-up mode")
    base, config = load_policy(path.with_name("step5-cpu_s0_d0.json"))
    return FollowupAgent(base, config, mode)

