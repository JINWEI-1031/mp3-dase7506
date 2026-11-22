"""Extract and retrieve bounded edit templates from the released training memory."""
from collections import Counter
import re

from sqlglot import exp

from repair_rules import error_message, parse
from step4_edits import COMPARISONS, simple


def terms(text, mask=False):
    if mask:
        text = re.sub(r"""'[^']*'|"[^"]*"|\b\d+(?:\.\d+)?\b""", " literal ", text)
    return set(re.findall(r"[a-z][a-z0-9]*", text.lower()))


def shape(sql):
    tree = parse(sql)
    if tree is None:
        return {"unparsed": 1}
    return dict(Counter(node.key for node in tree.walk()
                        if not isinstance(node, (exp.Identifier, exp.TableAlias, exp.Alias))))


def category(feedback):
    return (error_message(feedback).partition(":")[0] if not feedback["ok"]
            else ("public match" if feedback["matches_example"] else "public mismatch"))


def abstract_edits(draft, gold):
    old, new = simple(draft), parse(gold)
    if old is None or new is None:
        return []
    edits = []
    select = new if isinstance(new, exp.Select) else new.this
    if not isinstance(select, exp.Select):
        return []
    old_joins, new_joins = list(old.find_all(exp.Join)), list(new.find_all(exp.Join))
    if len(new_joins) > len(old_joins):
        edits.append("add_join")
    if old_joins and new_joins and [j.args.get("on") for j in old_joins] != [
            j.args.get("on") for j in new_joins]:
        edits.append("join_key")
    if ((isinstance(new, exp.Except) or any(isinstance(n, (exp.Exists, exp.In))
                                         for n in new.find_all(exp.Not)
                                         for n in [n.this]))
            and old.find(exp.Is)):
        edits.append("anti_join")
    old_order, new_order = old.args.get("order"), select.args.get("order")
    if new_order and (old_order is None or not old_order.find(exp.AggFunc)):
        if new_order.find(exp.Count) and select.args.get("group"):
            edits.append("frequency")
        if new_order.find(exp.Sum) and select.args.get("group"):
            edits.append("group_sum")
    comparisons = tuple(COMPARISONS.values())
    before = {node.expression.this: type(node) for node in old.walk()
              if isinstance(node, comparisons) and isinstance(node.expression, exp.Literal)}
    if any(isinstance(node, comparisons) and isinstance(node.expression, exp.Literal)
           and node.expression.this in before and before[node.expression.this] != type(node)
           for node in new.walk()):
        edits.append("comparator")
    return edits


def record(item, error=None, db=None):
    return {"id": item["id"], "db": db, "words": sorted(terms(item["question"])),
            "masked": sorted(terms(item["question"], mask=True)),
            "shape": shape(item["issue_sql"]), "error": error,
            "edits": abstract_edits(item["issue_sql"], item["gold_sql"])}


def distance_similarity(left, right):
    keys = left.keys() | right.keys()
    return sum(min(left.get(k, 0), right.get(k, 0)) for k in keys) / max(
        1, sum(max(left.get(k, 0), right.get(k, 0)) for k in keys))


def retrieve_local(index, question, draft, error, excluded, mode):
    query = terms(question, mask=mode == "hybrid")
    skeleton = shape(draft)
    ranked = []
    for item in index:
        if item["id"] in excluded or not item["edits"]:
            continue
        words = item["masked" if mode == "hybrid" else "words"]
        similarity = len(query.intersection(words)) / max(1, len(query.union(words)))
        if mode == "hybrid":
            similarity += .55 * distance_similarity(skeleton, item["shape"])
            similarity += .30 * (item["error"] == error)
        ranked.append((similarity, item["id"], item))
    return [item for _, _, item in sorted(ranked, key=lambda row: (-row[0], row[1]))[:8]]


def official_query(question, draft, feedback, mode):
    if mode == "error":
        question += " " + (error_message(feedback) or category(feedback))
    if mode == "structure":
        question += " " + " ".join(key for key, count in sorted(shape(draft).items())
                                   for _ in range(min(count, 3)))
    return question[:2000]
