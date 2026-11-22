"""Quantified plans anchored in checked predicates, with outer row grain retained."""
from dataclasses import replace
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity
from step4_edits import Schema
from step21_carry import conjuncts
from step29_scope import scope_signal, separate_witnesses
from step30_multiplicity import candidate_pool as witness_candidates, scope_plan
from step31_population import candidate_pool as population_candidates, population_plan
from step32_plan import compiled, context, cpu_plans, decompose, tree_plan

SCOPE_WORDS = re.compile(r"\b(?:both|but not|and not)\b", re.I)
FIXED_LIMIT = 6


def fragments(tree):
    """Expose one eligible predicate at a time while retaining the other conjuncts."""
    where = tree.args.get("where")
    parts = list(conjuncts(where.this)) if where else []
    yield tree, []
    if 2 <= len(parts) <= 5:
        for i, atom in enumerate(parts):
            piece = tree.copy().where(atom.copy(), append=False)
            yield piece, [p.copy() for j, p in enumerate(parts) if j != i]


def checked_source(obs, sql):
    return any(e["tool"] == "check" and e["observation"]["ok"] and identity(e["sql"]) == identity(sql)
               for e in obs["history"])


def discover(obs, baseline):
    tree = parse(baseline)
    if not isinstance(tree, exp.Select) or not checked_source(obs, baseline):
        return []
    result = []
    question = obs["task"]["question"]
    for piece, retained in fragments(tree):
        sql = sql_text(piece)
        pop = population_plan(obs, sql)
        if pop:
            # A literal in a retained conjunct must not repeat a population
            # qualifier. Moving one copy would leave a second, outer filter.
            inner = {v.this for v in pop.population.args["where"].find_all(exp.Literal)}
            if not any(v.this in inner for r in retained for v in r.find_all(exp.Literal)):
                result.append(("population", sql, pop, retained, pop.needs_schema))
    if SCOPE_WORDS.search(question):
        # Keep unrelated top-level predicates. A scope made from a subset
        # is valid only if every removed conjunct is independent of its values.
        where = tree.args.get("where")
        terms = scope_signal(obs, baseline)
        if where is not None and terms and terms.kind != "single_witness":
            values = set(terms.terms)
            atoms = list(conjuncts(where.this))
            active = [a for a in atoms if any(v.this in values for v in a.find_all(exp.Literal))]
            keep = [a for a in atoms if a not in active]
            if active and all(not p.find(exp.Select) for p in keep):
                sketch = tree.copy().where(*(a.copy() for a in active), append=False)
                sql = sql_text(sketch)
                proof = scope_plan(obs, sql, "preserve")
                if proof:
                    result.append(("witnesses", sql, proof, keep, True))
                elif not keep and len(tree.expressions) == 1:
                    measure = tree.expressions[0].unalias()
                    if isinstance(measure, exp.Count) and isinstance(measure.this, exp.Distinct):
                        result.append(("counted_group_witnesses", sql, terms, [], True))
    return result


def plans(obs, found, schema):
    for kind, sketch, proof, retained, needs_schema in found:
        if needs_schema and schema is None:
            continue
        if kind == "population":
            families = [(proof, True, "quantified comparison")]
            if proof.boundary:
                if proof.quantifier == "any":
                    # Date wording can support an earliest boundary or an
                    # existential reading. Keep the latter in the measured pool.
                    families.append((replace(proof, boundary=None), False, "existential reading"))
                else:
                    families.append((replace(proof, boundary=None), False, "strict SQL ALL"))
            candidates = [(c, selected, mode) for variant, selected, mode in families
                          for c in population_candidates(obs, sketch, variant, schema)]
        elif kind == "counted_group_witnesses":
            candidates = [({"sql": sql_text(query), "evidence": evidence}, True,
                           "independent groups for COUNT DISTINCT")
                          for query, _, rule, evidence in separate_witnesses(
                              obs, parse(sketch), Schema(schema), proof)
                          if rule == "independent_group_witnesses"]
        else:
            candidates = [(c, True, "root-correlated independent witnesses")
                          for c in witness_candidates(obs, sketch, schema, proof)]
        for candidate, selected, mode in candidates:
            query = parse(candidate["sql"])
            if not isinstance(query, exp.Select):
                continue
            if retained:
                query.where(*(p.copy() for p in retained), copy=False)
            yield tree_plan(query, {"kind": kind, "mode": mode, "evidence": candidate["evidence"],
                                    "retained_conjuncts": [sql_text(p) for p in retained],
                                    "eligible": selected})


def schedule_pool(obs):
    """Fix the same slot-seven projection pool for both scheduling policies."""
    c = context(obs)
    draft = parse(obs["issue_sql"])
    if not isinstance(draft, exp.Select) or not draft.args.get("order") or not draft.args.get("limit"):
        return []
    if not checked_source(obs, obs["issue_sql"]) or len(draft.expressions) <= len(c["columns"]):
        return []
    # Match the frozen step32 projection plan where an executed ranked
    # draft outputs extra columns. Verify all other relational slots match.
    old = decompose(draft)
    if old is None:
        return []
    result = []
    previous = {identity(e["sql"]) for e in obs["history"] if e["tool"] == "check"}
    for candidate in compiled(c, draft, cpu_plans(c, draft, False)):
        if candidate["basis"] != "projection matched public headings" or identity(candidate["sql"]) in previous:
            continue
        new = decompose(parse(candidate["sql"]))
        if all(value == getattr(new, field) for field, value in vars(old).items() if field != "output"):
            result.append(candidate | {"basis": {"kind": "fixed_slot_projection", "eligible": True,
                                                "evidence": "executed ranked draft and exact output headings"}})
    return result[:FIXED_LIMIT]

