"""Targeted CPU experiments. Inputs are limited to causal public observations."""
import json
from pathlib import Path
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import sql_text
from step4_agent import identity, passed
from step4_edits import Schema, simple, threshold_edits
from step5_agent import load_policy
from step5x_agent import ExpansionAgent
from step5x_edits import foreign_key_edits
from step5y_agent import evidence, mentioned, path_and_filter, top_terms, words


FAMILIES = ("projection", "predicate", "rank_scope", "contradiction")


def projection_gap(col, question):
    return (isinstance(col, exp.Column) and len(col.name) > 2
            and mentioned(col.name.replace("_", " "), question))


def explicit_predicates(tree, question):
    where = tree.args.get("where")
    if where:
        for term in top_terms(where.this):
            if (isinstance(term, (exp.EQ, exp.Like)) and isinstance(term.this, exp.Column)
                    and isinstance(term.expression, exp.Literal) and term.expression.is_string):
                value = term.expression.name.strip("%")
                if value and re.search(r"\bcontain(?:s|ing)?\s+['\"]?"
                                       + re.escape(value) + r"\b", question, re.I):
                    candidate = exp.Like(this=term.this.copy(), expression=exp.Literal.string(f"%{value}%"))
                    if candidate != term:
                        yield term, candidate, "explicit contains"


def conflicts(draft, question):
    old = simple(draft)
    if old is None:
        return []
    result = evidence(old, question) & {"contradiction", "count_rank", "direction", "top"}
    if list(explicit_predicates(old, question)) or list(threshold_edits(old, question)):
        result.add("explicit_predicate")
    return sorted(result)


def changed_only(old, new, allowed):
    return all(old.args.get(k) == new.args.get(k)
               for k in (old.args.keys() | new.args.keys()) - set(allowed))


def switch_proof(draft, hint, question, arity):
    old, new = simple(draft), simple(hint)
    if old is None or new is None or len(new.selects) != arity:
        return {}
    if not changed_only(old, new, {"expressions"}) or len(old.selects) != arity:
        pass
    else:
        differences = [(a, b) for a, b in zip(old.selects, new.selects) if a != b]
        if differences and all(isinstance(a, exp.Column) and isinstance(b, exp.Column)
                               and a.table.lower() == b.table.lower()
                               and not projection_gap(a, question) and projection_gap(b, question)
                               for a, b in differences):
            return {"projection": [[a.sql(), b.sql()] for a, b in differences]}
    if changed_only(old, new, {"where"}):
        for before, after, reason in explicit_predicates(old, question):
            fixed = old.copy()
            terms = top_terms(fixed.args["where"].this)
            next(term for term in terms if term == before).replace(after)
            if fixed == new:
                return {"predicate": [before.sql(), after.sql(), reason]}
        for _, sql, proof in threshold_edits(old, question):
            if simple(sql) == new:
                return {"predicate": proof}
    if changed_only(old, new, {"order", "limit", "group"}):
        # Aggregate rank must apply to the same input relation and predicates.
        before, after = evidence(old, question), evidence(new, question)
        if after < before and before - after <= {"count_rank", "direction", "top"}:
            if (old.args.get("group") == new.args.get("group")
                    or (not old.args.get("group") and new.args.get("group")
                        and new.args["group"].expressions == old.selects
                        and all(isinstance(col, exp.Column) for col in old.selects))):
                return {"rank_scope": sorted(before - after)}
    # A two-cohort self join can replace contradictory numeric bounds on one row.
    if ("contradiction" in evidence(old, question) and "both" in words(question)
            and not old.args.get("joins") and len(old.selects) == 1
            and isinstance(old.selects[0], exp.Column)
            and changed_only(old, new, {"from", "joins", "where", "expressions", "distinct"})):
        joins = new.args.get("joins") or []
        left, prior = new.args.get("from"), old.args.get("from")
        where = new.args.get("where")
        if (len(joins) == 1 and left and prior and where and new.args.get("distinct")
                and all(isinstance(t, exp.Table) for t in (left.this, prior.this, joins[0].this))
                and left.this.name.lower() == prior.this.name.lower() == joins[0].this.name.lower()
                and not joins[0].side and joins[0].kind in ("", "INNER")):
            key = old.selects[0].name.lower()
            a, b = left.this.alias_or_name.lower(), joins[0].this.alias_or_name.lower()
            on = joins[0].args.get("on")
            terms = top_terms(where.this)
            original = top_terms(old.args["where"].this)
            def bound_signature(items):
                return sorted((type(t).__name__, t.this.name.lower(), t.expression.name)
                              for t in items if type(t) in (exp.GT, exp.GTE, exp.LT, exp.LTE)
                              and isinstance(t.this, exp.Column) and isinstance(t.expression, exp.Literal))
            if (a != b and isinstance(on, exp.EQ)
                    and all(isinstance(c, exp.Column) and c.name.lower() == key
                            for c in (on.this, on.expression, new.selects[0]))
                    and {on.this.table.lower(), on.expression.table.lower()} == {a, b}
                    and new.selects[0].table.lower() in {a, b}
                    and len(terms) == len(original) == len(bound_signature(terms)) == 2
                    and bound_signature(terms) == bound_signature(original)
                    and {t.this.table.lower() for t in terms} == {a, b}):
                return {"contradiction": [t.sql() for t in terms]}
    return {}


def binding(schema, tree, col):
    options = [table for table, alias in schema.sources(tree).items()
               if col.name.lower() in schema.columns[table]
               and (not col.table or col.table.lower() == alias.lower())]
    return (options[0], col.name.lower()) if len(options) == 1 else None


def predicate_bindings(schema, tree):
    where = tree.args.get("where")
    return [binding(schema, tree, c) for c in where.find_all(exp.Column)] if where else []


def filter_variants(tree, schema, question):
    where = tree.args.get("where")
    if not where or None in predicate_bindings(schema, tree):
        return
    for old, new, reason in explicit_predicates(tree, question):
        fixed = tree.copy()
        next(t for t in top_terms(fixed.args["where"].this) if t == old).replace(new)
        yield sql_text(fixed), {"filter": reason, "from_predicate": old.sql()}
    for index, term in enumerate(top_terms(where.this)):
        if (type(term) in (exp.GT, exp.GTE, exp.LT, exp.LTE)
                and isinstance(term.this, exp.Column) and isinstance(term.expression, exp.Literal)):
            in_scope = exp.Select(expressions=[exp.Star()], where=exp.Where(this=term.copy()))
            for _, sql, proof in threshold_edits(in_scope, question):
                fixed = tree.copy()
                top_terms(fixed.args["where"].this)[index].replace(simple(sql).args["where"].this.copy())
                yield sql_text(fixed), proof
        # A quoted entity value on a non-name attribute can move to a unique
        # name attribute on the same relation if the question explicitly says named.
        if (isinstance(term, exp.EQ) and isinstance(term.this, exp.Column)
                and isinstance(term.expression, exp.Literal) and term.expression.is_string
                and re.search(r"\bnamed\s+['\"]?" + re.escape(term.expression.name)
                              + r"\b", question, re.I)):
            table, col = binding(schema, tree, term.this)
            names = [name for name in schema.columns[table] if name == "name" or name.endswith("_name")]
            if len(names) == 1 and names[0] != col:
                fixed = tree.copy()
                top_terms(fixed.args["where"].this)[index].set(
                    "this", schema.column(table, names[0], schema.sources(tree)[table]))
                yield sql_text(fixed), {"filter": "named entity", "attribute": [table, col, names[0]]}


class TargetedAgent(ExpansionAgent):
    def __init__(self, base, policy, settings):
        super().__init__(base, policy, "aggregate")
        self.settings = settings
        self.empty_active = False
        self.audit["empty"] = {"reason": "no draft check"}
        self.audit["path"] = {"reason": "baseline did not reach unchecked fallback"}

    def act(self, obs):
        history, draft = obs["history"], obs["issue_sql"]
        mode = self.settings.get("empty")
        if mode and len(history) == 1:
            outcome = history[0]["observation"]
            reason = ("draft failed public" if not passed(outcome) else
                      "draft public nonempty" if outcome["row_count"] != 0 else
                      "no budget for hint and check" if obs["remaining"] < 2 else None)
            flags = conflicts(draft, obs["task"]["question"]) if reason is None else []
            if reason is None and self.settings.get("trigger") == "conflict" and not flags:
                reason = "no explicit draft conflict"
            self.audit["empty"] = {"reason": reason or "requested hints", "conflicts": flags}
            if reason is None:
                self.empty_active = True
                return {"tool": "repair_hint"}
        if self.empty_active:
            return self.empty_action(obs)
        action = super().act(obs)
        path_mode = self.settings.get("path")
        if not path_mode or action["tool"] != "submit":
            return action
        checks = {e["sql"]: e["observation"] for e in history if e["tool"] == "check"}
        if any(passed(v) for v in checks.values()):
            self.audit["path"]["reason"] = "existing public pass"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            self.audit["path"]["reason"] = "no schema at baseline fallback"
            return action
        if not self.audit["path"].get("evaluated"):
            self.path_generate(obs, list(checks.items())[:2], Schema(tables), path_mode)
        if obs["remaining"] and self.pending:
            return {"tool": "check", "sql": self.pending.pop(0)}
        return action

    def empty_action(self, obs):
        history, draft = obs["history"], obs["issue_sql"]
        hints = next(e["observation"] for e in history if e["tool"] == "repair_hint")
        checked = {e["sql"]: e["observation"] for e in history if e["tool"] == "check"}
        q, arity = obs["task"]["question"], len(obs["example"]["columns"])
        details, valid = [], []
        audit = self.audit["empty"]
        audit["hints"] = details
        for raw in hints:
            sql = self.add(raw, "targeted_empty_hint", parent=draft)
            entry = {"sql": sql or raw}
            details.append(entry)
            if not sql or sql == draft or sql in valid:
                entry["reason"] = "duplicate or unusable"
                continue
            try:
                validate_sql(sql)
            except ValueError:
                entry["reason"] = "invalid SQL"
                continue
            proof = switch_proof(draft, sql, q, arity)
            valid.append(sql)
            entry["proof"] = proof
            if sql not in checked:
                entry["reason"] = "generated not checked"
            elif not passed(checked[sql]):
                entry["reason"] = "checked public fail"
            else:
                entry["reason"] = ("checked eligible evidence" if
                                   set(proof) & set(self.settings.get("families", FAMILIES))
                                   else "checked public pass, insufficient evidence")
        limit = self.settings.get("limit", 8)
        for sql in valid[:limit]:
            if obs["remaining"] and sql not in checked:
                return {"tool": "check", "sql": sql}
        for entry in details:
            if entry.get("reason") == "generated not checked":
                entry["reason"] = "generated not checked: cap" if entry["sql"] not in valid[:limit] else "generated not checked: budget"
        select = self.settings["empty"]
        chosen = draft
        for entry in details:
            sql = entry["sql"]
            if (entry["reason"].startswith("checked") and sql in checked and passed(checked[sql])
                    and (select == "first" or (select == "safe" and entry["reason"] == "checked eligible evidence"))):
                chosen = sql
                entry["reason"] = "selected: " + ("first public pass" if select == "first" else "explicit repair evidence")
                break
        audit["reason"] = ("switched" if chosen != draft else "keep arm" if select == "keep" else
                           "no eligible checked pass" if valid else "no usable distinct hint")
        for entry in details:
            if entry["reason"] == "checked eligible evidence":
                entry["reason"] = "checked eligible, retained draft or earlier selection"
        return {"tool": "submit", "sql": chosen}

    def path_generate(self, obs, parents, schema, mode):
        q, arity = obs["task"]["question"], len(obs["example"]["columns"])
        audit = self.audit["path"] = {"evaluated": True, "parents": [], "reason": "no viable candidate"}
        for parent, feedback in parents:
            info = {"sql": parent, "events": []}
            audit["parents"].append(info)
            events = info["events"]
            tree = simple(parent)
            if tree is None or tree.args.get("with") or not tree.args.get("where"):
                events.append("no simple filtered select")
                continue
            where = tree.args["where"]
            relevant = any(isinstance(t, exp.Literal) and mentioned(t.name.strip("%"), q)
                           for t in where.find_all(exp.Literal))
            if not relevant or (not tree.args.get("joins") and not any(
                    binding(schema, tree, c) is None for c in where.find_all(exp.Column))):
                events.append("no cross-table and question-anchored filter opportunity")
                continue
            paths = list(foreign_key_edits(tree, schema, feedback, q, arity))
            filters = list(filter_variants(tree, schema, q))
            if not paths:
                events.append("no supported new FK path or unchanged keys")
            if not filters:
                events.append("no explicit filter correction on an unambiguous binding")
            generated = []
            if mode == "path":
                generated = [(sql, proof) for _, sql, proof in paths]
            elif mode == "filter":
                generated = filters
            elif mode == "legacy":
                generated = [(sql, proof) for _, sql, proof in path_and_filter(tree, schema, feedback, q, arity)]
            else:
                for _, sql, proof in paths:
                    updated = simple(sql)
                    if updated is None:
                        continue
                    before, after = predicate_bindings(schema, tree), predicate_bindings(schema, updated)
                    if (None in after or len(before) != len(after)
                            or any(a is not None and a != b for a, b in zip(before, after))):
                        events.append("FK edit changed or left an ambiguous predicate binding")
                        continue
                    options = list(filter_variants(updated, schema, q))
                    if not options:
                        events.append("no explicit filter correction after FK repair")
                    generated.extend((sql, {**proof, **filter_proof}) for sql, filter_proof in options)
            for sql, proof in generated:
                candidate = simple(sql)
                if candidate is None or None in predicate_bindings(schema, candidate):
                    events.append("unresolved predicate binding")
                    continue
                if identity(sql) in self.audit["candidates"]:
                    events.append("duplicate candidate")
                    continue
                try:
                    validate_sql(sql)
                except ValueError:
                    events.append("invalid SQL")
                    continue
                self.pending.append(self.add(sql, "targeted_" + mode, evidence=proof, parent=parent))
                audit["reason"] = "generated candidates"
                if len(self.pending) == 2:
                    return


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    base, policy = load_policy(path.with_name("step5-cpu_s0_d0.json"))
    return TargetedAgent(base, policy, json.loads(path.read_text(encoding="utf-8")))
