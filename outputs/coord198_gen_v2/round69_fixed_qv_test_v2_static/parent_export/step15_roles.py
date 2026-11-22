"""Reuse grounded role edits with an independently ordered candidate pool."""
from collections import Counter

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step7_semantics import top_tables
from step10_bindings import role_edits


def relation_pool(obs, schema, fallback, mode):
    headings = obs["example"]["columns"]
    question = obs["task"]["question"]
    checks = {e["sql"]: e["observation"] for e in obs["history"] if e["tool"] == "check"}
    hints = [sql for e in obs["history"] if e["tool"] == "repair_hint" for sql in e["observation"]]
    parents = list(dict.fromkeys([*(sql for sql, result in checks.items() if result["ok"]), *hints, *checks]))
    seen = {identity(sql) for sql in (*checks, fallback)}
    pool, counts = {}, Counter()
    rule = "role_" + mode
    for sql in parents[:10]:
        parent = parse(sql)
        if not isinstance(parent, exp.Select):
            continue
        original_tables = set(top_tables(parent, schema).values())
        for family, query, priority in role_edits(parent, schema, headings, question, mode == "skeleton"):
            if family != rule:
                continue
            counts["raw"] += 1
            candidate = sql_text(query)
            key = identity(candidate)
            if key in seen:
                counts["already_checked_or_fallback"] += 1
                continue
            tables = set(top_tables(query, schema).values())
            item = {"sql": candidate, "rule": family, "parent": sql, "priority": priority,
                    "removed": sorted(original_tables - tables), "added": sorted(tables - original_tables),
                    "parent_checked_ok": bool(checks.get(sql, {}).get("ok"))}
            prior = pool.get(key)
            if prior:
                counts["duplicate"] += 1
                if item["priority"] > prior["priority"]:
                    pool[key] = item
                continue
            try:
                validate_sql(candidate)
            except ValueError:
                counts["invalid"] += 1
                continue
            pool[key] = item
    return list(pool.values()), dict(counts)
