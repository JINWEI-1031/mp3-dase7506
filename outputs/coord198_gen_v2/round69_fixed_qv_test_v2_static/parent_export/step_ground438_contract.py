"""Scope-resolved equality for plan approval, not string/alias equality."""
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_space435_common import structural_key

def semantic_key(sql,schema):
    tree=parse(sql)
    if tree is None:return None
    for select in tree.find_all(exp.Select):
        src=sources(select,schema)
        for c in select.find_all(exp.Column):
            if c.find_ancestor(exp.Select) is not select or c.table:continue
            owners=[a for a,t in src.items() if any(f['name'].lower()==c.name.lower() for f in t['columns'])]
            if len(owners)!=1:return None
            c.set('table',exp.to_identifier(owners[0]))
    return structural_key(sql_text(tree),schema)
