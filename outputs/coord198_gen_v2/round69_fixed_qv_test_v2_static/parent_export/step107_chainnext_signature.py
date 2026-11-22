"""Conservative alias/quote equivalence; preserves self-join roles and scopes."""
from sqlglot import exp
from repair_rules import parse,sql_text
from step4_agent import identity

def signature(sql):
    tree=parse(sql)
    if tree is None:return identity(sql)
    for scope_index,select in enumerate(tree.find_all(exp.Select)):
        tables=[t for t in select.find_all(exp.Table) if t.find_ancestor(exp.Select) is select]
        aliases={t.alias_or_name.lower():f's{scope_index}r{i}' for i,t in enumerate(tables)}
        if len(aliases)!=len(tables):return identity(sql)
        # Correlated bindings require full lineage; conservatively leave them.
        for col in select.find_all(exp.Column):
            if col.find_ancestor(exp.Select) is not select:continue
            if col.table:
                if col.table.lower() not in aliases:return identity(sql)
                col.set('table',exp.to_identifier(aliases[col.table.lower()]))
        for table in tables:table.set('alias',exp.TableAlias(this=exp.to_identifier(aliases[table.alias_or_name.lower()])))
    for node in tree.find_all(exp.Identifier):node.set('this',node.this.lower());node.set('quoted',False)
    return sql_text(tree)
