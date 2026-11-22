"""Local, single-scope output contracts; no answer- or database-based routing."""
import re
from sqlglot import exp
from repair_rules import parse


def projection(sql, obs, schema):
    tree = parse(sql)
    headers = obs['example']['columns']
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return None, 'nested_or_set_scope'
    if not headers or not all(re.fullmatch(r'[A-Za-z_]\w*', h) for h in headers):
        return None, 'non_column_contract'
    if len(tree.selects) <= len(headers):
        return None, 'no_excess_projection'
    tables = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}
    metadata = {t['name'].lower(): t for t in schema}
    if any(t not in metadata for t in tables.values()):
        return None, 'unknown_relation'
    keep = []
    for h in headers:
        matches = [x for x in tree.selects if isinstance(x.unalias(), exp.Column)
                   and x.unalias().name.lower() == h.lower()]
        if len(matches) != 1:
            return None, 'ambiguous_output_role'
        keep.append(matches[0])
    removed = [x for x in tree.selects if x not in keep]
    if not all(isinstance(x.unalias(), exp.AggFunc) for x in removed):
        return None, 'nonaggregate_extra_output'
    aliases = {x.alias.lower(): x.unalias() for x in removed if x.alias}
    # Removing an aggregate from SELECT does not remove its grouping or ranking.
    for c in list(tree.find_all(exp.Column)):
        if not c.table and c.name.lower() in aliases:
            c.replace(aliases[c.name.lower()].copy())
    for c in list(tree.find_all(exp.Column)):
        if c.table:
            continue
        owners = [a for a, t in tables.items() if any(z['name'].lower() == c.name.lower()
                  for z in metadata[t]['columns'])]
        if len(owners) > 1:
            preferred = [a for a in owners if c.name.lower().startswith(tables[a] + '_')]
            if len(preferred) != 1:
                return None, 'ambiguous_column_owner'
            owners = preferred
        if len(owners) != 1:
            return None, 'unresolved_column'
        c.set('table', exp.to_identifier(owners[0]))
    # Keep original expression objects after qualification, not a rebuilt plan.
    tree.set('expressions', keep)
    return tree.sql(dialect='sqlite'), 'excess_aggregate_output_and_schema_owner'
