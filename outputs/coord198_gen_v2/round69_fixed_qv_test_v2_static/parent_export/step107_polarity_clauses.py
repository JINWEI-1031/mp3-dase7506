"""Clause-local contradiction certificates, preserving the surrounding query."""
import re
from sqlglot import exp
from repair_rules import parse


def _tokens(text):
    return {w.rstrip('s') for w in re.findall(r'[a-z]+', text.lower())}


def exact_exclusions(tree, question):
    results = []
    where=tree.args.get('where')
    def conjunction(e):
        return conjunction(e.this)+conjunction(e.expression) if isinstance(e,exp.And) else [e]
    leaves=conjunction(where.this) if where else []
    for n in tree.find_all(exp.Not):
        if not any(n is leaf for leaf in leaves):
            continue
        like = n.this
        if not isinstance(like, exp.Like) or not isinstance(like.this, exp.Column) or not isinstance(like.expression, exp.Literal) or not like.expression.is_string:
            continue
        pattern = like.expression.this
        if not (pattern.startswith('%') and pattern.endswith('%')):
            continue
        value = pattern[1:-1]
        if not value or '%' in value or '_' in value:
            continue
        # Exact copular exclusion; contains/LIKE/substring language never licenses it.
        matches = list(re.finditer(r"([a-z_]+)\s+is\s+not\s+'((?:''|[^'])*)'", question.lower()))
        matched = [m for m in matches if m[2].replace("''", "'") == value.lower()
                   and _tokens(m[1]) & _tokens(like.this.name)]
        if len(matched) == 1:
            results.append((n, like.this, value))
    return results


def signals(sql, obs):
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return []
    result = []
    if exact_exclusions(tree, obs['task']['question']):
        result.append('exact_attribute_exclusion_vs_substring')
    last_errors = [str(e.get('observation', {}).get('error') or '').lower()
                   for e in obs['history'] if e['tool'] in ('check', 'execute')]
    if any('no such column' in e for e in last_errors):
        result.append('observed_unknown_column')
    return result


def candidates(sql, obs, schema):
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return [], ['nonflat_scope']
    tables = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}
    metadata = {t['name'].lower(): t for t in schema}
    if any(t not in metadata for t in tables.values()):
        return [], ['unknown_relation']
    def owners(c):
        return [a for a, t in tables.items() if (not c.table or c.table.lower() == a)
                and any(z['name'].lower() == c.name.lower() for z in metadata[t]['columns'])]
    output = []; rejections = []
    headers = obs['example']['columns']
    if len(headers) == len(tree.selects):
        replacements = []
        for expression, header in zip(tree.selects, headers):
            c = expression.unalias()
            if not isinstance(c, exp.Column) or owners(c) or not re.fullmatch(r'[A-Za-z_]\w*', header):
                continue
            possible = [a for a, t in tables.items() if (not c.table or c.table.lower() == a)
                        and any(z['name'].lower() == header.lower() for z in metadata[t]['columns'])]
            if len(possible) != 1 or not (_tokens(header) & _tokens(obs['task']['question'])):
                rejections.append('ambiguous_or_ungrounded_projection_role'); continue
            replacements.append((tree.selects.index(expression), exp.column(header, table=possible[0])))
        if replacements:
            repaired = tree.copy(); expressions = repaired.selects
            for i, c in replacements:
                expressions[i] = c
            repaired.set('expressions', expressions)
            output.append({'sql': repaired.sql(dialect='sqlite'), 'source': 'unknown_projection_contract',
                           'evidence': {'changed_columns': len(replacements), 'headers': headers},
                           'preserved': ['where', 'joins', 'group', 'having', 'order', 'limit', 'distinct']})
    exclusions = exact_exclusions(tree, obs['task']['question'])
    if len(exclusions) == 1:
        target, c, value = exclusions[0]; bound = owners(c)
        if len(bound) == 1:
            field = next(z for z in metadata[tables[bound[0]]]['columns'] if z['name'].lower() == c.name.lower())
            if any(t in field['type'].lower() for t in ('text', 'char')):
                repaired = tree.copy()
                # AST address by ordinal, so no other clause/literal is rewritten.
                ordinal = list(tree.find_all(exp.Not)).index(target)
                old = list(repaired.find_all(exp.Not))[ordinal]
                old.replace(exp.NEQ(this=old.this.this.copy(), expression=exp.Literal.string(value)))
                output.append({'sql': repaired.sql(dialect='sqlite'), 'source': 'exact_attribute_exclusion',
                               'evidence': {'column': c.sql(), 'copula': 'is not', 'quoted_value': value},
                               'preserved': ['all_other_predicates', 'joins', 'group', 'having', 'order', 'limit', 'distinct', 'null_exclusion']})
        else:
            rejections.append('ambiguous_exclusion_owner')
    return output, rejections
