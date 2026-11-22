"""Complete a strict public subset with one uniquely supported categorical branch."""
from collections import Counter
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import passed
from step4_edits import Schema
from step104_agent import Agent as WitnessAgent, create_agent as create_baseline


def completion_plan(observation):
    expected = observation['example']
    if (observation['remaining'] < 2 or expected['truncated'] or not 2 <= len(expected['rows']) <= 8
            or any(count != 1 for count in Counter(tuple(r) for r in expected['rows']).values())):
        return None
    checks = [e for e in observation['history'] if e['tool'] == 'check']
    if any(passed(e['observation']) for e in checks):
        return None
    tables = next((e['observation'] for e in observation['history'] if e['tool'] == 'schema'), None)
    if tables is None:
        return None
    schema = Schema(tables)
    wanted = Counter(tuple(r) for r in expected['rows'])
    for event in checks:
        result = event['observation']
        if (not result['ok'] or result['truncated'] or not result['rows']
                or not Counter(tuple(r) for r in result['rows']) < wanted):
            continue
        tree = parse(event['sql'])
        if (not isinstance(tree, exp.Select) or any(tree.args.get(k) for k in ('joins', 'group', 'having', 'order', 'limit', 'offset', 'with', 'distinct'))
                or not tree.args.get('where') or len(tree.expressions) != len(expected['columns'])
                or any(not isinstance(c, exp.Column) for c in tree.expressions)
                or [c.name.casefold() for c in tree.expressions] != [h.casefold() for h in expected['columns']]):
            continue
        root = tree.args.get('from')
        if root is None or not isinstance(root.this, exp.Table):
            continue
        condition = tree.args['where'].this
        if not isinstance(condition, (exp.EQ, exp.In)) or not isinstance(condition.this, exp.Column):
            continue
        subqueries = list(condition.find_all(exp.Select))
        if len(subqueries) != 1:
            continue
        inner = subqueries[0]
        inside = inner.args.get('from')
        if (inside is None or not isinstance(inside.this, exp.Table)
                or inside.this.name.casefold() != root.this.name.casefold()
                or not inner.args.get('group') or not inner.args.get('order')
                or not inner.args.get('limit') or inner.args['limit'].expression != exp.Literal.number(1)
                or any(inner.args.get(k) for k in ('joins', 'where', 'having', 'offset', 'with'))
                or len(inner.args['group'].expressions) != 1
                or not isinstance(inner.args['group'].expressions[0], exp.Column)
                or inner.args['group'].expressions[0].name.casefold() != condition.this.name.casefold()
                or len(list(inner.find_all(exp.AggFunc))) != 1
                or not isinstance(next(inner.find_all(exp.AggFunc)), exp.Count)):
            continue
        name, alias = root.this.name.casefold(), root.this.alias_or_name
        if name not in schema.tables or any(c.table.casefold() not in ('', alias.casefold()) for c in tree.expressions):
            continue
        fields = [field for field, meta in schema.columns[name].items()
                  if not meta['pk'] and re.search(r'CHAR|TEXT', meta['type'], re.I)
                  and re.search(r'(?:role|type|status|category).*code$', field, re.I)]
        if not 1 <= len(fields) <= 3:
            continue
        probe = tree.copy()
        probe.set('where', None)
        flag = exp.Case(ifs=[exp.If(this=condition.copy(), true=exp.Literal.number(1))], default=exp.Literal.number(0))
        probe.set('expressions', [*[c.copy() for c in tree.expressions],
                                 *[schema.column(name, f, alias) for f in fields], flag.as_('in_ranked_group')])
        probe.limit(9, copy=False)
        return {'query': tree, 'probe': sql_text(probe), 'fields': fields, 'alias': alias,
                'arity': len(tree.expressions), 'table': name}
    return None


def completion_candidate(plan, feedback, expected):
    if not feedback['ok'] or feedback['truncated'] or not feedback['rows']:
        return None
    rows, arity = feedback['rows'], plan['arity']
    wanted = {tuple(r) for r in expected}
    base = {tuple(r[:arity]) for r in rows if r[-1] == 1}
    if not base < wanted or not wanted <= {tuple(r[:arity]) for r in rows}:
        return None
    candidates = []
    for index, field in enumerate(plan['fields'], arity):
        values = {r[index] for r in rows if isinstance(r[index], str) and r[index]}
        for value in values:
            selected = {tuple(r[:arity]) for r in rows if r[-1] == 1 or r[index] == value}
            if selected == wanted:
                candidates.append((field, value))
    if len(candidates) != 1:
        return None
    field, value = candidates[0]
    query = plan['query'].copy()
    query.set('distinct', exp.Distinct())
    query.set('where', exp.Where(this=exp.or_(query.args['where'].this.copy(), exp.EQ(
        this=exp.column(field, table=plan['alias']), expression=exp.Literal.string(value)))))
    return {'sql': sql_text(query), 'kind': 'unique_public_categorical_completion',
            'table': plan['table'], 'field': field, 'value': value,
            'witness_count': len(rows), 'base_count': len(base), 'expected_count': len(wanted),
            'limitation': 'The added category is inferred from public rows and may be absent from the wording.'}


class Agent(WitnessAgent):
    plan_builder = staticmethod(completion_plan)
    candidate_builder = staticmethod(completion_candidate)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
