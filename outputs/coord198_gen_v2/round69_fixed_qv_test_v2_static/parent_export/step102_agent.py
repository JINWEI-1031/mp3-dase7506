"""Reduce order-sensitive floating AVG accumulation for simple grouped queries."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step4_edits import Schema
from step101_agent import AggregateSuffix, create_agent as create_baseline


def stable_average(observation, sql):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or not tree.args.get('group')
            or len(list(tree.find_all(exp.Select))) != 1
            or any(tree.args.get(k) for k in ('joins', 'where', 'having', 'distinct', 'with', 'limit', 'offset'))):
        return None
    aggregates = list(tree.find_all(exp.AggFunc))
    if len(aggregates) != 1 or not isinstance(aggregates[0], exp.Avg):
        return None
    operand = aggregates[0].this
    source = tree.args.get('from')
    if not isinstance(operand, exp.Column) or source is None or not isinstance(source.this, exp.Table):
        return None
    table = source.this
    if operand.table.casefold() not in ('', table.alias_or_name.casefold()):
        return None
    tables = next((e['observation'] for e in observation['history'] if e['tool'] == 'schema'), None)
    if tables is None:
        return {'needs_schema': True}
    schema = Schema(tables)
    meta = schema.columns.get(table.name.casefold(), {}).get(operand.name.casefold())
    if meta is None or not any(t in meta['type'].upper() for t in ('REAL', 'FLOAT', 'DOUBLE')):
        return None
    # Magnitude order limits cancellation and makes accumulation independent of row insertion.
    inner = exp.select('*').from_(table.copy())
    inner.order_by(exp.Abs(this=operand.copy()), operand.copy(), copy=False)
    query = tree.copy()
    query.set('from', exp.From(this=inner.subquery(table.alias_or_name)))
    return {'sql': sql_text(query), 'kind': 'magnitude_ordered_grouped_average',
            'table': table.name, 'measure': operand.name,
            'limitation': 'Floating aggregation remains approximate; public equality is required.'}


class AverageSuffix(AggregateSuffix):
    def act(self, observation):
        # Reuse the checked-candidate state machine with a separate candidate builder.
        if self.candidate or not observation['remaining']:
            return super().act(observation)
        candidate = stable_average(observation, self.fallback['sql'])
        if candidate is None:
            return self.fallback
        if candidate.get('needs_schema'):
            return {'tool': 'schema'} if observation['remaining'] >= 2 else self.fallback
        self.evidence = candidate
        self.candidate = candidate['sql']
        return {'tool': 'check', 'sql': self.candidate}


class Agent:
    def __init__(self, baseline):
        self.baseline, self.suffix = baseline, None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action['tool'] != 'submit':
                return action
            self.suffix = AverageSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
