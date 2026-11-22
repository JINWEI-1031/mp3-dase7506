"""Resolve a single extremum against an explicitly conflicting public example."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step102_agent import create_agent as create_baseline


def extremum_candidates(observation, sql):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or observation['example']['row_count'] != 1
            or observation['example']['truncated']
            or any(tree.args.get(k) for k in ('joins', 'group', 'having', 'distinct', 'with', 'order', 'limit', 'offset'))):
        return None
    checked = [e['observation'] for e in observation['history']
               if e['tool'] == 'check' and identity(e['sql']) == identity(sql)]
    if not checked or not checked[-1].get('ok') or passed(checked[-1]):
        return None
    where = tree.args.get('where')
    source = tree.args.get('from')
    if where is None or not isinstance(where.this, exp.EQ) or source is None or not isinstance(source.this, exp.Table):
        return None
    left, right = where.this.this, where.this.expression
    if not isinstance(left, exp.Column) or not isinstance(right, exp.Subquery):
        return None
    inner = right.this
    if (not isinstance(inner, exp.Select) or len(inner.expressions) != 1
            or any(inner.args.get(k) for k in ('joins', 'where', 'group', 'having', 'distinct', 'with', 'order', 'limit', 'offset'))):
        return None
    aggregate = inner.expressions[0].unalias()
    inside = inner.args.get('from')
    if (not isinstance(aggregate, (exp.Max, exp.Min)) or not isinstance(aggregate.this, exp.Column)
            or inside is None or not isinstance(inside.this, exp.Table)
            or inside.this.name.casefold() != source.this.name.casefold()
            or aggregate.this.name.casefold() != left.name.casefold()
            or left.table.casefold() not in ('', source.this.alias_or_name.casefold())
            or aggregate.this.table.casefold() not in ('', inside.this.alias_or_name.casefold())
            or len(list(tree.find_all(exp.Select))) != 2):
        return None
    candidates = []
    for desc in (isinstance(aggregate, exp.Max), isinstance(aggregate, exp.Min)):
        candidate = tree.copy()
        candidate.set('where', None)
        candidate.order_by(exp.Ordered(this=left.copy(), desc=desc), copy=False)
        candidate.limit(1, copy=False)
        candidates.append(sql_text(candidate))
    return candidates


class ExtremumSuffix:
    def __init__(self, fallback):
        self.fallback, self.candidates = fallback, None
        self.index, self.pending, self.evidence = 0, None, None

    def act(self, observation):
        if self.pending is not None:
            result = observation['history'][-1]['observation']
            if passed(result):
                self.evidence['public_example_overrode_extremum'] = self.index == 2
                return {'tool': 'submit', 'sql': self.pending}
            self.pending = None
        if self.candidates is None:
            self.candidates = extremum_candidates(observation, self.fallback['sql']) or []
            if self.candidates:
                self.evidence = {'kind': 'public_example_extremum_disambiguation',
                                 'candidates': self.candidates,
                                 'limitation': 'Opposite direction conflicts with the original extremum wording.'}
        while self.index < len(self.candidates) and observation['remaining']:
            sql = self.candidates[self.index]
            self.index += 1
            previous = next((e['observation'] for e in observation['history']
                             if e['tool'] == 'check' and identity(e['sql']) == identity(sql)), None)
            if previous is not None:
                if passed(previous):
                    self.evidence['public_example_overrode_extremum'] = self.index == 2
                    return {'tool': 'submit', 'sql': sql}
                continue
            self.pending = sql
            return {'tool': 'check', 'sql': sql}
        return self.fallback


class Agent:
    def __init__(self, baseline):
        self.baseline, self.suffix = baseline, None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action['tool'] != 'submit':
                return action
            self.suffix = ExtremumSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
