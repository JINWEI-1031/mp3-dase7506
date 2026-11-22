"""Review a grouped rank query whose bare output is not determined by its grouping key."""
import re

from sqlglot import exp
from repair_rules import parse
from step4_agent import identity, passed, continuation_conflicts
from step4_edits import Schema
from step38.event_grain import bind_column, join_graph
from step63_agent import generate
from step106_agent import create_agent as create_baseline


def rank_shape(observation, sql):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or len(tree.expressions) != 1
            or not isinstance(tree.expressions[0].unalias(), exp.Column)
            or not tree.args.get('group') or not tree.args.get('having')
            or any(tree.args.get(k) for k in ('with', 'offset', 'where'))
            or not re.search(r'\b(?:most|fewest|least)\b', observation['task']['question'], re.I)):
        return None
    having = tree.args['having'].this
    if (not isinstance(having, exp.EQ) or not isinstance(having.this, exp.Count)
            or not isinstance(having.expression, exp.Subquery)):
        return None
    inner = having.expression.this
    if (not isinstance(inner, exp.Select) or not inner.args.get('group')
            or not inner.args.get('order') or not inner.args.get('limit')
            or inner.args['limit'].expression != exp.Literal.number(1)):
        return None
    return tree


def undetermined_output(tree, tables):
    if not isinstance(tree, exp.Select) or not tree.args.get('group'):
        return None
    root = tree.args.get('from')
    if root is None or not isinstance(root.this, exp.Table):
        return None
    schema = Schema(tables)
    flat = exp.select('*').from_(root.this.copy())
    for join in tree.args.get('joins', []):
        flat.append('joins', join.copy())
    sources = schema.sources(flat)
    if not sources or any(not isinstance(c, exp.Column) for c in tree.args['group'].expressions):
        return None
    links = join_graph(flat, schema, sources) if flat.args.get('joins') else []
    if links is None:
        return None
    closure = {bind_column(c, schema, sources) for c in tree.args['group'].expressions}
    if None in closure:
        return None
    while True:
        old = closure.copy()
        for table in sources:
            keys = {(table, field) for field, meta in schema.columns[table].items() if meta['pk']}
            if keys and keys <= closure:
                closure.update((table, field) for field in schema.columns[table])
        for a, af, b, bf in links:
            if (a, af) in closure:
                closure.add((b, bf))
            if (b, bf) in closure:
                closure.add((a, af))
        if closure == old:
            break
    projected = []
    for item in tree.expressions:
        column = item.unalias()
        if not isinstance(column, exp.Column):
            return None
        pair = bind_column(column, schema, sources)
        if pair is None:
            return None
        projected.append(pair)
    return [pair for pair in projected if pair not in closure]


class ReviewSuffix:
    def __init__(self, fallback, proposer=generate):
        self.fallback, self.proposer = fallback, proposer
        self.candidate, self.evidence, self.done = None, None, False

    def act(self, observation):
        history, remaining = observation['history'], observation['remaining']
        if self.candidate:
            return {'tool': 'submit', 'sql': self.candidate} if passed(history[-1]['observation']) else self.fallback
        if self.done or not remaining:
            return self.fallback
        tree = rank_shape(observation, self.fallback['sql'])
        if tree is None:
            return self.fallback
        matched = any(e['tool'] == 'check' and identity(e['sql']) == identity(self.fallback['sql'])
                      and passed(e['observation']) for e in history)
        if not matched:
            return self.fallback
        tables = next((e['observation'] for e in history if e['tool'] == 'schema'), None)
        if tables is None:
            return {'tool': 'schema'} if remaining >= 2 else self.fallback
        invalid = undetermined_output(tree, tables)
        if not invalid:
            return self.fallback
        self.done = True
        sql = self.proposer(observation, tables)
        candidate = parse(sql)
        if (not isinstance(candidate, exp.Select) or len(candidate.expressions) != len(tree.expressions)
                or len(list(candidate.find_all(exp.Select))) != 1
                or undetermined_output(candidate, tables) != []):
            return self.fallback
        seen = {identity(e['sql']) for e in history if e['tool'] == 'check'}
        if identity(sql) in seen:
            return self.fallback
        self.evidence = {'kind': 'frozen_model_determined_group_review', 'undetermined_columns': invalid,
                         'sql': sql, 'semantic_conflicts': sorted(continuation_conflicts(sql, observation['task']['question'])),
                         'limitation': 'Model restores determined grouped output; public matching does not prove the requested entity or ranking direction.'}
        self.candidate = sql
        return {'tool': 'check', 'sql': sql}


class Agent:
    def __init__(self, baseline):
        self.baseline, self.suffix = baseline, None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action['tool'] != 'submit':
                return action
            self.suffix = ReviewSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
