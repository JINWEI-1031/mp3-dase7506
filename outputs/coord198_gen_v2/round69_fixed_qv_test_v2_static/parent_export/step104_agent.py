"""Infer one simple numeric boundary from complete public joined-row witnesses."""
from collections import Counter
import math
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import passed
from step4_edits import Schema
from step38.event_grain import join_graph, terms
from step103_agent import create_agent as create_baseline


def witness_plan(observation):
    if observation['remaining'] < 2 or observation['example']['truncated']:
        return None
    expected = observation['example']['rows']
    heads = observation['example']['columns']
    if not 2 <= len(expected) <= 8 or len(heads) != 2:
        return None
    checks = [e for e in observation['history'] if e['tool'] == 'check']
    if not checks or any(passed(e['observation']) for e in checks):
        return None
    tables = next((e['observation'] for e in observation['history'] if e['tool'] == 'schema'), None)
    if tables is None:
        return None
    schema = Schema(tables)
    tree = parse(checks[-1]['sql'])
    if (not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1
            or len(tree.expressions) != 2 or len(tree.args.get('joins', [])) != 1
            or any(tree.args.get(k) for k in ('where', 'distinct', 'group', 'having', 'order', 'limit', 'offset', 'with'))
            or any(not isinstance(c, exp.Column) for c in tree.expressions)
            or [c.name.casefold() for c in tree.expressions] != [h.casefold() for h in heads]):
        return None
    sources = schema.sources(tree)
    if len(sources) != 2 or not join_graph(tree, schema, sources):
        return None
    origin = {c.table.casefold() for c in tree.expressions}
    if len(origin) != 1 or not next(iter(origin)):
        return None
    alternatives = [name for name, alias in sources.items() if alias.casefold() not in origin
                    and all(h.casefold() in schema.columns[name] for h in heads)
                    and terms(name) & terms(observation['task']['question'])]
    if len(alternatives) != 1:
        return None
    target = alternatives[0]
    foreign_keys = {fk['from'].casefold() for fk in schema.tables[target]['foreign_keys']}
    measures = [name for name, meta in schema.columns[target].items()
                if name not in foreign_keys and not meta['pk']
                and re.search(r'REAL|FLOAT|DOUBLE|INT|NUMERIC|DECIMAL', meta['type'], re.I)
                and name not in {h.casefold() for h in heads}]
    if not 1 <= len(measures) <= 3:
        return None
    projected = [schema.column(target, h.casefold(), sources[target]) for h in heads]
    query = tree.copy()
    query.set('expressions', projected)
    probe = query.copy()
    probe.set('expressions', [*projected, *[schema.column(target, m, sources[target]) for m in measures]])
    # LIMIT 9 detects incomplete witnesses, rather than fitting to a truncated sample.
    probe.limit(9, copy=False)
    return {'query': query, 'probe': sql_text(probe), 'target': target,
            'measures': measures, 'alias': sources[target], 'fallback': checks[-1]['sql']}


def boundary_candidate(plan, feedback, expected):
    if not feedback['ok'] or feedback['truncated'] or not feedback['rows']:
        return None
    rows = feedback['rows']
    wanted = Counter(tuple(row) for row in expected)
    if not wanted <= Counter(tuple(row[:2]) for row in rows):
        return None
    candidates = []
    for index, measure in enumerate(plan['measures'], 2):
        if any(not isinstance(row[index], (int, float)) or not math.isfinite(row[index]) for row in rows):
            continue
        positive = [row[index] for row in rows if tuple(row[:2]) in wanted]
        negative = [row[index] for row in rows if tuple(row[:2]) not in wanted]
        if not positive or not negative:
            continue
        choices = []
        for above in (True, False):
            low, high = (max(negative), min(positive)) if above else (max(positive), min(negative))
            if low >= high:
                continue
            for power in range(12, -7, -1):
                scale = 10 ** power
                boundary = (math.ceil(low / scale) if above else math.floor(high / scale)) * scale
                if (low <= boundary < high if above else low < boundary <= high):
                    actual = Counter(tuple(row[:2]) for row in rows
                                     if (row[index] > boundary if above else row[index] < boundary))
                    if actual == wanted:
                        choices.append((above, boundary, power))
                        break
        if len(choices) == 1:
            candidates.append((measure, *choices[0]))
    if len(candidates) != 1:
        return None
    measure, above, boundary, power = candidates[0]
    query = plan['query'].copy()
    predicate = exp.GT if above else exp.LT
    query.where(predicate(this=exp.column(measure, table=plan['alias']),
                          expression=exp.Literal.number(boundary)), copy=False)
    return {'sql': sql_text(query), 'kind': 'complete_public_witness_numeric_boundary',
            'target': plan['target'], 'measure': measure, 'boundary': boundary,
            'rounding_power': power, 'witness_count': len(rows),
            'limitation': 'An inferred predicate and output binding may differ from the question wording.'}


class Agent:
    plan_builder = staticmethod(witness_plan)
    candidate_builder = staticmethod(boundary_candidate)

    def __init__(self, baseline):
        self.baseline, self.plan, self.candidate = baseline, None, None
        self.attempted, self.evidence = False, None

    def act(self, observation):
        if self.candidate:
            if passed(observation['history'][-1]['observation']):
                return {'tool': 'submit', 'sql': self.candidate}
            self.candidate = None
            return self.baseline.act(observation)
        if self.plan:
            candidate = self.candidate_builder(self.plan, observation['history'][-1]['observation'], observation['example']['rows'])
            self.plan = None
            if candidate and observation['remaining']:
                self.evidence, self.candidate = candidate, candidate['sql']
                return {'tool': 'check', 'sql': self.candidate}
        if not self.attempted and observation['history'] and observation['history'][-1]['tool'] == 'schema':
            self.plan = self.plan_builder(observation)
            if self.plan:
                self.attempted = True
                return {'tool': 'execute', 'sql': self.plan['probe']}
        return self.baseline.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
