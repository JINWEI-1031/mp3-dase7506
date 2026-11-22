"""Check a development-fitted lexical category alias against bounded public evidence."""
from functools import lru_cache
import json
from pathlib import Path
import re

from sqlglot import exp
from repair_rules import sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step10_bindings import field_words
from step59_agent import bound_literals, eligible_literals, sample_sql
from step105_agent import create_agent as create_baseline


@lru_cache(maxsize=1)
def learned_aliases():
    return json.loads(Path(__file__).with_name('step106-category-aliases.json').read_text(encoding='utf-8'))['aliases']


class AliasSuffix:
    def __init__(self, fallback):
        self.fallback, self.candidate, self.evidence = fallback, None, None

    def act(self, observation):
        history, remaining = observation['history'], observation['remaining']
        if self.candidate:
            return {'tool': 'submit', 'sql': self.candidate} if passed(history[-1]['observation']) else self.fallback
        if not remaining:
            return self.fallback
        tree, items = eligible_literals(self.fallback['sql'], observation['task']['question'])
        matches = [(item, alias) for item in items for alias in learned_aliases()
                   if item[1].this.casefold() == alias['source']
                   and alias['context'] in field_words(item[0].this.name)]
        if len(matches) != 1:
            return self.fallback
        item, alias = matches[0]
        tables = next((e['observation'] for e in history if e['tool'] == 'schema'), None)
        if tables is None:
            return {'tool': 'schema'} if remaining >= 3 else self.fallback
        schema = Schema(tables)
        binding = bound_literals(tree, [item], schema)
        if binding is None:
            return self.fallback
        width = re.fullmatch(r'(?:var)?char\(\s*(\d+)\s*\)', schema.columns[binding[0]][binding[1]]['type'], re.I)
        if width is None or not len(alias['target']) <= int(width[1]) < len(item[1].this):
            return self.fallback
        probe = sample_sql(schema, binding)
        sample = next((e['observation'] for e in history if e['tool'] == 'execute' and identity(e['sql']) == identity(probe)), None)
        if sample is None:
            return {'tool': 'execute', 'sql': probe} if remaining >= 2 else self.fallback
        values = [r[0] for r in sample['rows'] if len(r) == 1]
        if (not sample['ok'] or sample['truncated'] or not values
                or any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z.]{1,5}', v) for v in values)
                or any(v.casefold() == item[1].this.casefold() for v in values)):
            return self.fallback
        before = item[1].this
        item[1].replace(exp.Literal.string(alias['target']))
        sql = sql_text(tree)
        self.evidence = {'kind': 'development_fitted_category_alias', 'before': before,
                         'after': alias['target'], 'context': alias['context'], 'support': alias['support'],
                         'observed_codes': values, 'sql': sql,
                         'limitation': 'Fits development data; public agreement does not identify an unseen code uniquely.'}
        previous = next((e['observation'] for e in history if e['tool'] == 'check' and identity(e['sql']) == identity(sql)), None)
        if previous is not None:
            return {'tool': 'submit', 'sql': sql} if passed(previous) else self.fallback
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
            self.suffix = AliasSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
