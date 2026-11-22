"""Five independently selectable semantic repairs over the complete Step107 chain."""
import os
import re
from sqlglot import exp
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step107_restore_agent import create_agent as baseline
from step116_agent import plans as legacy_plans

FAMILIES = ('event', 'measure', 'above_average', 'any', 'exact')


def relevant(question, family):
    q = question.lower()
    return {
        'event': bool(re.search(r'launch\w*.*(?:most recent|latest)|(?:most recent|latest).*launch', q)),
        'measure': 'average amount' in q,
        'above_average': 'older than average' in q,
        'any': bool(re.search(r'\b(?:after|before) any\b', q)),
        'exact': bool(re.search(r'\b(?:one|two|three)\b', q)) and not bool(re.search(r'at least|no fewer|more than|minimum', q)),
    }[family]


def plans(obs, fallback, tables, families):
    q = obs['task']['question'].lower()
    result = []
    for family in FAMILIES:
        if family not in families or not relevant(q, family):
            continue
        if family in ('event', 'measure', 'above_average'):
            arm = 'temporal' if family == 'event' else 'aggregate'
            result.extend((family, sql) for _, sql in legacy_plans(obs, fallback, tables, {arm}))
            continue
        tree = parse(fallback['sql'])
        if not isinstance(tree, exp.Select):
            continue
        new = tree.copy()
        if family == 'any':
            # Limit the transformation to extrema used by a temporal comparison.
            old, replacement, comparison = (exp.Max, exp.Min, exp.GT) if 'after any' in q else (exp.Min, exp.Max, exp.LT)
            for node in list(new.find_all(comparison)):
                sub = node.expression
                if isinstance(sub, exp.Subquery) and isinstance(sub.this, exp.Select):
                    inner = sub.this
                    if len(inner.expressions) == 1 and isinstance(inner.expressions[0].unalias(), old):
                        item = inner.expressions[0].unalias()
                        item.replace(replacement(this=item.this.copy()))
        else:
            for node in list(new.find_all(exp.GTE)):
                if isinstance(node.this, exp.Count) and isinstance(node.expression, exp.Literal):
                    word = {'1': 'one', '2': 'two', '3': 'three'}.get(node.expression.this)
                    if word and re.search(r'\b' + word + r'\b', q):
                        node.replace(exp.EQ(this=node.this.copy(), expression=node.expression.copy()))
        if identity(sql_text(new)) != identity(fallback['sql']):
            result.append((family, sql_text(new)))
    seen = {identity(e['sql']) for e in obs['history'] if e['tool'] == 'check'}
    output = []
    for family, sql in result:
        if identity(sql) not in seen:
            output.append((family, sql)); seen.add(identity(sql))
    return output


class Agent:
    def __init__(self, checkpoint=None, *, families=None, schedule=None):
        self.base = baseline(checkpoint)
        self.families = set(families if families is not None else os.environ.get('MP3_STEP107_TRANSFER_ARMS', ','.join(FAMILIES)).split(','))
        self.schedule = schedule or os.environ.get('MP3_STEP107_TRANSFER_SCHEDULE', 'late')
        if self.schedule not in ('late', 'early_stop', 'early_review'):
            raise ValueError('Unknown semantic repair schedule')
        self.fallback = None
        self.pending = None
        self.early_tried = False
        self.phase = 'base'
        self.attempted = set()
        self.audit = []

    def review(self, obs, families):
        tables = next((e['observation'] for e in obs['history'] if e['tool'] == 'schema'), None)
        active = {f for f in families if f not in self.attempted and relevant(obs['task']['question'], f)}
        if not active or not obs['remaining']:
            return self.fallback
        needs_schema = bool(active & {'event', 'measure', 'above_average'})
        if needs_schema and tables is None and obs['remaining'] >= 2:
            return {'tool': 'schema'}
        for family, sql in plans(obs, self.fallback, tables, active):
            self.attempted.add(family)
            self.pending = (family, sql)
            self.audit.append({'family': family, 'phase': self.phase, 'sql': sql})
            return {'tool': 'check', 'sql': sql}
        self.attempted.update(active)
        return self.fallback

    def act(self, obs):
        if self.pending:
            family, sql = self.pending
            self.pending = None
            if passed(obs['history'][-1]['observation']):
                self.fallback = {'tool': 'submit', 'sql': sql}
                if self.phase == 'early' and self.schedule == 'early_stop':
                    self.phase = 'done'
                else:
                    self.phase = 'late'
        if self.phase == 'done':
            return self.fallback
        if self.phase == 'early':
            action = self.review(obs, self.families & {'measure', 'above_average'})
            if action['tool'] != 'submit':
                return action
            self.phase = 'base'; self.fallback = None
        if self.phase == 'base':
            if (self.schedule != 'late' and not self.early_tried and obs['history']
                    and obs['history'][-1]['tool'] == 'repair_hint'
                    and any(relevant(obs['task']['question'], f) for f in self.families & {'measure', 'above_average'})):
                self.early_tried = True; self.phase = 'early'
                self.fallback = {'tool': 'submit', 'sql': obs['issue_sql']}
                return self.act(obs)
            action = self.base.act(obs)
            if action['tool'] != 'submit':
                return action
            self.fallback = action; self.phase = 'late'
        action = self.review(obs, self.families)
        if action['tool'] == 'submit':
            self.phase = 'done'
        return action


def create_agent(checkpoint=None):
    return Agent(checkpoint)
