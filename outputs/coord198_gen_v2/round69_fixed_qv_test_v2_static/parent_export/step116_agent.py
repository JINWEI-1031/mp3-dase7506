"""Validation-selected structural repairs over frozen Step63+D; no benchmark IDs."""
import os
import re
from sqlglot import exp
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step63_agent import create_agent as baseline


def enabled():
    return set(os.environ.get('MP3_STEP116_ARMS', 'quantifier,temporal,aggregate').split(','))


def qid(s):
    return '"' + s.replace('"', '""') + '"'


def plans(obs, fallback, tables, arms):
    question = obs['task']['question'].lower()
    tree = parse(fallback['sql'])
    result = []
    if 'quantifier' in arms and isinstance(tree, exp.Select):
        new = tree.copy()
        changed = False
        if re.search(r'\bafter any\b', question):
            for node in list(new.find_all(exp.Max)):
                node.replace(exp.Min(this=node.this.copy()))
                changed = True
        if re.search(r'\bbefore any\b', question):
            for node in list(new.find_all(exp.Min)):
                node.replace(exp.Max(this=node.this.copy()))
                changed = True
        for node in list(new.find_all(exp.GTE)):
            if isinstance(node.this, exp.Count) and isinstance(node.expression, exp.Literal):
                value = node.expression.this
                words = {'2': 'two', '3': 'three', '1': 'one'}
                if value in words and re.search(r'\b' + words[value] + r'\b', question) and not re.search(r'at least|no fewer|more than|minimum', question):
                    node.replace(exp.EQ(this=node.this.copy(), expression=node.expression.copy()))
                    changed = True
        if changed:
            result.append(('quantifier', sql_text(new)))
    if not tables:
        return result
    schema = {t['name'].lower(): t for t in tables}
    if 'temporal' in arms and isinstance(tree, exp.Select) and re.search(r'launch\w*.*(?:most recent|latest)|(?:most recent|latest).*launch', question):
        # Bind the mentioned event attribute on the output entity, rather than a related event's time.
        projected = [x.unalias() for x in tree.expressions]
        if projected and all(isinstance(x, exp.Column) and x.table for x in projected):
            names = {x.table.lower() for x in projected}
            if len(names) == 1 and next(iter(names)) in schema:
                t = schema[next(iter(names))]
                fields = [c['name'] for c in t['columns'] if 'launch' in c['name'].lower()]
                if len(fields) == 1:
                    new = exp.select(*[x.copy() for x in tree.expressions]).from_(qid(t['name'])).order_by(exp.Ordered(this=exp.column(fields[0], quoted=True), desc=True)).limit(1)
                    result.append(('temporal', sql_text(new)))
    hints = next((e['observation'] for e in obs['history'] if e['tool'] == 'repair_hint'), [])
    parents = [fallback['sql'], *hints]
    for sql in parents:
        source = parse(sql)
        if not isinstance(source, exp.Select) or len(list(source.find_all(exp.Select))) != 1:
            continue
        sources = [(t.alias_or_name, schema[t.name.lower()]) for t in source.find_all(exp.Table) if t.name.lower() in schema]
        if 'normalize' in arms:
            new = source.copy()
            cols = obs['example']['columns']
            if len(new.expressions) > len(cols):
                selected = []
                for head in cols:
                    matches = [x for x in new.expressions if x.alias_or_name.lower() == head.lower()]
                    if len(matches) != 1:
                        selected = []
                        break
                    selected.append(matches[0].copy())
                if selected:
                    new.set('expressions', selected)
            # Resolve all occurrences together. Multiple owners allowed only for equated join keys.
            for c in new.find_all(exp.Column):
                if c.table:
                    continue
                owners = [alias for alias, t in sources if any(f['name'].lower() == c.name.lower() for f in t['columns'])]
                if len(owners) == 1:
                    c.set('table', exp.to_identifier(owners[0], quoted=True))
                elif len(owners) == 2:
                    equated = any(isinstance(e.this, exp.Column) and isinstance(e.expression, exp.Column) and e.this.name.lower() == c.name.lower() == e.expression.name.lower() and {e.this.table.lower(), e.expression.table.lower()} == {o.lower() for o in owners} for e in new.find_all(exp.EQ))
                    if equated:
                        c.set('table', exp.to_identifier(owners[0], quoted=True))
            if identity(sql_text(new)) != identity(sql):
                result.append(('normalize', sql_text(new)))
        if 'aggregate' in arms:
            if re.search(r'older than average', question) and len(obs['example']['columns']) == 2:
                for alias, t in sources:
                    ages = [c['name'] for c in t['columns'] if c['name'].lower() == 'age']
                    groups = [c['name'] for c in t['columns'] if c['name'].lower() in {h.lower() for h in obs['example']['columns']}]
                    if len(ages) == len(groups) == 1:
                        age, group, table = qid(ages[0]), qid(groups[0]), qid(t['name'])
                        expressions = ['COUNT(*)' if 'count' in h.lower() else group for h in obs['example']['columns']]
                        result.append(('aggregate', f'SELECT {", ".join(expressions)} FROM {table} WHERE {age} > (SELECT AVG({age}) FROM {table}) GROUP BY {group}'))
            if 'average amount' in question:
                new = source.copy()
                for avg in new.find_all(exp.Avg):
                    for c in avg.find_all(exp.Column):
                        matches = [(a, f['name']) for a, t in sources for f in t['columns'] if 'amount' in f['name'].lower() and 'purchas' in f['name'].lower()]
                        if len(matches) == 1:
                            c.set('this', exp.to_identifier(matches[0][1], quoted=True))
                            c.set('table', exp.to_identifier(matches[0][0], quoted=True))
                if identity(sql_text(new)) != identity(sql):
                    result.append(('aggregate', sql_text(new)))
    seen = {identity(e['sql']) for e in obs['history'] if e['tool'] == 'check'}
    output = []
    for family, sql in result:
        if identity(sql) not in seen:
            output.append((family, sql))
            seen.add(identity(sql))
    return output


class Agent:
    def __init__(self, checkpoint=None):
        self.base = baseline(checkpoint)
        self.arms = enabled()
        self.pending = None
        self.finished = False
        self.fallback = None
        self.early = False
        self.attempted = set()

    def act(self, obs):
        if self.pending:
            if passed(obs['history'][-1]['observation']):
                return {'tool': 'submit', 'sql': self.pending}
            self.pending = None
        if self.finished:
            return self.base.act(obs) if self.early else self.fallback
        hints = any(e['tool'] == 'repair_hint' for e in obs['history'])
        if self.fallback is None:
            # Try joint syntax/aggregation repairs before piecemeal attempts exhaust budget.
            if hints and self.arms & {'normalize', 'aggregate'}:
                self.early = True
                self.fallback = {'tool': 'submit', 'sql': obs['issue_sql']}
            else:
                action = self.base.act(obs)
                if action['tool'] != 'submit':
                    return action
                self.fallback = action
        tables = next((e['observation'] for e in obs['history'] if e['tool'] == 'schema'), None)
        relevant = bool(self.arms & {'normalize', 'aggregate'} and hints or 'temporal' in self.arms and 'launch' in obs['task']['question'].lower())
        if tables is None and relevant and obs['remaining'] >= 2:
            return {'tool': 'schema'}
        candidates = plans(obs, self.fallback, tables, self.arms)
        for family, sql in candidates:
            if family in self.attempted or not obs['remaining']:
                continue
            self.attempted.add(family)
            self.pending = sql
            return {'tool': 'check', 'sql': sql}
        self.finished = True
        return self.base.act(obs) if self.early else self.fallback


def create_agent(checkpoint=None):
    return Agent(checkpoint)
