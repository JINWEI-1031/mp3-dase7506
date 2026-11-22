"""Evidence-gated hint acquisition, candidate selection and bounded repair search."""
import re
from sqlglot import exp
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step107_frontier_selected import create_agent as baseline
from step107_frontier_search import materialized, expand, score, sources
from step123_agent import words, singular, entity_plans, grain_plans
from step127_agent import flags
from mp3_sql.sqlite_tool import validate_sql
from step107_chainnext_signature import signature

ARMS = ('baseline', 'priority', 'dedup', 'exit', 'combined')


def first_intent(q):
    return bool(re.search(r'\bfirst\s+(?!name\b|and\b)[a-z]', q.lower()))


def risk(sql, obs, schema=()):
    tree = parse(sql); q = obs['task']['question'].lower()
    if tree is not None and tree.find(exp.SetOperation): return set()
    if not isinstance(tree, exp.Select): return {'invalid_shape'}
    if any(isinstance(j.this, exp.Subquery) for j in tree.args.get('joins', [])):
        return set()
    root = tree.args.get('from')
    if root is not None and isinstance(root.this, exp.Subquery): return set()
    out = set(flags(sql, q, schema))
    if len(tree.expressions) != len(obs['example']['columns']): out.add('projection_width')
    if first_intent(q) and not (tree.args.get('order') and tree.args.get('limit')): out.add('missing_first_order')
    if re.search(r'\b(?:do not have|does not have|have no|has no|without)\b', q) and tree.find(exp.Count) and not (tree.find(exp.Exists) or tree.find(exp.Not)):
        out.add('negative_entity_grain')
    group = tree.args.get('group')
    if group and len(group.expressions) > 1: out.add('group_grain')
    where = tree.args.get('where')
    number_words = {'1':'one', '2':'two', '3':'three'}
    if where:
        for eq in where.find_all(exp.EQ):
            if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Literal) and not eq.expression.is_string:
                n = eq.expression.this
                if n != '0' and n not in words(q) and not re.search(r'\b'+re.escape(n)+r'\b', q) and number_words.get(n, '-') not in words(q):
                    out.add('unmentioned_numeric_filter')
    for eq in tree.find_all(exp.EQ):
        if eq.find_ancestor(exp.Join) is None: continue
        if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column) and eq.this.table != eq.expression.table:
            a, b = eq.this.name.lower(), eq.expression.name.lower()
            scope = eq.find_ancestor(exp.Select)
            bound = {alias.lower():t for alias,t in sources(scope,schema).items()}
            left, right = eq.this.table.lower(), eq.expression.table.lower()
            declared = False
            if left in bound and right in bound:
                declared = any(fk['from'].lower()==a and fk['table'].lower()==bound[right]['name'].lower() and fk['to'].lower()==b for fk in bound[left]['foreign_keys'])
                declared |= any(fk['from'].lower()==b and fk['table'].lower()==bound[left]['name'].lower() and fk['to'].lower()==a for fk in bound[right]['foreign_keys'])
            if declared: continue
            if a != b and (a.endswith('id') and b.endswith('id') or bool(re.search(r'id$', a)) != bool(re.search(r'id$', b))):
                out.add('join_binding')
    literals = [v.this for v in tree.find_all(exp.Literal) if v.is_string]
    if any(len(v.split()) == 2 and v.lower() in q for v in literals): out.add('compound_name')
    if schema:
        src = sources(tree, schema)
        request = re.search(r'\b(?:how many|number of)\s+(.+?)(?:\s+(?:are|is|were|related|for|per|each)\b|[?.]|$)', q)
        if request and tree.find(exp.Count):
            requested = {singular(w) for w in words(request[1])}
            targets = [t for t in schema if words(t['name']) and all(singular(w) in requested for w in words(t['name']))]
            if len(targets) == 1 and targets[0]['name'].lower() not in {t['name'].lower() for t in src.values()}:
                out.add('missing_counted_entity')
        projected = [e.unalias() for e in tree.expressions]
        if projected and all(isinstance(e, exp.Column) and e.table for e in projected):
            owner = {e.table.lower() for e in projected}
            for alias, table in src.items():
                noun = ' '.join(words(table['name']))
                if alias.lower() in owner and any(re.search(r'\b'+re.escape(noun)+r'\s+'+re.escape(v.lower())+r'\b', q) for v in literals):
                    out.add('entity_name_binding')
    return out


def rank(sql, obs, schema):
    return score(sql, obs, schema) - 3*len(risk(sql, obs, schema))


def preserves_values(sql, obs):
    new = parse(sql)
    if new is None: return False
    q = obs['task']['question'].lower()
    required = []
    old = parse(obs['issue_sql'])
    if old:
        for literal in old.find_all(exp.Literal):
            if literal.is_string:
                value = literal.this.strip('%').lower()
                if len(value) > 1 and value in q: required.append(set(words(value)))
    actual = set(words(' '.join(v.this for v in new.find_all(exp.Literal) if v.is_string)))
    return all(tokens <= actual for tokens in required)


def eligible(sql, obs):
    try: tree = validate_sql(sql)
    except ValueError: return False
    return len(tree.selects) == len(obs['example']['columns']) and preserves_values(sql, obs)


def semantic_edits(sql, obs, schema):
    out, _ = entity_plans(obs, sql, schema)
    out = [('entity_binding', s) for s in out]
    out += [('negative_grain', s) for s in grain_plans(obs, sql, schema)]
    tree = parse(sql)
    if isinstance(tree, exp.Select) and 'unmentioned_numeric_filter' in risk(sql, obs, schema):
        where = tree.args.get('where')
        if where and isinstance(where.this, exp.EQ) and isinstance(where.this.this, exp.Column) and where.this.this.name.lower().endswith('id'):
            new = tree.copy(); new.set('where', None)
            out.append(('remove_unmentioned_id_filter', sql_text(new)))
    return out


def candidates(seeds, obs, schema, depth):
    pool = expand(seeds, obs, schema, depth=depth)
    if depth:
        for item in list(pool[:12]):
            for kind, sql in semantic_edits(item['sql'], obs, schema):
                pool.append({'sql': sql, 'path': [*item['path'], kind], 'score': score(sql, obs, schema)})
    unique = {}
    for item in pool:
        if eligible(item['sql'], obs): unique.setdefault(identity(item['sql']), item)
    for item in unique.values(): item['score'] = rank(item['sql'], obs, schema)
    return sorted(unique.values(), key=lambda item: item['score'], reverse=True)[:48]


class Agent:
    def __init__(self, checkpoint=None, *, arm='combined'):
        if arm not in ARMS: raise ValueError('Unknown frontier arm')
        self.base = baseline(checkpoint); self.arm = arm
        self.fallback = None; self.pending = None; self.audit = []
        self.pool = []; self.searching = False

    def act(self, obs):
        if self.arm == 'baseline': return self.base.act(obs)
        checks = [e for e in obs['history'] if e['tool'] == 'check']
        schema = next((e['observation'] for e in obs['history'] if e['tool'] == 'schema'), None)
        if self.pending:
            self.audit[-1]['public_pass'] = passed(obs['history'][-1]['observation'])
            failed = not passed(obs['history'][-1]['observation'])
            self.pending = None
            if failed and self.arm in ('exit','combined'): return self.best(obs,schema or [])
        if not self.searching:
            action = self.base.act(obs)
            reserve = self.arm == 'reserve' and obs['remaining'] <= 2 and not any(passed(e['observation']) for e in checks) and len(checks) >= 2
            if action['tool'] != 'submit' and not reserve: return action
            self.fallback = action if action['tool'] == 'submit' else {'tool':'submit','sql':obs['issue_sql']}
            self.searching = True
        if not obs['remaining']:
            return self.best(obs, schema or []) if self.audit or self.pool else self.fallback
        risks = risk(self.fallback['sql'], obs, schema or [])
        has_hint = any(e['tool'] == 'repair_hint' for e in obs['history'])
        hint_on = True
        hint_risk = bool(risks & {'missing_first_order', 'negative_entity_grain', 'projection_width', 'invalid_shape'})
        if hint_on and not has_hint and hint_risk and obs['remaining'] >= 2:
            self.audit.append({'action':'request_hint', 'risk':sorted(risks)})
            return {'tool':'repair_hint'}
        if self.arm == 'hint' and not hint_risk: return self.fallback
        if not risks: return self.fallback
        if schema is None:
            if self.arm != 'hint' and obs['remaining'] >= 2: return {'tool':'schema'}
            schema = []
        seeds = [self.fallback['sql'], obs['issue_sql'], *materialized(self.base), *[e['sql'] for e in checks]]
        for event in obs['history']:
            if event['tool'] == 'repair_hint': seeds.extend(event['observation'])
        if self.arm == 'hint':
            seeds = [self.fallback['sql'], *[s for e in obs['history'] if e['tool']=='repair_hint' for s in e['observation']]]
        depth = 1
        if self.arm in ('priority','combined'): seeds = [self.fallback['sql']]
        self.pool = candidates(seeds, obs, schema, depth)
        checked = {identity(e['sql']) for e in checks}
        signatures = {signature(e['sql']) for e in checks}
        best = self.best(obs, schema)
        base_score = rank(best['sql'], obs, schema)
        for item in self.pool:
            if identity(item['sql']) in checked: continue
            if self.arm in ('dedup','combined') and signature(item['sql']) in signatures: continue
            if any(passed(e['observation']) for e in checks) and item['score'] < base_score + .5: continue
            self.pending = item['sql']
            self.audit.append({'sql':self.pending, 'score':item['score'], 'path':item['path'], 'risk':sorted(risks)})
            return {'tool':'check','sql':self.pending}
        return self.best(obs, schema)

    def best(self, obs, schema):
        choices = [e['sql'] for e in obs['history'] if e['tool']=='check' and passed(e['observation']) and eligible(e['sql'], obs)]
        if not choices: return self.fallback
        sql = max(choices, key=lambda s:rank(s, obs, schema))
        if identity(sql) == identity(self.fallback['sql']) or rank(sql, obs, schema) >= rank(self.fallback['sql'], obs, schema)+.5:
            return {'tool':'submit','sql':sql}
        return self.fallback


def create_agent(checkpoint=None): return Agent(checkpoint)
