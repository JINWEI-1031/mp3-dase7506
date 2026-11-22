"""Error-directed search with causal parent continuation; no task/database exceptions."""
import os,re
from sqlglot import exp
from repair_rules import parse,sql_text
from step_pool192_selective import Agent as Parent
from step_pool192_rules import generate,schema_errors,score,owned_columns
from step124_pool_search import sources
from step4_agent import identity,passed
from mp3_sql.sqlite_tool import validate_sql

def signature(sql,schema):
    tree=parse(sql)
    if tree is None:return identity(sql)
    for select in tree.find_all(exp.Select):
        src=sources(select,schema)
        for col in owned_columns(select):
            owners=[t['name'] for a,t in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
            if len(owners)==1:col.set('table',exp.to_identifier(owners[0].lower()))
        for table in [n for n in select.find_all(exp.Table) if n.find_ancestor(exp.Select) is select]:table.set('alias',None)
    for node in list(tree.find_all(exp.Alias)):node.replace(node.this.copy())
    for node in tree.find_all(exp.Identifier):node.set('this',node.this.lower());node.set('quoted',False)
    return sql_text(tree)

def qualified_repair(sql,obs,schema):
    tree=parse(sql)
    if tree is None:return None
    changed=False
    for select in tree.find_all(exp.Select):
        src=sources(select,schema)
        for col in owned_columns(select):
            if not col.table:continue
            local={a.lower():t for a,t in src.items()}
            if col.table.lower() not in local:continue
            if any(c['name'].lower()==col.name.lower() for c in local[col.table.lower()]['columns']):continue
            owners=[a for a,t in src.items() if any(c['name'].lower()==col.name.lower() for c in t['columns'])]
            if len(owners)==1:col.set('table',exp.to_identifier(owners[0]));changed=True
    return sql_text(tree) if changed else None

class Agent(Parent):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint);arm=os.environ.get('MP3_RECOVERY192_ARM','combined')
        self.features={'error','exit','priority','dedup'} if arm=='combined' else {arm}
        self.attempted=set();self.failures=0;self.reasons=[]
    def act(self,obs):
        if self.pending is not None and obs['history'][-1]['tool']=='check' and not passed(obs['history'][-1]['observation']):self.failures+=1
        return super().act(obs)
    def candidate(self,obs,schema,seeds,early=False):
        current=self.queue['sql'] if early else self.fallback['sql']
        checks=[e for e in obs['history'] if e['tool']=='check']
        new_value_evidence=bool('error' in self.features and checks and self.audit and self.audit[-1]['path'][-1] in ('shape','qualified_owner') and checks[-1]['observation']['ok'] and checks[-1]['observation']['row_count']==0 and obs['example']['row_count']>0)
        if 'exit' in self.features and early and ((self.failures>=1 and not new_value_evidence) or obs['remaining']<3):
            self.reasons.append('resume_parent_after_unproductive_search');return None
        families=set(self.families)
        if 'priority' in self.features:seeds=[current]
        if 'error' in self.features:
            if early:
                families&={'shape','entity'}
                checks=[e for e in obs['history'] if e['tool']=='check']
                if checks and checks[-1]['observation']['ok'] and checks[-1]['observation']['row_count']==0 and obs['example']['row_count']>0:
                    families.add('value')
            else:
                checks=[e for e in obs['history'] if e['tool']=='check']
                last=checks[-1]['observation'] if checks else {}
                allowed=set()
                if schema_errors(current,obs,schema):allowed|={'shape','entity'}
                if last.get('ok') and last.get('row_count')==0 and not last.get('matches_example'):allowed.add('value')
                q=obs['task']['question'].lower()
                if re.search(r'\bfirst\s+(?!names?\b|and\b)[a-z]',q):allowed.add('ordinal')
                if re.search(r'\bboth\s+[a-z]+\s+and\s+[a-z]+\s*[?.]?$',q):allowed.add('categories')
                families&=allowed
            seeds=[current,*[e['sql'] for e in obs['history'] if e['tool']=='check'][-1:]]
        self.pool=generate(seeds,obs,schema,families)
        if 'error' in self.features:
            fixed=qualified_repair(current,obs,schema)
            if fixed and identity(fixed)!=identity(current):self.pool.insert(0,{'sql':fixed,'path':['qualified_owner'],'score':score(fixed,obs,schema)})
        checked={identity(e['sql']) for e in obs['history'] if e['tool'] in ('check','execute')}
        semantic_checked={signature(e['sql'],schema) for e in obs['history'] if e['tool'] in ('check','execute')}
        matched=any(e['tool']=='check' and passed(e['observation']) for e in obs['history'])
        for item in self.pool:
            if identity(item['sql']) in checked or identity(item['sql'])==identity(current):continue
            if schema_errors(item['sql'],obs,schema):continue
            key=signature(item['sql'],schema)
            if 'dedup' in self.features and (key in semantic_checked or key in self.attempted):continue
            if not early and matched and item['score']<score(current,obs,schema)+.5:continue
            try:validate_sql(item['sql'])
            except ValueError:continue
            self.pending=item['sql'];self.attempted.add(key);self.audit.append({**item,'early':early,'remaining':obs['remaining']})
            return {'tool':'check','sql':item['sql']}
        return None

def create_agent(checkpoint=None):return Agent(checkpoint)
