"""Shared budget controller; deterministic CNL vs direct-AST same-edit ablation."""
from scope_cnl import compile_edit,qintent,parse,find_at
from sqlglot import exp
class Agent:
 def __init__(self,parent,mode='cnl'):self.parent=parent;self.mode=mode;self.held=None;self.phase=None;self.done=False;self.audit=[]
 def trial(self,o):
  schema=next((h['observation'] for h in o['history'] if h['tool']=='schema'),None)
  if schema is None or not o['remaining']:self.audit.append({'stage':'abstain','reason':'schema_or_budget_missing'});return self.held
  c,w,d=compile_edit(o['task']['question'],self.held['sql'],schema);self.audit.append({'stage':'compile','candidate':c,'reason':w})
  if not c:return self.held
  if self.mode=='direct_ast':
   # Same frozen edit contract, no language/model choice: direct-AST menu control.
   t=parse(self.held['sql']);path=c['edit']['node_path'];branches=[]
   for old_branch in (parse(c['sql']).this,parse(c['sql']).expression):
    b=t.copy();find_at(b,path).replace(find_at(old_branch,path).copy());branches.append(b)
   direct=exp.Intersect(this=branches[0],expression=branches[1],distinct=True).sql(dialect='sqlite');assert direct==c['sql'];c['sql']=direct
  if any(h['tool']=='check' and h['sql']==c['sql'] for h in o['history']):self.audit.append({'stage':'abstain','reason':'candidate_already_checked'});return self.held
  self.pending=c['sql'];self.phase='check';self.audit.append({'stage':'sent','sql':c['sql'],'remaining':o['remaining']});return {'tool':'check','sql':c['sql']}
 def act(self,o):
  if self.phase=='schema':self.phase=None;return self.trial(o)
  if self.phase=='check':
   self.phase=None;fb=o['history'][-1]['observation'];ok=bool(fb.get('ok') and fb.get('matches_example'));self.audit.append({'stage':'checked','public_pass':ok,'selection_evidence':'explicit independent populations contradict same-row or union scope'})
   return {'tool':'submit','sql':self.pending} if ok else self.held
  a=self.parent.act(o)
  if a['tool']!='submit' or self.done:return a
  self.done=True;self.held=a
  if any(h['tool']=='check' and h['sql']==a['sql'] and h['observation'].get('matches_example') for h in o['history']):
   self.audit.append({'stage':'abstain','reason':'parent_explicit_public_KEEP_no_extra_tools'});return a
  if not qintent(o['task']['question'])[0]:return a
  if not isinstance(parse(a['sql']),exp.Select):self.audit.append({'stage':'abstain','reason':'existing_set_scope_or_nonselect_preserved'});return a
  if not any(h['tool']=='schema' for h in o['history']):
   if o['remaining']<2:self.audit.append({'stage':'abstain','reason':'schema_and_check_budget_missing'});return a
   self.phase='schema';self.audit.append({'stage':'schema','remaining':o['remaining']});return {'tool':'schema'}
  return self.trial(o)
