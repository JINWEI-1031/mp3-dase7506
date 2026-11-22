"""Terminal-only routing: untouched real parent -> frozen failure union -> compiler first."""
import copy,hashlib
from gate_online import eligibility_online
from b_gate import receipt_tier
from runtime import RuntimeIntegrityError
from union_late_agent import Agent as UnionLate
from compiler_hardened import compile_candidates
from mp3_sql.sqlite_tool import validate_sql

def shape(example):
 def tag(v):
  if v is None:return 'NULL'
  if isinstance(v,(bool,int)):return 'INTEGER'
  if isinstance(v,float):return 'REAL'
  if isinstance(v,str):return 'TEXT'
  return 'UNKNOWN'
 width=len(example['columns']);return {'width':width,'types':[sorted({tag(r[i]) for r in example['rows'] if len(r)==width}) or ['UNKNOWN'] for i in range(width)]}
def admitted_prefix(history):
 out=[]
 for e in history[:6]:
  if e['tool']=='check':out.append({'tool':'check','sql':e['sql']})
  elif e['tool']=='schema':out.append({'tool':'schema','observation':copy.deepcopy(e['observation'])})
  else:out.append({'tool':e['tool']})
 return out
class MemoParent:
 def __init__(self,action,obs):self.action=copy.deepcopy(action);self.obs=copy.deepcopy(obs);self.calls=0
 def act(self,obs):
  self.calls+=1
  if self.calls!=1 or obs!=self.obs:raise RuntimeIntegrityError('legacy fallback reused or altered terminal observation')
  return copy.deepcopy(self.action)
class Agent:
 def __init__(self,parent,runtime,state,compiler=compile_candidates):self.parent=parent;self.runtime=runtime;self.state=state;self.compiler=compiler;self.audit=[];self.attempted=False
 def act(self,obs):
  action=self.parent.act(obs)
  if action['tool']!='submit':return action
  parent=copy.deepcopy(action);h=copy.deepcopy(obs['history']);original_obs=copy.deepcopy(obs);n=self.state.completions;attempts=len(self.state.attempts)
  current=eligibility_online(obs,parent['sql'],n);B=receipt_tier(parent['sql'],h);eligible=current['eligible'] or (B['proved'] and type(n) is int and 0<=n<6);gate={**current,'eligible':eligible,'route':current['route'] if current['eligible'] else B['route'],'B_receipt_tier':B,'current194_gate_eligible':current['eligible']};self.audit.append({'stage':'terminal_failure_union_gate','gate':gate})
  if not eligible:return parent
  if self.attempted:raise RuntimeIntegrityError('terminal rescue attempted more than once')
  self.attempted=True
  def invariant(expect_count,expect_attempts):
   if obs!=original_obs or obs['history']!=h or self.state.completions!=expect_count or len(self.state.attempts)!=expect_attempts:raise RuntimeIntegrityError('compiler/terminal actual history or real counter mutation')
  try:
   args={'question':obs['task']['question'],'example_structure':shape(obs['example']),'actual_paid_prefix':admitted_prefix(h)}
   try:result=self.compiler(**copy.deepcopy(args))
   finally:invariant(n,attempts)
   if not isinstance(result,dict) or result.get('status') not in ['GENERATED','UNKNOWN_NO_SUPPORTED_CANDIDATE'] or not isinstance(result.get('candidates'),list) or len(result['candidates'])>2 or bool(result['candidates'])!=(result['status']=='GENERATED'):raise RuntimeIntegrityError('compiler result contract')
   self.audit.append({'stage':'pure_compiler','admitted_inputs':args,'result':copy.deepcopy(result),'new_model_calls':0,'new_paid_actions':0})
   if result['candidates']:
    first=result['candidates'][0];sql=first['SQL'];tree=validate_sql(sql)
    if first['SQL_SHA256']!=hashlib.sha256(sql.encode()).hexdigest() or first['output_width']!=args['example_structure']['width'] or len(tree.selects)!=args['example_structure']['width']:raise RuntimeIntegrityError('first complete candidate integrity or width')
    step=first['source_step']
    if type(step) is not int or not 1<=step<=len(args['actual_paid_prefix']) or args['actual_paid_prefix'][step-1]!={'tool':'check','sql':first['source_SQL']}:raise RuntimeIntegrityError('compiler source not an actually paid first6 check')
    invariant(n,attempts);self.audit.append({'stage':'submit_first_compiler_UNCHECKED','candidate_index':0,'SQL':sql,'candidate_checked':False,'candidate_all3':'UNKNOWN','new_paid_actions':0,'late_model_calls':0});return {'tool':'submit','sql':sql}
  except BaseException as error:
   self.audit.append({'stage':'UNKNOWN_TECHNICAL_STOP','phase':'compiler','error':repr(error),'actual_completions':self.state.completions});raise RuntimeIntegrityError('compiler hard failure/integrity mismatch') from error
  memo=MemoParent(parent,obs);legacy=UnionLate(memo,self.runtime,self.state,'original_none',gate)
  try:return legacy.act(obs)
  finally:
   self.audit.extend(legacy.audit)
   if obs!=original_obs or not n<=self.state.completions<=min(n+1,6) or not attempts<=len(self.state.attempts)<=attempts+1:raise RuntimeIntegrityError('legacy terminal fallback history/counter violation')
