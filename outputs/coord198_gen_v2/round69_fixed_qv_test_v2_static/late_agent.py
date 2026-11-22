"""Online terminal-only wrapper. No case0 certificate/DB/gold/score or ID routing."""
import copy,json
from gate_online import eligibility_online
from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity
from runtime import RuntimeIntegrityError,InputUnavailable
SYSTEM='Repair the SQLite query to answer the question. Output only the corrected SQL.'
class Agent:
 def __init__(self,parent,runtime,state,view):self.parent=parent;self.runtime=runtime;self.state=state;self.view=view;self.attempted=False;self.audit=[]
 def act(self,obs):
  action=self.parent.act(obs)
  if action['tool']!='submit':return action
  parent=copy.deepcopy(action);gate=eligibility_online(obs,parent['sql'],self.state.completions);self.audit.append({'stage':'online_gate','gate':gate})
  if self.attempted or not gate['eligible']:return parent
  schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
  if schema is None:self.audit.append({'stage':'abstain','reason':'missing_paid_schema'});return parent
  self.attempted=True;h=copy.deepcopy(obs['history']);before=self.state.completions;draft=obs['issue_sql'] if self.view=='original_none' else parent['sql'];payload={'question':obs['task']['question'],'draft':draft,'columns':copy.deepcopy(obs['example']['columns']),'schema':copy.deepcopy(schema)};messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]
  def invariant(expected=None):
   after=self.state.completions
   if type(after) is not int or not before<=after<=min(6,before+1) or obs['history']!=h or (expected is not None and after!=expected):raise RuntimeIntegrityError('late counter/history invariant violation')
  try:
   try:prepared=self.runtime.preflight_late(messages)
   finally:invariant(before)
  except InputUnavailable as e:
   self.audit.append({'stage':'exact_parent_fallback','reason':'pre_input_unavailable','error':repr(e),'actual_completions':self.state.completions});return parent
  except BaseException as e:
   self.audit.append({'stage':'UNKNOWN_TECHNICAL_STOP','phase':'preflight','error':repr(e),'actual_completions':self.state.completions});raise RuntimeIntegrityError('late preflight hard failure') from e
  try:
   try:text=self.runtime.late(prepared,do_sample=False,max_new_tokens=512)
   finally:invariant()
   invariant(before+1)
  except BaseException as e:
   self.audit.append({'stage':'UNKNOWN_TECHNICAL_STOP','phase':'provider','error':repr(e),'actual_completions':self.state.completions});raise RuntimeIntegrityError('late provider or integrity failure') from e
  try:
   sql=text.strip();tree=validate_sql(sql)
   if len(tree.selects)!=len(obs['example']['columns']):raise ValueError('projection_width')
   if identity(sql) in {identity(e['sql']) for e in h if e['tool']=='check'}|{identity(parent['sql'])}:raise ValueError('duplicate_actual_check_or_parent')
  except ValueError as e:
   cause=e;seen=set()
   while cause is not None and id(cause) not in seen:
    seen.add(id(cause))
    if isinstance(cause,(MemoryError,RecursionError,RuntimeIntegrityError)) or any(x in str(cause).lower() for x in ['memory','recursion','resource','timeout']):raise RuntimeIntegrityError('hard parser failure') from e
    cause=cause.__cause__ or cause.__context__
   self.audit.append({'stage':'exact_parent_fallback','reason':'ordinary_parse_rejection','error':repr(e),'new_paid_actions':0});return parent
  finally:invariant(before+1)
  self.audit.append({'stage':'submit_new_unchecked_SQL','view':self.view,'SQL':sql,'candidate_checked':False,'candidate_all3':'UNKNOWN','new_paid_actions':0,'fake_ACK_or_PASS':False});return {'tool':'submit','sql':sql}
