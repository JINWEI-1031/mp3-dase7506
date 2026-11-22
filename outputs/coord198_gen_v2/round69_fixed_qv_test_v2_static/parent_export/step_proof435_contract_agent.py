"""Charged semantic review with exact held-parent-state restoration."""
import copy,hashlib,pickle
from step_evidence28_192_frozen import create_agent as baseline
from step_proof435_provider import candidates
from step_proof435_contract_evidence import certificate
from step_space435_common import structural_key
from step4_agent import passed

def state_hash(state):
    import json
    def normalize(v,seen):
        if v is None or isinstance(v,(str,int,float,bool)):return v
        if callable(v):return {'callable':getattr(v,'__module__','')+'.'+getattr(v,'__qualname__',type(v).__name__)}
        if id(v) in seen:return {'cycle':type(v).__name__}
        seen=seen|{id(v)}
        if isinstance(v,dict):return {str(k):normalize(x,seen) for k,x in sorted(v.items(),key=lambda item:str(item[0]))}
        if isinstance(v,(tuple,list)):return [normalize(x,seen) for x in v]
        if isinstance(v,set):return sorted([normalize(x,seen) for x in v],key=str)
        if hasattr(v,'sql'):return {'ast':v.sql()}
        if hasattr(v,'__dict__'):return {'class':type(v).__module__+'.'+type(v).__name__,'state':normalize(vars(v),seen)}
        return {'type':type(v).__name__,'repr':repr(v)}
    return hashlib.sha256(json.dumps(normalize(state,set()),sort_keys=True).encode()).hexdigest()

class Agent:
    def __init__(self,checkpoint=None,arm='strict'):
        self.base=baseline(checkpoint);self.arm=arm;self.audit=[];self.telemetry=[]
        self.held=None;self.snapshot=None;self.pending=None;self.saved=None;self.used=False;self.generated=set()
    def act(self,obs):
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        tier={'sample':'sample','contradiction':'contradiction'}.get(self.arm,'preserve')
        if self.pending is not None:
            c=self.pending;self.pending=None;ok=passed(obs['history'][-1]['observation']);self.audit[-1]['public_pass']=ok
            self.base.__dict__.clear();self.base.__dict__.update(copy.deepcopy(self.snapshot))
            restored=state_hash(self.base.__dict__)==state_hash(self.snapshot)
            self.telemetry.append({'stage':'restore_parent','exact_state_restored':restored,'remaining':obs['remaining'],'held_action':self.held})
            action=self.held;self.held=None
            if ok:
                if action['tool']=='submit' or self.arm=='early':return {'tool':'submit','sql':c['sql']}
                self.saved=c
            return action
        if self.held is not None:
            action=self.held;self.held=None
        else:action=self.base.act(obs)
        terminal=action['tool']=='submit'
        if terminal and self.saved:
            proof=certificate(action['sql'],self.saved,obs,schema,tier)
            self.telemetry.append({'stage':'final_reselection','parent':action['sql'],'certificate':proof})
            if proof['approved']:return {'tool':'submit','sql':self.saved['sql']}
        if self.used:return action
        checks=[e for e in obs['history'] if e['tool']=='check']
        early=self.arm in ('early','opportunity','control','contract') and not terminal and schema is not None and obs['remaining']>=4 and checks and not passed(checks[-1]['observation'])
        if not terminal and not early:return action
        if not obs['remaining']:
            self.telemetry.append({'stage':'not_triggered','reason':'budget_exhausted'});return action
        if schema is None:
            if terminal and obs['remaining']>=2:self.held=action;return {'tool':'schema'}
            self.telemetry.append({'stage':'not_triggered','reason':'schema_budget'});return action
        current=action.get('sql',checks[-1]['sql'] if checks else obs['issue_sql'])
        generator='control' if self.arm=='control' else 'name' if self.arm=='name' else 'extended'
        pool,diagnostic=candidates(current,obs,schema,generator=generator)
        # A different source is permitted only as a question-grounded plan, and
        # still needs an explicit contradiction against the actual current SQL.
        if early and not pool and current!=obs['issue_sql']:
            pool,diagnostic=candidates(obs['issue_sql'],obs,schema,generator=generator)
            for c in pool:c['seed_source']='original_issue_grounded_plan'
        checked={structural_key(e['sql'],schema) for e in obs['history'] if e['tool'] in ('check','execute')}
        eligible=[]
        for c in pool:
            key=structural_key(c['sql'],schema)
            if key not in self.generated and len(self.generated)>=24:continue
            self.generated.add(key);c['certificate']=certificate(current,c,obs,schema,tier)
            if c['certificate']['approved'] and key not in checked and key!=structural_key(current,schema):eligible.append(c)
        self.telemetry.append({'stage':'generated','early':early,'remaining':obs['remaining'],'current':current,'pool':pool,'eligible':len(eligible),**diagnostic})
        if terminal:self.used=True
        if not eligible:return action
        c=eligible[0];self.used=True;self.pending=c;self.held=action;self.snapshot=copy.deepcopy(self.base.__dict__)
        self.audit.append({**c,'early':early,'remaining':obs['remaining'],'parent':current})
        return {'tool':'check','sql':c['sql']}

def create_agent(checkpoint=None):return Agent(checkpoint)
