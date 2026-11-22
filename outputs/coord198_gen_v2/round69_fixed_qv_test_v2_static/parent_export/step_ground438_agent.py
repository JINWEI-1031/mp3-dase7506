import copy
from pathlib import Path
import json
from step_proof435_frozen import create_agent as baseline
from step_ground438_operators import population_join
from step_ground438_values import request,apply
from step_proof435_contract_agent import state_hash
from step_space435_common import structural_key
from step4_agent import passed
from step_ground438_contract import semantic_key

class Agent:
    def __init__(self,checkpoint=None,arm='combined',fold=None):
        self.base=baseline(checkpoint);self.arm=arm;self.audit=[];self.telemetry=[];self.held=None;self.snapshot=None;self.pending=None;self.probe=None;self.saved=None;self.used=False
        support=json.loads((Path(__file__).parent/'ground438_support.json').read_text());self.support=support['all_support'] if fold is None else support['out_of_fold_support'][str(fold)]
    def eligible(self,c,parent,obs,schema):
        if c['source']=='ground438_values':return structural_key(parent,schema)==structural_key(c['parent'],schema)
        cs,_=population_join(parent,obs,schema,self.support)
        target=semantic_key(c['sql'],schema)
        return target is not None and any(semantic_key(v['sql'],schema)==target for v in cs) and semantic_key(parent,schema)!=target
    def act(self,obs):
        if self.arm=='control':return self.base.act(obs)
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.probe:
            plan=self.probe;self.probe=None;cs,reason=apply(plan,obs['history'][-1]['observation']);self.telemetry.append({'stage':'probe_feedback','reason':reason,'pool':cs,'remaining':obs['remaining']})
            if cs and obs['remaining']:
                c={**cs[0],'parent':plan['parent']};self.pending=c;self.audit.append({**c,'remaining':obs['remaining']});return {'tool':'check','sql':c['sql']}
            self.base.__dict__.clear();self.base.__dict__.update(copy.deepcopy(self.snapshot));action=self.held;self.held=None;return action
        if self.pending:
            c=self.pending;self.pending=None;ok=passed(obs['history'][-1]['observation']);self.audit[-1]['public_pass']=ok
            self.base.__dict__.clear();self.base.__dict__.update(copy.deepcopy(self.snapshot));self.telemetry.append({'stage':'restore_parent','exact_state_restored':state_hash(vars(self.base))==state_hash(self.snapshot),'held_action':self.held,'remaining':obs['remaining']})
            action=self.held;self.held=None
            if ok:
                if action['tool']=='submit' and self.arm not in ('feedback_only','operator_feedback'):self.telemetry.append({'stage':'forced_replacement','approved':True});return {'tool':'submit','sql':c['sql']}
                self.saved=c
            return action
        action=self.base.act(obs)
        if action['tool']=='submit' and self.saved:
            approved=self.arm not in ('feedback_only','operator_feedback') and self.eligible(self.saved,action['sql'],obs,schema)
            self.telemetry.append({'stage':'final_reselection','approved':approved,'parent':action['sql'],'feedback_only':self.arm=='feedback_only'})
            if approved:return {'tool':'submit','sql':self.saved['sql']}
        if self.used or schema is None:return action
        terminal=action['tool']=='submit';checks=[e for e in obs['history'] if e['tool']=='check']
        if not terminal and not (obs['remaining']>=4 and checks and not passed(checks[-1]['observation'])):return action
        if not obs['remaining']:return action
        current=action.get('sql',checks[-1]['sql'] if checks else obs['issue_sql']);pool=[]
        if self.arm in ('operator','combined','feedback_only','operator_feedback'):pool,reason=population_join(current,obs,schema,self.support)
        self.telemetry.append({'stage':'generated','pool':pool,'remaining':obs['remaining'],'current':current,'support':self.support})
        checked={structural_key(e['sql'],schema) for e in obs['history'] if e['tool'] in ('check','execute')}
        pool=[c for c in pool if structural_key(c['sql'],schema) not in checked and self.eligible(c,current,obs,schema)]
        if pool:
            self.used=True;self.held=action;self.snapshot=copy.deepcopy(vars(self.base));self.pending=pool[0];self.audit.append({**pool[0],'parent':current,'remaining':obs['remaining']});return {'tool':'check','sql':pool[0]['sql']}
        if self.arm in ('values','combined','feedback_only') and obs['remaining']>=(2 if terminal else 5):
            plan,reason=request(current,obs,schema)
            if plan:
                self.used=True;self.probe=plan;self.held=action;self.snapshot=copy.deepcopy(vars(self.base));self.telemetry.append({'stage':'request_value_evidence','plan':plan,'remaining':obs['remaining']});return {'tool':'execute','sql':plan['sql']}
        return action

def create_agent(checkpoint=None):return Agent(checkpoint)
