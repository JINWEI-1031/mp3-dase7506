import copy,json
from pathlib import Path
from step_ground438_frozen import create_agent as baseline
from step_proof435_contract_agent import state_hash
from step4_agent import passed
from step_space435_common import structural_key
import step_graph439_edges as E,step_graph439_items as I
from step_graph439_contract import approve

class Agent:
    def __init__(self,checkpoint=None,arm='combined',fold=None):
        self.base=baseline(checkpoint);self.arm=arm;self.audit=[];self.telemetry=[];self.used=False;self.held=None;self.snapshot=None;self.probe=None;self.pending=None;self.saved=None
        meta=json.loads((Path(__file__).parent/'graph439_support.json').read_text());self.support=meta['all_support'] if fold is None else meta['fold_support'][str(fold)]
    def restore(self,obs):
        self.base.__dict__.clear();self.base.__dict__.update(copy.deepcopy(self.snapshot));self.telemetry.append({'stage':'restore_parent','exact_state_restored':state_hash(vars(self.base))==state_hash(self.snapshot),'held_action':self.held,'remaining':obs['remaining']})
        action=self.held;self.held=None;return action
    def send(self,c,obs):
        self.pending=c;self.audit.append({**c,'remaining':obs['remaining'],'held_action':self.held});return {'tool':'check','sql':c['sql']}
    def act(self,obs):
        if self.arm=='control':return self.base.act(obs)
        schema=next((h['observation'] for h in obs['history'] if h['tool']=='schema'),None)
        if self.probe:
            kind,p=self.probe;self.probe=None;fb=obs['history'][-1]['observation'];cs,reason=E.apply(p,fb,obs) if kind=='edges' else I.apply(p,fb)
            self.telemetry.append({'stage':'probe_feedback','kind':kind,'pool':cs,'reason':reason,'feedback':fb,'remaining':obs['remaining']})
            if cs and obs['remaining']:
                c={**cs[0],'parent':p['parent']};valid,why=approve(c,p['parent'],obs,schema,self.support)
                if valid:return self.send(c,obs)
                self.telemetry.append({'stage':'guard_reject','reason':why})
            return self.restore(obs)
        if self.pending:
            c=self.pending;self.pending=None;ok=passed(obs['history'][-1]['observation']);self.audit[-1]['public_pass']=ok;action=self.restore(obs)
            if ok:
                if action['tool']=='submit' and self.arm!='feedback_only':self.telemetry.append({'stage':'forced_replacement','approved':True,'parent':action['sql'],'candidate':c['sql']});return {'tool':'submit','sql':c['sql']}
                self.saved=c
            return action
        action=self.base.act(obs)
        if action['tool']=='submit' and self.saved:
            valid,reason=approve(self.saved,action['sql'],obs,schema,self.support);valid=valid and self.arm!='feedback_only'
            self.telemetry.append({'stage':'final_reselection','approved':valid,'reason':reason,'parent':action['sql'],'candidate':self.saved['sql'],'feedback_only':self.arm=='feedback_only'})
            if valid:return {'tool':'submit','sql':self.saved['sql']}
        if self.used or schema is None or not obs['remaining']:return action
        terminal=action['tool']=='submit';checks=[h for h in obs['history'] if h['tool']=='check']
        early=self.arm!='terminal' and obs['remaining']>=4 and checks and not passed(checks[-1]['observation'])
        if not terminal and not early:return action
        current=action.get('sql',checks[-1]['sql'] if checks else obs['issue_sql']);checked={structural_key(h['sql'],schema) for h in obs['history'] if h['tool'] in ['check','execute']};pool=[];reasons=[];plan=None;kind=None
        if self.arm in ['edges','combined','terminal','feedback_only','schema_only']:
            plan,reason=E.request(current,obs,schema);reasons.append({'source':'edge','reason':reason})
            if plan and self.arm=='schema_only':pool=[E.compile_candidate(plan,obs)];plan=None
            elif plan:kind='edges'
        if not plan and not pool and self.arm in ['items','combined','terminal','feedback_only']:
            pool,reason=I.generate(current,obs,schema,self.support);reasons.append({'source':'items','reason':reason})
            if not pool:plan,reason=I.request(current,obs,schema,self.support['binary_state_literal']);kind='items' if plan else None;reasons.append({'source':'state','reason':reason})
        self.telemetry.append({'stage':'generated','pool':pool,'plan':plan,'kind':kind,'reasons':reasons,'current':current,'remaining':obs['remaining'],'early':not terminal})
        eligible=[c for c in pool if structural_key(c['sql'],schema) not in checked and approve(c,current,obs,schema,self.support)[0]]
        if eligible:
            self.used=True;self.held=action;self.snapshot=copy.deepcopy(vars(self.base));return self.send({**eligible[0],'parent':current},obs)
        if plan and obs['remaining']>=(2 if terminal else 4):
            if any(h['tool']=='execute' and h.get('sql')==plan['sql'] for h in obs['history']):return action
            self.used=True;self.held=action;self.snapshot=copy.deepcopy(vars(self.base));self.probe=(kind,plan);self.telemetry.append({'stage':'request_probe','kind':kind,'plan':plan,'remaining':obs['remaining']});return {'tool':'execute','sql':plan['sql']}
        return action

def create_agent(checkpoint=None):return Agent(checkpoint)
