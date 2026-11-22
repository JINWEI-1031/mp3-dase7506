import copy
from step_graph439_qualified_agent import Agent as Parent
from step_graph439_contract import plan_key
from step_synth441_seeded_sketch import generate,static_check
from step_proof435_contract_agent import state_hash
from step4_agent import passed
class Agent:
    def __init__(self,checkpoint=None,arm='combined',fold=None):
        self.base=Parent(checkpoint,arm='combined',fold=fold);self.arm=arm;self.audit=[];self.telemetry=[];self.used=False;self.pending=None;self.queue=[];self.saved=None;self.snapshot=None;self.held=None
    def restore(self,obs):
        self.base.__dict__.clear();self.base.__dict__.update(copy.deepcopy(self.snapshot));self.telemetry.append({'stage':'restore_parent','exact_state_restored':state_hash(vars(self.base))==state_hash(self.snapshot),'held_action':self.held,'remaining':obs['remaining'],'budget_not_refunded':True});a=self.held;self.held=None;return a
    def send(self,c,obs):
        self.pending=c;self.audit.append({**c,'remaining':obs['remaining'],'held_action':self.held});return {'tool':'check','sql':c['sql']}
    def admissible(self,c,current,obs,schema):
        pool,_=generate(current,obs,schema,'combined');target=plan_key(c['sql'],schema)
        return target is not None and target!=plan_key(current,schema) and any(x['static_admit'] and (self.arm!='strict' or x['old_strict_admit']) and plan_key(x['sql'],schema)==target for x in pool)
    def act(self,obs):
        if self.arm=='control':return self.base.act(obs)
        schema=next((h['observation'] for h in obs['history'] if h['tool']=='schema'),None)
        if self.pending:
            c=self.pending;self.pending=None;ok=passed(obs['history'][-1]['observation']);self.audit[-1]['public_pass']=ok
            if not ok and self.queue and obs['remaining']:
                return self.send(self.queue.pop(0),obs)
            action=self.restore(obs)
            if ok:
                self.saved=c
                if action['tool']=='submit' and self.arm!='keep_parent':
                    self.telemetry.append({'stage':'forced_replacement','parent':action['sql'],'candidate':c['sql'],'uncertainty':c['uncertainty']});return {'tool':'submit','sql':c['sql']}
            return action
        action=self.base.act(obs)
        if action['tool']=='submit' and self.saved:
            valid=self.arm!='keep_parent' and self.admissible(self.saved,action['sql'],obs,schema)
            self.telemetry.append({'stage':'final_reselection','approved':valid,'parent':action['sql'],'candidate':self.saved['sql'],'uncertainty':self.saved['uncertainty']})
            if valid:return {'tool':'submit','sql':self.saved['sql']}
        if self.used or schema is None or not obs['remaining']:return action
        terminal=action['tool']=='submit';checks=[h for h in obs['history'] if h['tool']=='check'];early=self.arm!='terminal' and obs['remaining']>=3 and checks and not passed(checks[-1]['observation'])
        if not terminal and not early:return action
        current=action.get('sql',checks[-1]['sql'] if checks else obs['issue_sql']);family=self.arm if self.arm in ['count','extreme'] else 'combined';pool,reasons=generate(current,obs,schema,family)
        self.telemetry.append({'stage':'generated','pool':pool,'reasons':reasons,'current':current,'remaining':obs['remaining'],'early':not terminal})
        checked={plan_key(h['sql'],schema) for h in obs['history'] if h['tool']=='check'}
        eligible=[c for c in pool if c['static_admit'] and (self.arm!='strict' or c['old_strict_admit']) and plan_key(c['sql'],schema) not in checked]
        if not eligible:return action
        self.used=True;self.held=action;self.snapshot=copy.deepcopy(vars(self.base));self.queue=eligible[:2]
        return self.send(self.queue.pop(0),obs)
def create_agent(checkpoint=None):return Agent(checkpoint)
