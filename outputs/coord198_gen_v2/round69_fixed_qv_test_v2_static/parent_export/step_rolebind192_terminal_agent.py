"""Causal interception with held parent actions, real budget and full telemetry."""
import os
from step_recovery192_balanced import Agent as Recovery
from step_recovery192 import signature
from step4_agent import passed,identity
from step_pool192_rules import schema_errors
from step_rolebind192_rules import signals
from step_rolebind192_terminal_rules import generate

class Agent:
    def __init__(self,checkpoint=None,arm=None):
        self.base=Recovery(checkpoint);self.arm=arm or os.environ.get('MP3_ROLEBIND192_ARM','both')
        self.audit=[];self.pool=[];self.phase='rolebind192';self.held=None;self.pending=None;self.attempted=False;self.telemetry=[]
    def act(self,obs):
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.pending:
            ok=passed(obs['history'][-1]['observation']);self.audit[-1]['public_pass']=ok;sql=self.pending;self.pending=None
            if ok:return {'tool':'submit','sql':sql}
            self.telemetry.append({'stage':'restore_parent','remaining':obs['remaining']})
        if self.held is not None:
            action=self.held
        else:
            action=self.base.act(obs)
        if self.arm=='control':return action
        new_trigger=self.arm in ('trigger','both');new_generator=self.arm in ('generator','both')
        actionable=action['tool']=='submit'
        sql=action.get('sql',obs['issue_sql']);risk=signals(sql,obs,schema or [])
        required=3 if action['tool']=='check' else 1
        # Old schedule: only intervene after parent completion and observed schema.
        if not actionable or self.attempted or not risk or obs['remaining']<required or not new_trigger and schema is None:
            self.held=None;return action
        self.held=action
        if schema is None:
            if obs['remaining']>=required+1:return {'tool':'schema'}
            self.held=None;return action
        self.attempted=True
        seeds=[sql]
        self.pool=generate(seeds,obs,schema,new=new_generator,cap=24)
        checked={signature(e['sql'],schema) for e in obs['history'] if e['tool'] in ('check','execute')}
        event={'stage':'generated','risk':risk,'remaining':obs['remaining'],'parent_action':action,'pool':self.pool};self.telemetry.append(event)
        for item in self.pool:
            if signature(item['sql'],schema) in checked or schema_errors(item['sql'],obs,schema):continue
            # Weak generic temporal alternatives cannot override an existing match.
            if item['confidence']<5:continue
            self.pending=item['sql'];self.audit.append({**item,'remaining':obs['remaining']});return {'tool':'check','sql':self.pending}
        self.held=None;return action

def create_agent(checkpoint=None):return Agent(checkpoint)
