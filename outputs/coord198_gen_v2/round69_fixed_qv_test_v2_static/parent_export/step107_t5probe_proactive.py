"""Independent fifth arm: native source replacement + ungated terminal review."""
import copy
from step107_t5probe_agent import Agent as Native
from step107_t5probe_contract import approve,inspect
from step107_feedback_review import feedback
from step4_agent import identity

class Agent:
    def __init__(self,checkpoint=None,provider=None,uncertainty_policy='schema_supported'):
        self.base=Native(checkpoint,arm='native_replace',provider=provider,uncertainty_policy=uncertainty_policy)
        self.uncertainty_policy=uncertainty_policy;self.phase='base';self.held_action=None;self.pending=None;self.audit=[]

    def act(self,obs):
        if self.phase=='done':return copy.deepcopy(self.held_action)
        if self.phase=='checked':
            allowed,certificate=approve(self.held_action['sql'],self.pending,obs,mode='soft',uncertainty_policy=self.uncertainty_policy)
            self.audit.append({'event':'proactive_decision','remaining':obs['remaining'],'accepted':allowed,'certificate':certificate})
            self.phase='done'
            return {'tool':'submit','sql':self.pending} if allowed else copy.deepcopy(self.held_action)
        if self.phase=='base':
            action=self.base.act(obs)
            if action['tool']!='submit':return action
            self.held_action=copy.deepcopy(action);self.phase='review'
            self.audit.append({'event':'parent_submit_held','remaining':obs['remaining'],'parent_certificate':inspect(action['sql'],obs)})
        if not obs['remaining']:
            self.phase='done';self.audit.append({'event':'proactive_no_budget'});return copy.deepcopy(self.held_action)
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if schema is None:
            if obs['remaining']<2:
                self.phase='done';self.audit.append({'event':'proactive_schema_plus_check_unavailable'});return copy.deepcopy(self.held_action)
            return {'tool':'schema'}
        # No diagnose/hard-conflict trigger here. Native owns the one-completion
        # episode memo; reuse is only within this actual controller instance.
        sql=self.base.native(obs,schema,'proactive_terminal_review')
        if not sql:
            self.phase='done';return copy.deepcopy(self.held_action)
        if identity(sql)==identity(self.held_action['sql']):
            self.phase='done';self.audit.append({'event':'proactive_same_parent'});return copy.deepcopy(self.held_action)
        observed=feedback(sql,obs)
        if observed is not None and 'matches_example' in observed:
            allowed,certificate=approve(self.held_action['sql'],sql,obs,mode='soft',uncertainty_policy=self.uncertainty_policy)
            self.audit.append({'event':'proactive_existing_feedback','accepted':allowed,'certificate':certificate});self.phase='done'
            return {'tool':'submit','sql':sql} if allowed else copy.deepcopy(self.held_action)
        self.pending=sql;self.phase='checked'
        self.audit.append({'event':'proactive_sent_check','remaining':obs['remaining'],'candidate':sql})
        return {'tool':'check','sql':sql}

def create_agent(checkpoint=None):return Agent(checkpoint)
