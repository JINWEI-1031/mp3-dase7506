"""Complete controller with one native T5 source; no D/Coder loading path."""
import copy
import hashlib
from step107_polarity_selected import create_agent as complete
from step107_t5probe_model import generate,serialize
from step107_feedback_review import diagnose,proof,feedback
from step107_frontier_agent import eligible
from step4_agent import identity,continuation_conflicts
from step107_t5probe_contract import inspect,approve

ARMS=('native_replace','local_review','strict_review','soft_review')

def forbidden(*args,**kwargs):raise RuntimeError('D/Coder model loading is disabled in the sole-T5 controller')

class Agent:
    def __init__(self,checkpoint=None,arm='native_replace',provider=None,uncertainty_policy='schema_supported'):
        assert arm in ARMS
        self.arm=arm;self.provider=provider or generate;self.memo={};self.audit=[];self.parent=None;self.phase='base';self.pending=None;self.uncertainty_policy=uncertainty_policy
        def proposer(obs,schema):
            if arm!='native_replace':
                self.audit.append({'event':'original_D_site_disabled','remaining':obs['remaining']});return ''
            return self.native(obs,schema,'original_generation_site')
        import step63_agent as old
        import step5_model
        import step107_agent as review
        old.active_model=forbidden;step5_model.load_model=forbidden
        old.generate=proposer;old.ModelSuffix.__init__.__defaults__=(proposer,)
        review.generate=proposer;review.ReviewSuffix.__init__.__defaults__=(proposer,)
        self.base=complete(checkpoint)

    def native(self,obs,schema,site):
        text=serialize(obs['task']['question'],schema);key=hashlib.sha256(text.encode()).hexdigest()
        if key in self.memo:
            sql=self.memo[key];self.audit.append({'event':'same_native_prompt_reused','site':site,'remaining':obs['remaining']})
        elif self.memo:
            self.audit.append({'event':'one_native_completion_cap','site':site});return ''
        else:
            sql=self.provider(obs['task']['question'],schema,obs['task']['id'],site);self.memo[key]=sql
            self.audit.append({'event':'native_generated','site':site,'remaining':obs['remaining'],'sql':sql,'prompt_sha256':key})
        # Retain the full inherited rule chain; also enforce its explicit value,
        # width and question-intent guards before a native proposal is sent.
        certificate=inspect(sql,obs)
        if certificate['hard_conflicts']:
            self.audit.append({'event':'native_rejected_precheck','site':site,'sql':sql,'certificate':certificate})
            return ''
        return sql

    def act(self,obs):
        if self.phase=='checked':
            ok,why=approve(self.parent['sql'],self.pending,obs,mode=self.arm.split('_')[0],uncertainty_policy=self.uncertainty_policy)
            self.audit.append({'event':'terminal_native_review','accepted':ok,'guard':why,'remaining':obs['remaining']})
            self.phase='done';return {'tool':'submit','sql':self.pending} if ok else copy.deepcopy(self.parent)
        if self.phase=='done':return copy.deepcopy(self.parent)
        if self.phase=='base':
            action=self.base.act(obs)
            if action['tool']!='submit' or self.arm=='native_replace':return action
            # The actual entire controller object remains held. Intercept only
            # its submit; rejection returns the exact pending action, no rollback.
            self.parent=copy.deepcopy(action);self.phase='review'
        if not obs['remaining'] or (not diagnose(self.parent['sql'],obs) and not inspect(self.parent['sql'],obs)['hard_conflicts']):
            self.phase='done';return copy.deepcopy(self.parent)
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if schema is None:
            if obs['remaining']<2:self.phase='done';return copy.deepcopy(self.parent)
            return {'tool':'schema'}
        sql=self.native(obs,schema,'terminal_guarded_review')
        if not sql or identity(sql)==identity(self.parent['sql']):self.phase='done';return copy.deepcopy(self.parent)
        seen=feedback(sql,obs)
        if seen is not None:
            ok,why=approve(self.parent['sql'],sql,obs,mode=self.arm.split('_')[0],uncertainty_policy=self.uncertainty_policy);self.phase='done'
            self.audit.append({'event':'existing_native_feedback_review','accepted':ok,'guard':why})
            return {'tool':'submit','sql':sql} if ok else copy.deepcopy(self.parent)
        self.pending=sql;self.phase='checked';return {'tool':'check','sql':sql}

def create_agent(checkpoint=None):return Agent(checkpoint)
