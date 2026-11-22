"""TERMINAL-QP2 timing-only intervention over immutable Integration1 machinery.

QP-R, Evidence/compare, token/model meter and candidate validation are imported
unchanged. Only a genuine completed parent's submit may open the extra review.
No model, schema or check is inserted before that action. No parent continuation
afterwards can consume review-origin feedback or choose a weaker held SQL.
"""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import step107_qwen_integration1_agent as frozen
from step107_qwen_integration1_evidence import analyze,record
from step107_qwen_proactive_protocol import ParentFrame,opportunity

PARENTS={
 'polarity':('step107_polarity_selected','52ddb4eb2c7616987ac628f3610aa8f8b435821d52a1759c74f8a3fbda823bd4'),
 'graph439':('step_graph439_frozen','8c24ea878e6eff0376d251900372f8241bc86cb1f6d46a2efc6b8a0f3e481d86')}
CONFIG='outputs/step107-terminal-qp2-preregister.json'


class TimedAudit(list):
    def __init__(self,owner):super().__init__();self.owner=owner
    def append(self,event):
        obs=getattr(self.owner,'actual_obs',None)
        super().append({**event,'monotonic_ns':time.perf_counter_ns(),
            'actual_history_length':len(obs['history']) if obs else None,
            'actual_remaining':obs['remaining'] if obs else None})


class Agent(frozen.Agent):
    def __init__(self,parent,adapter_sha,policy='margin0_tie_candidate',proposer=None):
        super().__init__(parent,adapter_sha,policy,'repair_original',proposer)
        self.final_action=None;self.actual_obs=None;self.audit=TimedAudit(self)

    def act(self,obs):
        self.actual_obs=obs
        return super().act(obs) # Original CURRENT owner and same-bundle meter.

    def _act(self,obs):
        self.calls+=1
        if self.calls>9:raise RuntimeError('Official episode step cap')
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.phase=='schema':
            assert self.final_action is not None and schema is not None
            return self.generate_review(obs,schema) # Exact old generation/guards.
        if self.phase=='check':
            assert self.final_action is not None
            assert obs['history'][-1]['tool']=='check' and obs['history'][-1].get('sql')==self.candidate
            evidence,facts=analyze(obs,self.candidate,schema or [])
            if evidence.status=='HARD_CONFLICT' or not frozen.public(obs,self.candidate):
                self.rejected.add(frozen.identity(self.candidate))
            self.audit.append({'stage':'checked','candidate':self.candidate,
                'public_pass':frozen.public(obs,self.candidate),'evidence':record(evidence),'facts':facts})
            self.phase=None
            if frozen.identity(self.candidate) in self.rejected:
                self.audit.append({'stage':'terminal_reject_keep_final','action':self.final_action})
                return copy.deepcopy(self.final_action)
            return self.finish(obs,copy.deepcopy(self.final_action),schema or [])
        if self.phase=='restore':
            # Inherited generate_review uses this on invalid/duplicate/capped
            # proposals. The held action is already the genuine final submit.
            self.phase=None
            assert self.frame.parent is self.parent and self.frame.restore_action()==self.final_action
            self.audit.append({'stage':'restore_terminal_submit','action':self.final_action})
            if self.candidate and frozen.identity(self.candidate) in self.rejected:
                # A rejected proposal may be identical to an already bad parent.
                # Its review provenance must not blacklist/rewrite that genuine
                # final action into some earlier fallback. Rejection means keep
                # the exact completed parent, irrespective of offline truth.
                self.audit.append({'stage':'terminal_reject_keep_final','action':self.final_action})
                return copy.deepcopy(self.final_action)
            return self.finish(obs,copy.deepcopy(self.final_action),schema or [])
        assert self.final_action is None,'Never advance a completed parent after review'
        try:action=self.parent.act(copy.deepcopy(obs))
        except frozen.CompletionCap:
            # Meter behavior is unchanged. Do not pretend a partially interrupted
            # Python stack returned a final parent action or open an extra probe.
            self.used=True
            self.audit.append({'stage':'parent_completion_cap_interruption'})
            return {'tool':'submit','sql':self.fallback(obs)}
        self.audit.append({'stage':'parent_action','action':action})
        if 'sql' in action:
            self.owned_sql.add(frozen.identity(action['sql']));self.parent_sql=action['sql']
        if action['tool']!='submit':return action
        self.final_action=copy.deepcopy(action)
        self.audit.append({'stage':'parent_final','action':action,'genuine_parent_final':True})
        if self.used:return self.finish(obs,action,schema or [])
        evidence,facts=analyze(obs,action['sql'],schema or [])
        allowed,reason=opportunity(obs,action,evidence,self.ledger)
        self.audit.append({'stage':'opportunity','parent_action':action,'eligible':allowed,
                           'reason':reason,'evidence':record(evidence),'facts':facts})
        if not allowed:return self.finish(obs,action,schema or [])
        self.used=True;self.frame=ParentFrame.capture(self.parent,action,obs)
        self.pre_review_owned=set(self.owned_sql)
        if schema is None:self.phase='schema';return {'tool':'schema'}
        return self.generate_review(obs,schema)


def configuration():
    config=json.loads(Path(os.environ.get('MP3_TQP_CONFIG',CONFIG)).read_text(encoding='utf-8'))
    key=os.environ.get('MP3_TQP_PARENT','polarity')
    if key not in PARENTS:raise ValueError('Only the two preregistered immutable parent entries are permitted')
    module,expected=PARENTS[key];path=Path('submission')/(module+'.py')
    assert config['parents'][key]['entry']==path.as_posix()
    assert config['parents'][key]['entry_sha256']==expected==hashlib.sha256(path.read_bytes()).hexdigest()
    return config,key,module


def create_agent(checkpoint=None):
    config,key,module=configuration()
    adapter=config['adapter'];directory=Path(os.environ['MP3_ADAPTER_DIR'])
    for name,expected in adapter['sha256'].items():
        with (directory/name).open('rb') as file:assert hashlib.file_digest(file,'sha256').hexdigest()==expected
    frozen.install_meter()
    factory=importlib.import_module(module).create_agent
    return Agent(factory(checkpoint),adapter['sha256']['adapter_model.safetensors'],
                 os.environ.get('MP3_TQP_POLICY','margin0_tie_candidate'))
