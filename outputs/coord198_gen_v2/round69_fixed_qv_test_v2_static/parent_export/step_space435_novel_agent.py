"""One grounded review, optional charged evidence, and causal held-action recovery."""
import re
from sqlglot import exp
from repair_rules import parse
from step_evidence28_192_frozen import create_agent as baseline
from step_space435_novel_provider import candidates,probe,apply_probe
from step_space435_common import structural_key,guard
from step_pool192_rules import schema_errors
from step4_agent import passed

class Agent:
    def __init__(self,checkpoint=None,arm='abc'):
        self.base=baseline(checkpoint);self.arm=arm;self.audit=[];self.pool=[];self.telemetry=[];self.phase='space435';self.held=None;self.pending=None;self.probe_pending=None;self.used=False;self.generated_keys=set()
    def choose(self,current,obs,schema,early=False):
        cap=48 if self.arm=='a48' else 24;raw,diagnostic=candidates(current,obs,schema,self.arm,cap);self.pool=[]
        for c in raw:
            key=structural_key(c['sql'],schema)
            if key not in self.generated_keys and len(self.generated_keys)>=cap:diagnostic['rejections'].append({'reason':'episode_candidate_cap'});continue
            self.generated_keys.add(key);self.pool.append(c)
        diagnostic['unique_candidates_this_episode']=len(self.generated_keys)
        self.telemetry.append({'stage':'generated','early':early,'remaining':obs['remaining'],'current':current,'pool':self.pool,**diagnostic})
        checked={structural_key(e['sql'],schema) for e in obs['history'] if e['tool'] in ('check','execute')}
        for c in self.pool:
            if structural_key(c['sql'],schema) in checked or structural_key(c['sql'],schema)==structural_key(current,schema):continue
            confidence=c['confidence']
            if early and c['path'][0]=='negative_relation' and schema_errors(current,obs,schema)>0:confidence=max(confidence,9)
            if confidence<(9 if early else 5):continue
            self.audit.append({**c,'early':early,'remaining':obs['remaining'],'effective_confidence':confidence});self.pending=c['sql'];self.used=True;return {'tool':'check','sql':self.pending}
        return None
    def act(self,obs):
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if self.pending:
            sql=self.pending;self.pending=None;ok=passed(obs['history'][-1]['observation']);self.audit[-1]['public_pass']=ok
            if ok:return {'tool':'submit','sql':sql}
            self.telemetry.append({'stage':'restore_parent','remaining':obs['remaining']});action=self.held;self.held=None;return action
        if self.probe_pending:
            plan=self.probe_pending;self.probe_pending=None;cs,reason=apply_probe(plan,obs['history'][-1]['observation'],obs,schema);self.telemetry.append({'stage':'supplemental_evidence','reason':reason,'candidate_count':len(cs)})
            if cs:
                key=structural_key(cs[0]['sql'],schema);cap=48 if self.arm=='a48' else 24
                if key not in self.generated_keys and len(self.generated_keys)>=cap:cs=[];self.telemetry.append({'stage':'guard_reject','reason':'episode_candidate_cap'})
                else:self.generated_keys.add(key)
            if cs and obs['remaining']:
                c=cs[0];self.audit.append({**c,'remaining':obs['remaining'],'early':self.held['tool']!='submit'});self.pending=c['sql'];return {'tool':'check','sql':self.pending}
            action=self.held;self.held=None;return action
        action=self.held if self.held is not None else self.base.act(obs)
        if self.used:self.held=None;return action
        checks=[e for e in obs['history'] if e['tool']=='check'];current=action.get('sql',checks[-1]['sql'] if checks else obs['issue_sql'])
        terminal=action['tool']=='submit';early=self.arm in ('c','abc','bc','novel') and not terminal and action['tool']!='check' and schema is not None and obs['remaining']>=4 and checks and not passed(checks[-1]['observation'])
        # Existing valid queued checks always execute before any new review.
        if not terminal and not early:return action
        if not obs['remaining']:
            self.telemetry.append({'stage':'not_triggered','reason':'budget_exhausted'});return action
        if schema is None:
            if terminal and obs['remaining']>=2:self.held=action;return {'tool':'schema'}
            self.telemetry.append({'stage':'not_triggered','reason':'schema_budget'});return action
        self.held=action
        proposed=self.choose(current,obs,schema,early)
        if proposed:return proposed
        if self.arm in ('c','abc','bc','novel') and obs['remaining']>=(5 if early else 2):
            p=probe(current,obs,schema)
            if p:
                self.used=True;self.probe_pending=p;self.telemetry.append({'stage':'request_evidence','sql':p['sql'],'early':early,'remaining':obs['remaining']});return {'tool':'execute','sql':p['sql']}
        if terminal:self.used=True
        self.held=None;return action

def create_agent(checkpoint=None):return Agent(checkpoint)
