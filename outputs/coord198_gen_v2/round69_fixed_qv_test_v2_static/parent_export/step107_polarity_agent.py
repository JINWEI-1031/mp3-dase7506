"""Complete historical chain plus one terminal evidence-bound correction."""
from step107_independent_selected import create_agent as complete_parent
from step107_scopeflow_selected import create_agent as historical_control
from step107_chainnext_resume import Suspension
from step107_polarity_ir import request, generate, question_ir
from step107_polarity_clauses import candidates, signals
from step107_polarity_role import request as role_request, generate as role_generate
from step4_agent import passed, identity

ARMS=('control','guard','scoped','projection','exclusion','clauses','combined','role_terminal','role_early')


class Agent:
    def __init__(self, checkpoint=None, arm='guard'):
        self.arm=arm
        self.base=historical_control(checkpoint) if arm=='control' else complete_parent(checkpoint)
        self.held=None;self.pending=None;self.probe=None;self.role_probe=None;self.used=False;self.suspension=None;self.audit=[]

    def schedule_role(self,action,obs,schema,early):
        if schema is None:return None
        plan,reason=role_request(action['sql'],obs,schema)
        if plan is None:return None
        self.audit.append({'stage':'role_evidence','reason':reason,'early':early,'remaining':obs['remaining'],
                           'field':plan['field'],'question_phrase':plan['question_phrase']})
        if obs['remaining']<(3 if early else 2):
            self.audit.append({'stage':'rejected','reason':'role_probe_check_and_held_action_budget'});return None
        if any(e['tool']=='execute' and e['sql']==plan['probe'] for e in obs['history']):
            self.audit.append({'stage':'rejected','reason':'role_probe_already_used'});return None
        self.suspension=Suspension(self.base,action);self.role_probe=plan;self.used=True
        self.audit.append({'stage':'request_role_probe','early':early,'held_tool':action['tool']})
        return {'tool':'execute','sql':plan['probe']}

    def restore(self,obs):
        self.used=True;self.base=self.suspension.parent
        self.audit.append({'stage':'restore_parent','remaining':obs['remaining']})
        return self.suspension.resume(obs)

    def act(self,obs):
        if self.arm=='control':return self.base.act(obs)
        if self.role_probe:
            plan=self.role_probe;self.role_probe=None;sql,reason=role_generate(plan,obs['history'][-1]['observation'])
            self.audit.append({'stage':'role_feedback','reason':reason})
            seen={identity(e['sql']) for e in obs['history'] if e['tool']=='check'}
            if sql is not None and obs['remaining'] and identity(sql) not in seen:
                self.pending=sql;self.audit.append({'stage':'generated_and_sent','source':'observed_role_value','sql':sql})
                return {'tool':'check','sql':sql}
            return self.restore(obs)
        if self.pending:
            sql=self.pending;self.pending=None;ok=passed(obs['history'][-1]['observation'])
            self.audit.append({'stage':'checked','public_pass':ok,'remaining':obs['remaining']})
            if ok:
                self.audit.append({'stage':'selected','sql':sql});return {'tool':'submit','sql':sql}
            return self.restore(obs)
        if self.probe:
            plan=self.probe;self.probe=None;sql,reason=generate(plan,obs['history'][-1]['observation'])
            self.audit.append({'stage':'probe_feedback','reason':reason})
            if sql is not None and obs['remaining']:
                self.pending=sql;self.audit.append({'stage':'generated_and_sent','source':'quantifier_ir','sql':sql,'ir':plan['ir']})
                return {'tool':'check','sql':sql}
            return self.restore(obs)
        action=self.held if self.held is not None else self.base.act(obs);self.held=None
        if self.used:return action
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        terminal=action['tool']=='submit'
        if self.arm in ('role_terminal','role_early') and (terminal or self.arm=='role_early' and action['tool']=='check'):
            proposed=self.schedule_role(action,obs,schema,not terminal)
            if proposed:return proposed
        if not terminal:return action
        ir,reason=question_ir(obs['task']['question'])
        local_signals=signals(action['sql'],obs) if self.arm in ('projection','exclusion','clauses','combined') else []
        if self.arm=='projection':local_signals=[s for s in local_signals if s=='observed_unknown_column']
        if self.arm=='exclusion':local_signals=[s for s in local_signals if s=='exact_attribute_exclusion_vs_substring']
        if ir is None and not local_signals:return action
        self.audit.append({'stage':'evidence','ir':ir,'local_signals':local_signals,'remaining':obs['remaining']})
        schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        if schema is None:
            need=3 if ir is not None else 2
            if obs['remaining']<need:
                self.audit.append({'stage':'rejected','reason':'schema_plus_trial_budget'});return action
            self.held=action;self.audit.append({'stage':'request_schema','remaining':obs['remaining']});return {'tool':'schema'}
        self.suspension=Suspension(self.base,action)
        seen={identity(e['sql']) for e in obs['history'] if e['tool']=='check'}
        if local_signals:
            pool,rejections=candidates(action['sql'],obs,schema)
            if self.arm=='projection':pool=[c for c in pool if c['source']=='unknown_projection_contract']
            if self.arm=='exclusion':pool=[c for c in pool if c['source']=='exact_attribute_exclusion']
            self.audit.append({'stage':'clause_pool','generated':len(pool),'rejections':rejections})
            for c in pool:
                if identity(c['sql']) in seen:continue
                if not obs['remaining']:
                    self.audit.append({'stage':'rejected','reason':'check_budget'});return action
                self.used=True;self.pending=c['sql'];self.audit.append({'stage':'generated_and_sent',**c})
                return {'tool':'check','sql':self.pending}
        if ir is not None:
            plan,reason=request(action['sql'],obs,schema,scoped=self.arm in ('scoped','combined'))
            self.audit.append({'stage':'quantifier_plan','reason':reason,'generated':int(plan is not None)})
            if plan is not None and obs['remaining']>=2:
                self.used=True;self.probe=plan;self.audit.append({'stage':'request_probe','evidence':plan['evidence']})
                return {'tool':'execute','sql':plan['probe']}
            if plan is not None:self.audit.append({'stage':'rejected','reason':'probe_plus_check_budget'})
        self.used=True
        return action


def create_agent(checkpoint=None):return Agent(checkpoint)
