"""Once-only grounded intervention with exact pending-action parent recovery."""
from step107_independent_selected import create_agent as baseline
from step107_chainnext_resume import Suspension
from step107_scopeflow_rules import projection
from step4_agent import passed, identity
from step107_scopeflow_sets import request, generate

ARMS = ('control', 'terminal', 'early', 'trigger', 'sets', 'early_sets')


class Agent:
    def __init__(self, checkpoint=None, arm='early'):
        self.base = baseline(checkpoint)
        self.arm = arm
        self.used = False
        self.pending = None
        self.suspension = None
        self.audit = []
        self.held = None
        self.probe = None

    def act(self, obs):
        if self.probe:
            plan = self.probe; self.probe = None
            candidate, reason = generate(plan, obs['history'][-1]['observation'])
            self.audit.append({'stage': 'probe_feedback', 'reason': reason})
            if candidate is not None and obs['remaining']:
                self.pending = candidate
                self.audit.append({'stage': 'generated_and_sent', 'sql': candidate})
                return {'tool': 'check', 'sql': candidate}
            self.base = self.suspension.parent
            self.audit.append({'stage': 'restore_parent', 'remaining': obs['remaining']})
            return self.suspension.resume(obs)
        if self.pending:
            sql = self.pending
            self.pending = None
            ok = passed(obs['history'][-1]['observation'])
            self.audit.append({'stage': 'checked', 'public_pass': ok, 'remaining': obs['remaining']})
            if ok:
                self.audit.append({'stage': 'selected', 'sql': sql})
                return {'tool': 'submit', 'sql': sql}
            self.base = self.suspension.parent
            self.audit.append({'stage': 'restore_parent', 'remaining': obs['remaining']})
            return self.suspension.resume(obs)
        action = self.held if self.held is not None else self.base.act(obs)
        self.held = None
        if self.arm == 'control' or self.used:
            return action
        terminal = action['tool'] == 'submit'
        if not terminal and (self.arm in ('terminal', 'sets') or action['tool'] != 'check'):
            return action
        schema = next((e['observation'] for e in obs['history'] if e['tool'] == 'schema'), None)
        if terminal and self.arm in ('sets', 'early_sets'):
            if schema is None:
                # Only request schema for an explicit, bounded two-category intent.
                import re
                if obs['remaining'] >= 3 and re.search(r'\b(?:both [a-z]+ and [a-z]+|[a-z]+ but not [a-z]+)[?.]?$', obs['task']['question'].lower().strip()):
                    self.held = action
                    self.audit.append({'stage': 'request_schema', 'remaining': obs['remaining']})
                    return {'tool': 'schema'}
            else:
                plan, reason = request(action['sql'], obs, schema)
                self.audit.append({'stage': 'set_diagnosis', 'reason': reason, 'remaining': obs['remaining']})
                if plan is not None and obs['remaining'] >= 2:
                    self.used = True; self.suspension = Suspension(self.base, action); self.probe = plan
                    self.audit.append({'stage': 'evidence', 'reason': 'two_category_scope', 'early': False, 'remaining': obs['remaining']})
                    return {'tool': 'execute', 'sql': plan['probe']}
            if self.arm == 'sets':return action
        if schema is None:
            return action
        candidate, reason = projection(action['sql'], obs, schema)
        if candidate is None:
            return action
        seen = {identity(e['sql']) for e in obs['history'] if e['tool'] == 'check'}
        self.audit.append({'stage': 'evidence', 'early': not terminal, 'reason': reason,
                           'remaining': obs['remaining'], 'sql': candidate})
        # One extra charged check and the held parent action must both fit.
        if obs['remaining'] < (1 if terminal else 2) or identity(candidate) in seen:
            self.audit.append({'stage': 'rejected', 'reason': 'budget_or_already_checked'})
            return action
        self.used = True
        self.audit.append({'stage': 'opportunity', 'early': not terminal})
        if self.arm == 'trigger':
            return action
        self.suspension = Suspension(self.base, action)
        self.pending = candidate
        self.audit.append({'stage': 'generated_and_sent', 'sql': candidate})
        return {'tool': 'check', 'sql': candidate}


def create_agent(checkpoint=None):
    return Agent(checkpoint)
