import json
from scope_controller import Agent
from step107_polarity_selected import create_agent as parent_factory
class ModuleParent:
 def act(self,o):
  if not o['history']:return {'tool':'check','sql':o['issue_sql']}
  if any(h['tool']=='check' and h['sql']==o['issue_sql'] and h['observation'].get('matches_example') for h in o['history']):return {'tool':'submit','sql':o['issue_sql']}
  if not any(h['tool']=='schema' for h in o['history']):return {'tool':'schema'}
  return {'tool':'submit','sql':o['issue_sql']}
def create_agent(checkpoint=None):
 cfg={'arm':'integrated'}
 if checkpoint:
  with open(checkpoint,encoding='utf-8') as f:cfg=json.load(f)
 arm=cfg['arm']
 if arm=='parent':return parent_factory(cfg.get('parent_checkpoint'))
 if arm=='module_only':return Agent(ModuleParent(),'direct_ast')
 if arm=='integrated':return Agent(parent_factory(cfg.get('parent_checkpoint')),'direct_ast')
 raise ValueError('Unknown frozen arm')
