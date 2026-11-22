"""Mechanism ablations in new controller objects; frozen modules untouched."""
from step107_polarity_selected import create_agent as complete
from step107_frontier_agent import Agent as Frontier,eligible
from step4_agent import identity,passed

ARMS=('control','without_category_alias','late_measure_review','keep_public_parent')

class KeepPublicParent(Frontier):
    def best(self,obs,schema):
        proposed=super().best(obs,schema);parent=self.fallback
        matched=parent and any(e['tool']=='check' and identity(e['sql'])==identity(parent['sql']) and passed(e['observation']) for e in obs['history'])
        if matched and eligible(parent['sql'],obs):
            if identity(proposed['sql'])!=identity(parent['sql']):
                self.counterfactual_events.append({'kind':'preserve_checked_eligible_parent','remaining':obs['remaining'],
                    'parent_sql':parent['sql'],'otherwise_selected_sql':proposed['sql']})
            return parent
        return proposed

def configure(node,arm,path='root',changes=None):
    changes=[] if changes is None else changes
    module=type(node).__module__
    if arm=='without_category_alias' and module=='step106_agent' and type(node).__name__=='Agent':
        changes.append({'mechanism':'development_category_alias','path':path,'action':'bypass optional suffix; retain its full baseline'})
        return node.baseline,changes
    if arm=='late_measure_review' and module=='step107_transfer_agent':
        changes.append({'mechanism':'measure_review_order','path':path,'before':node.schedule,'after':'late'})
        node.schedule='late'
    if arm=='keep_public_parent' and module=='step107_frontier_agent':
        changed=KeepPublicParent.__new__(KeepPublicParent);changed.__dict__.update(node.__dict__);changed.counterfactual_events=[];node=changed
        changes.append({'mechanism':'public_parent_override','path':path,'action':'retain checked eligible fallback in best(); generation and semantic eligibility unchanged'})
    for name in ('base','baseline'):
        child=getattr(node,name,None)
        if child is not None and hasattr(child,'act'):
            replacement,changes=configure(child,arm,path+'.'+name,changes);setattr(node,name,replacement)
    return node,changes

def graph(node,path='root'):
    records=[{'path':path,'module':type(node).__module__,'class':type(node).__name__,
              'phase':getattr(node,'phase',None),'schedule':getattr(node,'schedule',None)}]
    for name in ('base','baseline'):
        child=getattr(node,name,None)
        if child is not None and hasattr(child,'act'):records+=graph(child,path+'.'+name)
    return records

def events(node):
    records=[]
    if hasattr(node,'counterfactual_events'):records+=node.counterfactual_events
    for name in ('base','baseline'):
        child=getattr(node,name,None)
        if child is not None and hasattr(child,'act'):records+=events(child)
    return records

class Agent:
    def __init__(self,checkpoint=None,arm='control'):
        assert arm in ARMS
        self.base,self.configuration=configure(complete(checkpoint),arm);self.arm=arm
        if arm!='control':assert len(self.configuration)==1,self.configuration
    def act(self,obs):return self.base.act(obs)

def create_agent(checkpoint=None):return Agent(checkpoint)
