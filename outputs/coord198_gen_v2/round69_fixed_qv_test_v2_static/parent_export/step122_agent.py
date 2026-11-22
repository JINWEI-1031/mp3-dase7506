"""E0: reject impossible semantic plans before charged schema reads."""
import re
from sqlglot import exp
from repair_rules import parse
from step117_agent import Agent as SemanticStage
from step119_agent import Agent as BridgeStage
from step120_agent import Agent as LatestStage


def entity_possible(obs,sql):
    t=parse(sql)
    if not isinstance(t,exp.Select) or any(t.args.get(k) for k in ('group','having','order','limit','distinct')):
        return False
    cols=[e.unalias() for e in t.expressions]
    if not cols or not all(isinstance(c,exp.Column) and c.table for c in cols) or len({c.table.lower() for c in cols})!=1:
        return False
    literals=[v.this for v in t.find_all(exp.Literal) if v.is_string]
    return len(literals)==1 and bool(re.search(r'\b'+re.escape(cols[0].table.lower())+r'\s+'+re.escape(literals[0].lower())+r'\b',obs['task']['question'].lower()))


class EfficientSemanticStage(SemanticStage):
    def act(self,obs):
        q=obs['task']['question'].lower()
        if not ('older than average' in q or 'average amount' in q):
            self.arms.discard('aggregate')
        return super().act(obs)


class EfficientBridgeStage(BridgeStage):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint);self.base=EfficientSemanticStage(checkpoint)

    def act(self,obs):
        action=super().act(obs)
        if action['tool']=='schema' and self.fallback is not None:
            t=parse(self.fallback['sql']);tables=list(t.find_all(exp.Table)) if t else [];joins=t.args.get('joins',[]) if t else []
            if any(x.alias for x in tables) or len(joins)!=1 or joins[0].args.get('side') or joins[0].args.get('kind') not in (None,'','INNER') or not isinstance(joins[0].args.get('on'),exp.EQ):
                return self.fallback
        return action


class Agent(LatestStage):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint);self.base=EfficientBridgeStage(checkpoint)

    def act(self,obs):
        action=super().act(obs)
        if action['tool']=='schema' and self.fallback is not None:
            q=obs['task']['question'].lower()
            negation=('negation' in self.arms and 'how many' in q and 'do not have' in q and bool(re.search(r'longer than\s+\d',q)) and len(obs['example']['columns'])==1)
            entity='entity' in self.arms and entity_possible(obs,self.fallback['sql'])
            if not negation and not entity:
                return self.fallback
        return action


def create_agent(checkpoint=None):return Agent(checkpoint)
