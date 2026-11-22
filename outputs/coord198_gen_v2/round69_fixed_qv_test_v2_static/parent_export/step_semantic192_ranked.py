"""Rank field alternatives by question/field role agreement, never oracle labels."""
from repair_rules import parse
from sqlglot import exp
from step_semantic192_rules import tokens
from step_semantic192_agent import semantic_score as base_score
from step_semantic192_fresh import Agent as Fresh
def score(item,current,obs,schema):
    value=base_score(item,current,obs,schema);tree=parse(item['sql']);q=tokens(obs['task']['question'])
    if tree is not None:
        for expression in tree.expressions:
            for col in expression.find_all(exp.Column):
                for word in tokens(col.name)-{'id','code'}:
                    value+=1 if word in q else .5 if len(word)>=4 and any(len(v)>=4 and word[:4]==v[:4] for v in q) else 0
    return value
class Agent(Fresh):
    def __init__(self,checkpoint=None):
        super().__init__(checkpoint)
        import step_semantic192_agent as api
        api.semantic_score=score
def create_agent(checkpoint=None):
    agent=Agent(checkpoint);agent.semantic_families={'event','relation','count','negative'};return agent
