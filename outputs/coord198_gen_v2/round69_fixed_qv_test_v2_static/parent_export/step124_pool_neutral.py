"""Do not infer bag semantics merely from the absence of the word DISTINCT."""
import re
from sqlglot import exp
from repair_rules import parse
from step124_pool_scoped import score as scoped_score

def score(sql,obs,schema):
    value=scoped_score(sql,obs,schema);tree=parse(sql)
    if tree is not None and not re.search(r'\b(?:distinct|different|unique)\b',obs['task']['question'],re.I):
        value+=1.2*sum(bool(s.args.get('distinct')) for s in tree.find_all(exp.Select))
    return value

def create_agent(checkpoint=None):
    import step124_pool_agent as controller
    import step124_pool_search as search
    controller.score=score;search.score=score
    return controller.Agent(checkpoint)
