"""Keep zero-event coverage; require strong domain evidence before checking it."""
import re
from step_evidence28_192_composed import generate as composed,signals

def explicit_zero_domain(question):
    return bool(re.search(r'\ball\b|\bfor\s+(?:each|every)\b|\b(?:including|include)\b.{0,50}\b(?:zero|no|without)\b',question,re.I))

def generate(seeds,obs,schema,*,new=True,cap=24,mode='scoped'):
    pool=composed(seeds,obs,schema,new=new,cap=cap)
    strong=explicit_zero_domain(obs['task']['question'])
    for c in pool:
        if c['path'][0]=='zero_event_left' or c.get('evidence',{}).get('include_zero_entities'):
            if not strong:c['confidence']=4
            c['evidence']={**c.get('evidence',{}),'zero_domain_evidence':'explicit_all_or_for_each' if strong else 'ambiguous_each_domain'}
    return sorted(pool,key=lambda c:-c['confidence'])[:cap]
