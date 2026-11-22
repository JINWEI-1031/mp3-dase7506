"""At most two evidence edits: preserve a structural edit while binding outputs."""
from repair_rules import parse
from step4_agent import identity
from step_evidence28_192_rules import generate as single,projection,signals
from step_semantic192_rules import typed_literals
from step_pool192_rules import schema_errors

def generate(seeds,obs,schema,*,new=True,cap=24,mode='composed'):
    pool=single(seeds,obs,schema,new=new,cap=cap,mode='combined');out=list(pool)
    for first in pool[:8]:
        if first['path'][0] not in ('zero_event_left','evidenced_bridge','count_event_role'):continue
        tree=parse(first['sql'])
        if tree is None:continue
        for second in projection(tree,obs,schema):
            if identity(second['sql'])==identity(first['sql']):continue
            if not typed_literals(second['sql'],schema) or schema_errors(second['sql'],obs,schema):continue
            out.append({**second,'path':first['path']+second['path'],'confidence':max(first['confidence'],second['confidence'])+1,'evidence':{'first':first['evidence'],'second':second['evidence']},'source':'evidence28_192_composed'})
    unique={}
    for c in out:
        key=identity(c['sql'])
        if key not in unique or unique[key]['confidence']<c['confidence']:unique[key]=c
    return sorted(unique.values(),key=lambda c:-c['confidence'])[:cap]
