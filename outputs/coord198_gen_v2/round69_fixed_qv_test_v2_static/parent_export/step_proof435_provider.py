"""Independent IR plus a separately measurable existing root-name guard."""
from step_proof435_ir import generate
from step_proof435_evidence import name_conflict
from step107_independent_plan import plan as transferred_name_plan
from step_space435_common import diverse

def candidates(current,obs,schema,generator='extended',tier='preserve'):
    if generator=='name':
        conflicts=name_conflict(current,obs,schema);items=[]
        if conflicts:
            for c in transferred_name_plan(obs,schema):items.append({'sql':c['sql'],'source':'proof435_name_transfer','family':'name_plan','path':['independent_name_fk_plan'],'evidence':{**c['evidence'],'contradictions':conflicts},'confidence':11})
        reasons=[] if conflicts else [{'reason':'no_explicit_name_role_conflict'}]
    else:items,reasons=generate(current,obs,schema,extended=generator=='extended',controls=True)
    pool,rejected=diverse(items,obs,schema,24);return pool,{'rejections':reasons+rejected,'candidate_cap':24,'generator':generator}
