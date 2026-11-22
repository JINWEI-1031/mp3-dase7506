"""B+C integration deliberately excludes weak raw multi-seed replacements."""
from step_space435_provider import candidates as original,zero_conflict,probe,apply_probe
def candidates(current,obs,schema,arm='bc',cap=24):
    pool,diag=original(current,obs,schema,'b',cap);out=[]
    for c in pool:
        reason=zero_conflict(c['sql'],current,obs,schema)
        if reason=='weak_zero_population_domain':c={**c,'confidence':min(c['confidence'],4)};diag['rejections'].append({'reason':reason,'policy':'retained_low_priority'})
        elif reason:diag['rejections'].append({'reason':reason});continue
        out.append(c)
    return sorted(out,key=lambda c:-c['confidence']),diag
