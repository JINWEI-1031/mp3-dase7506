"""Only new plan evidence may displace a publicly matched parent result."""
import re
from repair_rules import parse
from step4_agent import passed
from step_space435_common import structural_key
from step_space435_bc_provider import candidates as baseline,probe,apply_probe

def candidates(current,obs,schema,arm='novel',cap=24):
    pool,diag=baseline(current,obs,schema,'bc',cap);tree=parse(current)
    matched=any(e['tool']=='check' and passed(e['observation']) and structural_key(e['sql'],schema)==structural_key(current,schema) for e in obs['history'])
    out=[]
    for c in pool:
        if matched and c.get('source')!='space435_plan':diag['rejections'].append({'reason':'no_new_evidence_over_matched_parent','source':c.get('source')});continue
        if c['path'][0]=='extremal_population_output_plan':
            plan=c['plan'];same=plan['output_entity']==plan['ranked_entity'];all_requested=bool(re.search(r'\ball\b|\bevery\b',obs['task']['question'],re.I))
            if same and not all_requested:
                if matched and tree is not None and tree.args.get('order') and tree.args.get('limit'):
                    c={**c,'confidence':4};diag['rejections'].append({'reason':'rank_domain_already_resolved_by_parent','policy':'retained_low_priority'})
                elif plan.get('tie_policy')=='all':c={**c,'confidence':4};diag['rejections'].append({'reason':'tie_population_not_explicit','policy':'retained_low_priority'})
    
        out.append(c)
    return sorted(out,key=lambda c:-c['confidence']),diag
