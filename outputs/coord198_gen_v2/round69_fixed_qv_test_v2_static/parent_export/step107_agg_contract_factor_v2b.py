"""PURE-V2B: polarity-safe COUNT spans; quantity mechanism stays frozen V2.

NL normalization is internal parser data, never a tool/model prompt. Negated
requirements receive no new global hard/support certificate in this revision.
"""
import re
from sqlglot import exp
import step107_agg_contract_factor_v2 as v2

_N=r'(-?\d+(?:\.\d+)?)'
_PATTERNS=[
    (r'\b(?:no|not)\s+more\s+than\s*'+_N,'<=','NEGATED_STRICT_UPPER'),
    (r'\b(?:no|not)\s+(?:fewer|less)\s+than\s*'+_N,'>=','NEGATED_STRICT_LOWER'),
    (r'\bnot\s+at\s+least\s*'+_N,'<','NEGATED_INCLUSIVE_LOWER'),
    (r'\bnot\s+at\s+most\s*'+_N,'>','NEGATED_INCLUSIVE_UPPER'),
    (r'\bwithout\s+at\s+least\s*'+_N,None,'WITHOUT_SCOPE_UNRESOLVED'),
    (r'\bnot\s+exactly\s*'+_N,None,'NON_SINGLE_INTERVAL_NEGATION'),
    (r'\bat\s+least\s*'+_N,'>=','POSITIVE_INCLUSIVE_LOWER'),
    (_N+r'\s*or\s*more\b','>=','POSITIVE_INCLUSIVE_LOWER'),
    (r'\bmore\s+than\s*'+_N,'>','POSITIVE_STRICT_LOWER'),
    (r'\bat\s+most\s*'+_N,'<=','POSITIVE_INCLUSIVE_UPPER'),
    (_N+r'\s*or\s*fewer\b','<=','POSITIVE_INCLUSIVE_UPPER'),
    (r'\bfewer\s+than\s*'+_N,'<','POSITIVE_STRICT_UPPER'),
    (r'\bexactly\s*'+_N,'=','POSITIVE_EQUALITY')]


def _requirements(question):
    candidates=[]
    for priority,(pattern,operator,polarity) in enumerate(_PATTERNS):
        for m in re.finditer(pattern,question,re.I):
            candidates.append({'start':m.start(),'end':m.end(),'text':m[0],
                'value':m[1],'operator':operator,'polarity':polarity,'priority':priority})
    chosen=[]
    for candidate in sorted(candidates,key=lambda x:(x['start'],-(x['end']-x['start']),x['priority'])):
        if not any(candidate['start']<c['end'] and c['start']<candidate['end'] for c in chosen):chosen.append(candidate)
    # Explicit display-only exclusions are outside the count quantifier.
    output_only=[]
    for m in re.finditer(r'(?:do\s+)?not\s+(?:show|display|return)\s+(?:(?:their|the)\s+)?(?:names?|labels?|descriptions?)\b',question,re.I):
        output_only.append({'start':m.start(),'end':m.end(),'text':m[0]})
    masked=list(question)
    for span in chosen+output_only:
        masked[span['start']:span['end']]=[' ']*(span['end']-span['start'])
    remaining=''.join(masked)
    unresolved=[{'start':m.start(),'end':m.end(),'text':m[0]} for m in re.finditer(r'\b(?:no|not|without|except|unless|neither|nor)\b',remaining,re.I)]
    return chosen,output_only,unresolved


def analyze(question,actual_schema,sql_ast):
    result=v2.analyze(question,actual_schema,sql_ast)
    if not isinstance(question,str) or len(question)>8192 or not isinstance(sql_ast,exp.Select):return result
    requirements,output_only,unresolved=_requirements(question)
    if not requirements:return result
    tables,error=v2.v1._schema(actual_schema)
    count=[]
    if not error or actual_schema is None:
        if len(requirements)==1:
            r=requirements[0]
            source={'question_span':{k:r[k] for k in ('start','end','text')},
                'quantifier_polarity':r['polarity'],'normalized_operator':r['operator'],
                'unresolved_negation_spans':unresolved,'unrelated_output_negation_spans':output_only,
                'normalization_scope':'Local COUNT requirement only; no original question/model/tool input replaced.'}
            assumptions=[v2._assumption('QUANTIFIER_POLARITY','Full longest quantifier span has the stated polarity.',
                'PARSED_LOCAL_POLARITY' if r['operator'] is not None and not unresolved else 'UNRESOLVED'),
                v2._assumption('COUNT_UNIT_ALIGNMENT','This COUNT measures the requested objects.'),
                v2._assumption('POPULATION_ALIGNMENT','Negation/exception and population outside the parsed span preserve this local COUNT interpretation.')]
            reason=(r['polarity'] if r['operator'] is None else 'NEGATION_SCOPE_UNRESOLVED' if unresolved else
                    'NONCOUNT_REQUEST_NOT_INTEGER_NORMALIZED' if re.search(r'\b(?:sum|summed|average|mean)\b',question,re.I) else None)
            if reason:
                count=[v2._signal('COUNT_BOUND','REQUESTED_OBJECT_COUNT','conditional_unknown',reason,source,assumptions)]
            else:
                phrases={'>=':'at least','>':'more than','<=':'at most','<':'fewer than','=':'exactly'}
                noun=re.match(r'\s+([A-Za-z_]+)',question[r['end']:])
                normalized=phrases[r['operator']]+' '+r['value']+' '+(noun[1] if noun else 'objects')
                count=v2._bounds(normalized,sql_ast.copy(),tables)
                for s in count:
                    s['source'].update(source)
                    s['necessary_assumptions']=[assumptions[0]]+s['necessary_assumptions']
        else:
            count=[v2._signal('COUNT_BOUND','REQUESTED_OBJECT_COUNT','conditional_unknown',
                'MULTIPLE_QUANTIFIER_SCOPES_UNRESOLVED',{'quantifier_spans':requirements},
                [v2._assumption('QUANTIFIER_POLARITY','Multiple scopes are not collapsed into one interval.')])]
    result['conditional_signals']=[s for s in result['conditional_signals'] if s['family']!='COUNT_BOUND']+count
    result['conditional_summary']={family:{state:sum(s['family']==family and s['status']==state for s in result['conditional_signals'])
        for state in ('conditional_match','conditional_conflict','conditional_unknown')} for family in ('COUNT_BOUND','QUANTITY')}
    # V1 did not certify these NL polarities. Do not inherit a false hard/support
    # verdict produced by its positive substring parser.
    if unresolved or any(not r['polarity'].startswith('POSITIVE_') for r in requirements):
        result['global']={'status':'UNKNOWN','hard_conflict':False,'full_question_correctness_certified':False,
                          'reason':'negated_quantifier_requires_separate_global_scope_proof'}
        result['global_source']='V2B_no_new_global_certificate_for_negated_quantifier'
    result['version']='PURE-V2B'
    result['COUNT_quantifier_polarity']={'spans':requirements,'unresolved':unresolved,'output_only':output_only}
    return result
