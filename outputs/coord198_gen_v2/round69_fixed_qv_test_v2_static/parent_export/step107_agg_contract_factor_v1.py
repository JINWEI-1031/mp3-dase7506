"""Independent pure AST research factor. Never imported by a running Agent.

Only COUNT bounds and explicit SUM/AVG quantity ownership are investigated.
SUPPORTED describes these bounded obligations, never whole-query correctness.
Inputs contain no histories, task identities, gold labels, tools or callbacks.
"""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import re
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

SUPPORTED = 'SUPPORTED'
CONFLICT = 'EXPLICIT_CONFLICT'
UNKNOWN = 'UNKNOWN'


def _words(value):
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', value)
    result = []
    for word in re.findall(r'[a-z]+', value.lower()):
        if word.endswith('ies'): word = word[:-3]+'y'
        elif len(word)>3 and word.endswith('s') and not word.endswith('ss'): word = word[:-1]
        result.append(word)
    return set(result)


def _unknown(layer, reason, detail, slot='contract', scope=None):
    return {'layer':layer,'reason':reason,'detail':detail,'slot':slot,'scope':scope,
            'requested_action':'NONE'}


def _interval(operator, value):
    """Intrinsic COUNT domain only. Never called for SUM/AVG or column types."""
    value = Decimal(str(value))
    if not value.is_finite() or abs(value)>=2**63: return None
    floor = int(value.to_integral_value(rounding=ROUND_FLOOR))
    ceil = int(value.to_integral_value(rounding=ROUND_CEILING))
    lower, upper = 0, None
    if operator == '>': lower = max(0,floor+1)
    elif operator == '>=': lower = max(0,ceil)
    elif operator == '<': upper = ceil-1
    elif operator == '<=': upper = floor
    elif operator == '=':
        if value != floor: return {'empty':True,'lower':0,'upper':-1}
        lower = max(0,floor); upper = floor
    else: return None
    if upper is not None and upper<lower: return {'empty':True,'lower':0,'upper':-1}
    return {'empty':False,'lower':lower,'upper':upper}


def _schema(value):
    if value is None: return {}, 'SCHEMA_NOT_OBSERVED'
    if not isinstance(value,list): return {}, 'SCHEMA_FORMAT_UNSUPPORTED'
    tables = {}
    for table in value:
        if not isinstance(table,dict) or set(table)-{'name','columns','foreign_keys'}:
            return {}, 'UNOBSERVED_SCHEMA_ANNOTATIONS_REJECTED'
        if not isinstance(table.get('name'),str) or not isinstance(table.get('columns'),list):
            return {}, 'SCHEMA_FORMAT_UNSUPPORTED'
        name = table['name'].casefold()
        if name in tables: return {}, 'SCHEMA_NAME_AMBIGUOUS'
        columns = {}
        if not isinstance(table.get('foreign_keys',[]),list) or any(
                not isinstance(fk,dict) or set(fk)-{'table','from','to'}
                for fk in table.get('foreign_keys',[])):
            return {}, 'UNOBSERVED_SCHEMA_ANNOTATIONS_REJECTED'
        for col in table['columns']:
            if not isinstance(col,dict) or set(col)-{'name','type','not_null','pk'}:
                return {}, 'UNOBSERVED_SCHEMA_ANNOTATIONS_REJECTED'
            if not isinstance(col.get('name'),str) or col['name'].casefold() in columns:
                return {}, 'SCHEMA_NAME_AMBIGUOUS'
            if not isinstance(col.get('type',''),str) or ('not_null' in col and type(col['not_null']) is not bool):
                return {}, 'SCHEMA_FORMAT_UNSUPPORTED'
            columns[col['name'].casefold()] = dict(col)
        tables[name] = {'name':table['name'],'columns':columns}
    return tables, None


def _selected(scope):
    return {name.casefold(): source for name,(_,source) in scope.selected_sources.items()}


def _plain_source(scope, seen=None):
    """One base-row stream, or transparent column-only derived projection."""
    seen = set() if seen is None else seen
    if id(scope) in seen: return None
    seen = seen|{id(scope)}; sources = _selected(scope)
    if len(sources)!=1 or scope.expression.args.get('joins'): return None
    source = next(iter(sources.values()))
    if isinstance(source,exp.Table): return source.name.casefold()
    if not isinstance(source,Scope) or not isinstance(source.expression,exp.Select): return None
    query = source.expression
    if any(query.args.get(k) for k in ('where','group','having','distinct','limit','qualify')): return None
    if any(not isinstance(e.unalias(),(exp.Column,exp.Star)) for e in query.expressions): return None
    return _plain_source(source,seen)


def _resolve(column, scope, tables, seen=None):
    if not isinstance(column,exp.Column): return None
    seen = set() if seen is None else seen
    if (id(scope),column.sql()) in seen: return None
    seen = seen|{(id(scope),column.sql())}; owners = []
    for alias,source in _selected(scope).items():
        if column.table and column.table.casefold()!=alias: continue
        if isinstance(source,exp.Table):
            table = tables.get(source.name.casefold())
            if table and column.name.casefold() in table['columns']:
                owners.append((source.name.casefold(),column.name.casefold()))
        elif isinstance(source,Scope):
            for projection in source.expression.selects:
                if projection.alias_or_name.casefold()==column.name.casefold():
                    owner = _resolve(projection.unalias(),source,tables,seen)
                    if owner: owners.append(owner)
    return owners[0] if len(owners)==1 else None


def _groups(scope,tables):
    group = scope.expression.args.get('group')
    if not group: return []
    values = [_resolve(e,scope,tables) for e in group.expressions]
    return sorted(values) if all(values) else None


def _required_groups(question, table):
    words = _words(question); result = []
    for name,col in table['columns'].items():
        cw = _words(col['name'])-{'id','key'}
        explicit_by = re.search(r'\b(?:by|per)\s+(?:each\s+)?'+re.escape(' '.join(sorted(cw)))+r'\b',question,re.I) if cw else None
        conventional = cw and cw <= words and bool(cw&{'year','month','day','department','group'})
        if explicit_by or conventional: result.append(name)
    return sorted(result)


def _number(node):
    if isinstance(node,exp.Literal) and not node.is_string:
        if len(node.this)>80: return None
        try:
            # SQLite real numeric literals use IEEE754, not arbitrary precision.
            # No database is run: preserve rounding before integer normalization.
            value = (Decimal.from_float(float(node.this)) if any(c in node.this.lower() for c in '.e')
                     else Decimal(node.this))
        except (ValueError,ArithmeticError): return None
        return value if value.is_finite() else None
    if isinstance(node,exp.Neg):
        value = _number(node.this)
        return -value if value is not None else None
    return None


def _expand_alias(node,scope,tables):
    while isinstance(node,exp.Paren): node = node.this
    if isinstance(node,exp.Column) and not node.table:
        matching = [e.unalias() for e in scope.expression.selects if e.alias and e.alias.casefold()==node.name.casefold()]
        # SQLite names can prefer physical columns over an identical SELECT alias.
        if matching and _resolve(node,scope,tables) is None and len(matching)==1: return matching[0]
    return node


def _visible(scope,root):
    if scope is root: return True
    parent = scope.parent
    if parent is None: return False
    sources = list(_selected(parent).values())
    if len(sources)!=1 or sources[0] is not scope: return False
    query = parent.expression
    if any(query.args.get(k) for k in ('joins','where','having','group','distinct','limit','qualify')): return False
    if any(not isinstance(e.unalias(),(exp.Column,exp.Star)) for e in query.selects): return False
    return _visible(parent,root)


def _unit(aggregate,scope,tables,entity_count):
    base = _plain_source(scope)
    if not base or base not in tables: return None,'ROW_GRAIN_UNPROVEN'
    if scope.expression.args.get('where'): return None,'POPULATION_FILTER_UNRESOLVED'
    if aggregate.find_ancestor(exp.Window): return None,'WINDOW_GRAIN_UNSUPPORTED'
    if aggregate.find_ancestor(exp.Filter): return None,'AGGREGATE_FILTER_SCOPE_UNSUPPORTED'
    table = tables[base]
    keys = [c for c in table['columns'].values() if c.get('pk')]
    if not keys or any(c.get('not_null') is not True for c in keys):
        return None,'ENTITY_IDENTITY_NULL_OR_GRAIN_UNPROVEN' if entity_count else 'ROW_IDENTITY_GRAIN_UNPROVEN'
    operand = aggregate.this
    if isinstance(aggregate,exp.Count):
        if isinstance(operand,exp.Star): return (base,'rows'),None
        distinct = isinstance(operand,exp.Distinct)
        if distinct:
            if len(operand.expressions)!=1: return None,'COMPOSITE_DISTINCT_UNSUPPORTED'
            operand = operand.expressions[0]
        owner = _resolve(operand,scope,tables)
        if not owner or owner[0]!=base: return None,'COUNT_OWNER_OR_EXPRESSION_UNRESOLVED'
        column = table['columns'][owner[1]]
        if column.get('not_null') is not True: return None,'COUNT_NULL_POLICY_UNPROVEN'
        if distinct:
            keys = [n for n,c in table['columns'].items() if c.get('pk')]
            if keys != [owner[1]]: return None,'FULL_IDENTITY_DISTINCT_UNPROVEN'
        return (base,'rows'),None
    owner = _resolve(operand,scope,tables)
    if not owner: return None,'MEASURE_LINEAGE_UNRESOLVED'
    column = tables[owner[0]]['columns'][owner[1]]
    if column.get('not_null') is not True: return None,'MEASURE_NULL_POLICY_UNPROVEN'
    return owner,None


def analyze(question, actual_schema, sql_ast):
    """Return explanations only. Pure/deterministic; no input AST/schema mutation."""
    unknowns = []; hypotheses = []; tables,error = _schema(actual_schema)
    def unknown(layer,reason,detail,slot='contract',scope=None):
        value = _unknown(layer,reason,detail,slot,scope)
        if value not in unknowns: unknowns.append(value)
        return value
    def finish(results,actual,complete):
        states = [r['status'] for r in results]
        if len(states)>1 and SUPPORTED in states and any(s!=SUPPORTED for s in states):
            unknown('COMPARISON_PROOF','ALTERNATIVE_HYPOTHESIS_SURVIVES','One plausible interpretation matches while another does not.')
        globally_blocked = any(x['slot']=='contract' for x in unknowns)
        hard = bool(complete and states and all(s==CONFLICT for s in states) and not globally_blocked)
        status = CONFLICT if hard else SUPPORTED if complete and states and all(s==SUPPORTED for s in states) and not unknowns else UNKNOWN
        if status==UNKNOWN and not unknowns:
            unknown('COMPARISON_PROOF','ALTERNATIVE_HYPOTHESIS_SURVIVES','Not all still-plausible hypotheses establish the same verdict.')
        return {'version':'agg-contract-factor-v1','scope':'bounded_COUNT_threshold_and_explicit_measure_only',
            'required_hypotheses':hypotheses,'hypothesis_enumeration_complete':complete,
            'actual':actual,'hypothesis_results':results,'unknown_explanations':unknowns,
            'global':{'status':status,'hard_conflict':hard,'full_question_correctness_certified':False,
                'reason':'all_finite_hypotheses_explicitly_conflict' if hard else 'bounded_obligations_supported' if status==SUPPORTED else 'unresolved_or_alternative_hypothesis'},
            'requested_action':'NONE','tool_requests':[]}
    if not isinstance(question,str) or not isinstance(sql_ast,exp.Query):
        unknown('ACTUAL_SQL_EVIDENCE','INPUT_FORMAT_UNSUPPORTED','Require question string and an already parsed read-only Query AST.')
        return finish([],[],False)
    if len(question)>8192:
        unknown('REQUIRED_INTENT','QUESTION_COMPLEXITY_UNSUPPORTED','Research interface bounded to8192 question characters.')
        return finish([],[],False)
    if error:
        unknown('ACTUAL_SQL_EVIDENCE',error,'Only actual schema fields name/columns/foreign_keys and name/type/not_null/pk are accepted.')
        return finish([],[],False)
    if len(list(sql_ast.walk()))>2000:
        unknown('ACTUAL_SQL_EVIDENCE','AST_COMPLEXITY_UNSUPPORTED','Research factor bounded to2000 nodes.')
        return finish([],[],False)
    tree = sql_ast.copy(); scopes = list(traverse_scope(tree) or [])
    if not scopes or len(scopes)>12 or not isinstance(tree,exp.Select):
        unknown('ACTUAL_SQL_EVIDENCE','QUERY_SCOPE_UNSUPPORTED','Set operations/large scope graphs are not certified.')
        return finish([],[],False)
    root = next((s for s in scopes if s.expression is tree),None)
    if root is None:
        unknown('ACTUAL_SQL_EVIDENCE','ROOT_SCOPE_UNRESOLVED','No root SELECT scope.')
        return finish([],[],False)
    if re.search(r'\b(?:manager|supervisor|manages?|managing|among|whose|without|except|qualified)\b',question,re.I):
        unknown('REQUIRED_INTENT','POPULATION_PHRASE_UNCONSUMED','Population/manager/quantifier parsing is outside this two-mechanism factor.')
    if re.search(r'\b(?:times|multiplied|divided|ratio|weighted|net)\b|\b(?:before|after)\s+tax\b',question,re.I):
        unknown('REQUIRED_INTENT','MEASURE_COMPOSITION_UNRESOLVED','Compound requested measure is not a single proven column unit.')
    if re.search(r'\b(?:who|where|than|higher|lower|of)\s*[?.]*$',question,re.I):
        unknown('REQUIRED_INTENT','QUESTION_TRUNCATED','Incomplete final phrase cannot be filled in.')
    patterns = [(r'at\s+least\s*(-?\d+(?:\.\d+)?)','>='),
        (r'(-?\d+(?:\.\d+)?)\s*or\s*more','>='),(r'more\s+than\s*(-?\d+(?:\.\d+)?)','>'),
        (r'at\s+most\s*(-?\d+(?:\.\d+)?)','<='),(r'(-?\d+(?:\.\d+)?)\s*or\s*fewer','<='),
        (r'fewer\s+than\s*(-?\d+(?:\.\d+)?)','<'),(r'exactly\s*(-?\d+(?:\.\d+)?)','=')]
    bounds = [(m,op) for pattern,op in patterns for m in re.finditer(pattern,question,re.I)]
    remaining_language = question
    for match,_ in sorted(bounds,key=lambda x:x[0].start(),reverse=True):
        remaining_language = remaining_language[:match.start()]+remaining_language[match.end():]
    if re.search(r'\b(?:or|not|both|neither|either)\b',remaining_language,re.I):
        unknown('REQUIRED_INTENT','BOOLEAN_OR_NEGATION_SCOPE_UNRESOLVED','Additional quantifier/Boolean language not certified.')
    explicit_sum = re.search(r'\bsum(?:med)?\b',question,re.I)
    explicit_avg = re.search(r'\b(?:average|mean)\b',question,re.I)
    if explicit_sum or explicit_avg:
        if re.search(r'\b(?:where|during|before|after|from|in)\b',question,re.I):
            unknown('REQUIRED_INTENT','MEASURE_POPULATION_UNCONSUMED','Additional population restriction is not certified by owner matching.')
    qwords = _words(question); complete = True
    if bounds and not (explicit_sum or explicit_avg):
        if len(bounds)!=1:
            unknown('REQUIRED_INTENT','QUANTIFIER_BOUND_AMBIGUOUS','Exactly one explicit count comparison is supported.')
            return finish([],[],False)
        match,operator = bounds[0]
        noun = re.match(r'\s+([A-Za-z_]+)',question[match.end():])
        if noun and re.search(r'\b(?:who|whose|that|from|in|of|with|and|or|without|except)\b',question[match.end()+noun.end():],re.I):
            unknown('REQUIRED_INTENT','COUNT_POPULATION_TAIL_UNCONSUMED','Counted-entity restriction after threshold is not certified.')
        candidates = [(name,t) for name,t in tables.items() if noun and _words(noun[1])&_words(t['name'])]
        if not candidates:
            unknown('REQUIRED_INTENT','COUNTED_ROLE_UNRESOLVED','Threshold must explicitly name a schema-bound counted entity.')
            return finish([],[],False)
        complete = len(candidates)==1
        if len(candidates)>1:
            unknown('REQUIRED_INTENT','COUNTED_ROLE_AMBIGUOUS','Multiple schema entity roles; lexical candidates do not prove exhaustive intent.')
        for name,table in candidates[:2]:
            hypotheses.append({'id':'H'+str(len(hypotheses)+1),'kind':'COUNT_BOUND','owner':name,
                'groups':_required_groups(question,table),'interval':_interval(operator,match[1]),
                'question_span':{'start':match.start(),'end':match.end(),'text':match[0]}})
    else:
        columns = []
        for name,table in tables.items():
            for col,meta in table['columns'].items():
                words = _words(meta['name'])-{'id','key'}
                numeric = bool(re.search(r'INT|REAL|NUM|DEC|FLOAT|DOUBLE',meta.get('type',''),re.I))
                if words and words<=qwords and numeric and not words&{'year','month','day','group','department'}:
                    columns.append((name,col,table))
        if not columns or not (explicit_sum or explicit_avg or re.search(r'\btotal\b',question,re.I)):
            unknown('REQUIRED_INTENT','QUANTITY_UNIT_UNRESOLVED','No explicit measured column/operator binding; numeric type alone is insufficient.')
            return finish([],[],False)
        # Explicit owner words disambiguate identical measure column names.
        named = [x for x in columns if _words(tables[x[0]]['name'])<=qwords]
        if named: columns = named
        complete = len(columns)==1 and bool(explicit_sum or explicit_avg) and not (explicit_sum and explicit_avg)
        if len(columns)>1:
            unknown('REQUIRED_INTENT','MEASURE_ROLE_AMBIGUOUS','All explored owners can conflict without proving all reasonable intended owners/plans are covered.')
        if not (explicit_sum or explicit_avg):
            unknown('REQUIRED_INTENT','ADDITIVITY_UNPROVEN','Total/numeric column permits more than an assumed SUM interpretation.')
        for owner,col,table in columns[:2]:
            hypotheses.append({'id':'H'+str(len(hypotheses)+1),'kind':'MEASURE','owner':owner,'column':col,
                'function':'AVG' if explicit_avg and not explicit_sum else 'SUM','groups':_required_groups(question,table),
                'operator_span':{'start':(explicit_avg or explicit_sum).start(),'end':(explicit_avg or explicit_sum).end(),'text':(explicit_avg or explicit_sum)[0]} if explicit_avg or explicit_sum else None,
                'integral_measure_domain_proven':False})
        if bounds:
            unknown('COMPARISON_PROOF','NONCOUNT_BOUND_DOMAIN_UNPROVEN','SUM/AVG bounds are never COUNT-normalized, even with INTEGER columns.')
    if not complete:
        unknown('REQUIRED_INTENT','HYPOTHESIS_SET_INCOMPLETE','No all-hypothesis hard verdict from truncation or unspecified additivity.')
    actual = []
    operators = {exp.GT:'>',exp.GTE:'>=',exp.LT:'<',exp.LTE:'<=',exp.EQ:'='}
    for index,scope in enumerate(scopes):
        for node in scope.expression.walk():
            if node.find_ancestor(exp.Select) is not scope.expression: continue
            if type(node) not in operators: continue
            operator = operators[type(node)]; lhs = _expand_alias(node.this,scope,tables); value = _number(node.expression)
            if value is None:
                value = _number(node.this); lhs = _expand_alias(node.expression,scope,tables)
                operator = {'>':'<','>=':'<=','<':'>','<=':'>=','=':'='}[operator]
            if value is None or not isinstance(lhs,exp.AggFunc): continue
            parent = node.parent; boolean_safe = True
            while parent is not None and parent is not scope.expression:
                if isinstance(parent,(exp.Or,exp.Not)): boolean_safe = False
                parent = parent.parent
            predicate_position = node.find_ancestor(exp.Having) is not None
            having = scope.expression.args.get('having')
            predicate = having.this if having else None
            while isinstance(predicate,exp.Paren): predicate = predicate.this
            predicate_position = predicate_position and predicate is node
            actual.append({'scope':index,'kind':'BOUND','aggregate':lhs,'scope_object':scope,
                'operator':operator,'value':str(value),'visible':_visible(scope,root),
                'boolean_safe':boolean_safe and predicate_position,
                'interval':_interval(operator,value) if isinstance(lhs,exp.Count) else None})
        for aggregate in scope.expression.find_all(exp.AggFunc):
            if aggregate.find_ancestor(exp.Select) is scope.expression:
                actual.append({'scope':index,'kind':'MEASURE','aggregate':aggregate,
                               'scope_object':scope,'visible':_visible(scope,root)})
    results = []
    for hypothesis in hypotheses:
        slots = []; hu = []; relevant = [a for a in actual if a['kind']==('BOUND' if hypothesis['kind']=='COUNT_BOUND' else 'MEASURE')]
        # Multiple possible aggregate sites are unresolved, never cherry-picked.
        if len(relevant)!=1:
            hu.append(unknown('ACTUAL_SQL_EVIDENCE','AGGREGATE_SITE_UNRESOLVED','Expected exactly one relevant aggregate site.',hypothesis['id']))
        else:
            a = relevant[0]; scope = a['scope_object']; aggregate = a['aggregate']
            if not a['visible']:
                hu.append(unknown('ACTUAL_SQL_EVIDENCE','UNRELATED_OR_CORRELATED_SCOPE','Aggregate not transparently linked to returned root rows.',hypothesis['id'],a['scope']))
            if hypothesis['kind']=='COUNT_BOUND' and not a.get('boolean_safe'):
                hu.append(unknown('COMPARISON_PROOF','BOUND_BOOLEAN_OR_POSITION_UNPROVEN','This narrow factor requires an isolated HAVING comparison; extra Boolean predicates/output/OR/NOT are unresolved.',hypothesis['id'],a['scope']))
            if hypothesis['kind']=='MEASURE' and tables[hypothesis['owner']]['columns'][hypothesis['column']].get('not_null') is not True:
                hu.append(unknown('ACTUAL_SQL_EVIDENCE','REQUIRED_MEASURE_NULL_POLICY_UNPROVEN','Required quantity NULL policy is not established.',hypothesis['id'],a['scope']))
            if hypothesis['kind']=='MEASURE' and scope.expression.args.get('having'):
                hu.append(unknown('COMPARISON_PROOF','MEASURE_POPULATION_HAVING_UNRESOLVED','Additional group-population filter not certified by measure-owner matching.',hypothesis['id'],a['scope']))
            if any(scope.expression.args.get(k) for k in ('limit','qualify')):
                hu.append(unknown('ACTUAL_SQL_EVIDENCE','LIMIT_OR_WINDOW_POPULATION_UNRESOLVED','Additional population restriction not certified.',hypothesis['id'],a['scope']))
            unit,why = _unit(aggregate,scope,tables,hypothesis['kind']=='COUNT_BOUND')
            if why: hu.append(unknown('ACTUAL_SQL_EVIDENCE',why,'Missing native-row/NULL/owner proof.',hypothesis['id'],a['scope']))
            actual_groups = _groups(scope,tables)
            required_groups = sorted((hypothesis['owner'],g) for g in hypothesis['groups'])
            if actual_groups is None:
                hu.append(unknown('ACTUAL_SQL_EVIDENCE','GROUP_LINEAGE_UNRESOLVED','Grouping expression binding unresolved.',hypothesis['id'],a['scope']))
            elif actual_groups!=required_groups:
                hu.append(unknown('COMPARISON_PROOF','GROUP_GRAIN_NOT_MATCHED','Required grouping and actual grouping differ; no grain equivalence proof.',hypothesis['id'],a['scope']))
            if hypothesis['kind']=='COUNT_BOUND':
                if not isinstance(aggregate,exp.Count):
                    hu.append(unknown('COMPARISON_PROOF','COUNT_DOMAIN_NOT_ESTABLISHED','SUM/AVG are not intrinsically integer COUNT results.',hypothesis['id'],a['scope']))
                elif a['interval'] is None or hypothesis['interval'] is None:
                    hu.append(unknown('COMPARISON_PROOF','COUNT_BOUND_LITERAL_UNSUPPORTED','Nonfinite/overflow-scale bounds are outside the proven numeric domain.',hypothesis['id'],a['scope']))
                elif unit and unit[0]!=hypothesis['owner']:
                    hu.append(unknown('REQUIRED_INTENT','COUNTED_ROLE_NOT_MATCHED','A different owner may represent a different entity grain.',hypothesis['id'],a['scope']))
                else:
                    slots.append({'slot':'count_interval','status':SUPPORTED if a['interval']==hypothesis['interval'] else CONFLICT,
                        'required':hypothesis['interval'],'actual':a['interval'],
                        'proof':'Nonnegative integer domain is intrinsic to COUNT, at this scope only.','scope':a['scope']})
            else:
                required = (hypothesis['owner'],hypothesis['column'])
                function = aggregate.key.upper()
                if unit:
                    same = function==hypothesis['function'] and unit==required
                    slots.append({'slot':'measure_function_owner_unit','status':SUPPORTED if same else CONFLICT,
                        'required':{'function':hypothesis['function'],'lineage':list(required)},
                        'actual':{'function':function,'lineage':list(unit)},'scope':a['scope'],
                        'proof':'Explicit question operator/column binding; resolved base row stream and nonnull operand. No JOIN/FK rewards.'})
        state = UNKNOWN if hu else CONFLICT if any(s['status']==CONFLICT for s in slots) else SUPPORTED if slots else UNKNOWN
        if hu:
            for slot in slots:
                slot['conditional_relation_without_missing_proofs'] = slot['status']
                slot['status'] = UNKNOWN
                slot['missing_proof_reasons'] = [u['reason'] for u in hu]
        results.append({'hypothesis':hypothesis['id'],'status':state,'slots':slots,'unknown_reasons':[u['reason'] for u in hu]})
    public_actual = [{'scope':a['scope'],'kind':a['kind'],'function':a['aggregate'].key.upper(),
        'expression':a['aggregate'].sql(dialect='sqlite'),'visible_root_population':a['visible'],
        'integer_bound':a.get('interval'),'bound_domain':'COUNT_NONNEGATIVE_INTEGER' if isinstance(a['aggregate'],exp.Count) else 'NONCOUNT_UNPROVEN'} for a in actual]
    return finish(results,public_actual,complete)
