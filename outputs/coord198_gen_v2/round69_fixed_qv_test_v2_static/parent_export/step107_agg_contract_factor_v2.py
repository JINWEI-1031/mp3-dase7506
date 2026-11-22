"""PURE-V2: conditional evidence is separate from V1 global certificates.

No selector, tools, model, history, labels or identity inputs. Plural quantity
and row-entity interpretations remain alternatives, never a forced SUM rule.
"""
import re
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
import step107_agg_contract_factor_v1 as v1


def _tokens(text):
    irregular = {'children':'child','people':'person','men':'man','women':'woman'}
    return {irregular.get(w,w) for w in v1._words(text)}


def _assumption(name,detail,status='UNPROVEN'):
    return {'name':name,'detail':detail,'status':status}


def _signal(family,hypothesis,status,reason,source,assumptions,facts=None):
    return {'family':family,'hypothesis':hypothesis,'status':status,'reason':reason,
            'source':source,'necessary_assumptions':assumptions,'facts':facts or {},
            'hard_certificate':False,'requested_action':'NONE'}


def _scopes(tree):
    scopes=list(traverse_scope(tree) or [])
    root=next((s for s in scopes if s.expression is tree),None)
    return scopes,root


def _bounds(question,tree,tables):
    patterns=[(r'at\s+least\s*(-?\d+(?:\.\d+)?)','>='),
        (r'(-?\d+(?:\.\d+)?)\s*or\s*more','>='),(r'more\s+than\s*(-?\d+(?:\.\d+)?)','>'),
        (r'at\s+most\s*(-?\d+(?:\.\d+)?)','<='),(r'(-?\d+(?:\.\d+)?)\s*or\s*fewer','<='),
        (r'fewer\s+than\s*(-?\d+(?:\.\d+)?)','<'),(r'exactly\s*(-?\d+(?:\.\d+)?)','=')]
    required=[(m,op) for pattern,op in patterns for m in re.finditer(pattern,question,re.I)]
    if len(required)!=1: return []
    match,op=required[0]; expected=v1._interval(op,match[1])
    source={'question_span':{'start':match.start(),'end':match.end(),'text':match[0]},
            'domain':'Intrinsic nonnegative integer COUNT only'}
    noun=re.match(r'\s+([A-Za-z_]+)',question[match.end():])
    source['requested_count_phrase']=noun[1] if noun else None
    assumptions=[_assumption('COUNT_UNIT_ALIGNMENT','This particular COUNT counts all and only the requested objects.'),
        _assumption('POPULATION_ALIGNMENT','This COUNT scope represents the requested population and quantifier.'),
        _assumption('NULL_AND_MULTIPLICITY_ALIGNMENT','COUNT operand, DISTINCT, NULL extension and fanout preserve that unit.')]
    scopes,root=_scopes(tree); found=[]
    operators={exp.GT:'>',exp.GTE:'>=',exp.LT:'<',exp.LTE:'<=',exp.EQ:'='}
    for index,scope in enumerate(scopes):
        having=scope.expression.args.get('having')
        predicate=having.this if having else None
        while isinstance(predicate,exp.Paren): predicate=predicate.this
        for node in scope.expression.walk():
            if node.find_ancestor(exp.Select) is not scope.expression or type(node) not in operators: continue
            lhs=node.this; value=v1._number(node.expression); operator=operators[type(node)]
            if value is None:
                value=v1._number(node.this);lhs=node.expression
                operator={'>':'<','>=':'<=','<':'>','<=':'>=','=':'='}[operator]
            raw_lhs=lhs;lhs=v1._expand_alias(lhs,scope,tables)
            if value is None or not isinstance(lhs,exp.AggFunc): continue
            details={**source,'scope':index,'SQL_expression':node.sql(dialect='sqlite'),
                'count_expression':lhs.sql(dialect='sqlite'),
                'counted_unit_equivalence_between_parent_and_candidate_proven':False}
            facts={'required_interval':expected,'actual_function':lhs.key.upper(),
                   'actual_scope_sources':sorted(_selected_tables(scope)),
                   'actual_interval':None,'actual_count_operand':lhs.this.sql(dialect='sqlite')}
            reason=None
            if re.search(r'\b(?:sum|summed|average|mean)\b',question,re.I): reason='REQUEST_IS_NOT_PROVEN_COUNT_DOMAIN'
            elif not isinstance(lhs,exp.Count): reason='NONCOUNT_AGGREGATE_NOT_INTEGER_NORMALIZED'
            elif not tables and isinstance(raw_lhs,exp.Column): reason='ALIAS_BINDING_REQUIRES_OBSERVED_SCHEMA'
            elif root is None or not v1._visible(scope,root): reason='COUNT_SCOPE_NOT_ROOT_ALIGNED'
            elif predicate is not node: reason='COUNT_BOUND_BOOLEAN_OR_POSITION_UNRESOLVED'
            actual=v1._interval(operator,value) if isinstance(lhs,exp.Count) else None
            facts['actual_interval']=actual
            if actual is None or expected is None: reason=reason or 'COUNT_LITERAL_DOMAIN_UNSUPPORTED'
            status='conditional_unknown' if reason else 'conditional_match' if actual==expected else 'conditional_conflict'
            found.append(_signal('COUNT_BOUND','REQUESTED_OBJECT_COUNT',status,
                reason or ('COUNT_INTEGER_INTERVAL_EQUIVALENT' if actual==expected else 'COUNT_INTEGER_INTERVAL_DIFFERENT'),
                details,assumptions,facts))
    if not found:
        found.append(_signal('COUNT_BOUND','REQUESTED_OBJECT_COUNT','conditional_unknown',
            'NO_COMPARABLE_COUNT_BOUND_IN_ALIGNED_SCOPE',source,assumptions))
    return found


def _selected_tables(scope):
    result=set()
    for source in v1._selected(scope).values():
        if isinstance(source,exp.Table): result.add(source.name.casefold())
        elif hasattr(source,'expression'):
            base=v1._plain_source(source)
            if base: result.add(base)
    return result


def _keys(table):
    return [name for name,col in sorted(table['columns'].items(),key=lambda p:p[1].get('pk',0)) if col.get('pk')]


def _request(question):
    match=re.search(r'\b(most|least|fewest)\s+(?:number\s+of\s+)?([A-Za-z_]+)',question,re.I)
    direction=None; target=None; mode=None
    if match:
        raw=match[2].lower()
        if not (raw in {'children','people','men','women'} or raw.endswith('s') and not raw.endswith('ss')): return None
        direction='DESC' if match[1].lower()=='most' else 'ASC';mode='plural_extreme'
        target_match=re.search(r'\b(?:which|what)\s+([A-Za-z_]+)',question,re.I)
        if target_match and target_match[1].lower() not in {'is','are','the'}:
            target_words=_tokens(target_match[1])-{'id','key'}
            if len(target_words)==1: target=next(iter(target_words))
    else:
        match=re.search(r'\b(?:how\s+many|number\s+of)\s+([A-Za-z_]+)',question,re.I)
        if not match: return None
        raw=match[1].lower();mode='quantity_count';target=None
    if len(_tokens(raw))!=1: return None
    noun=next(iter(_tokens(raw)))
    return {'unit':noun,'raw_unit_phrase':raw,'mode':mode,'direction':direction,'target':target,
        'question_span':{'start':match.start(),'end':match.end(),'text':match[0]}}


def _targets(request,owner,tables):
    target=request['target']
    if target is None: return []
    values=[]
    for name,table in tables.items():
        if target in _tokens(table['name']):
            keys=_keys(table)
            if keys: values.append(tuple((name,k) for k in keys))
        if name==owner:
            for col,meta in table['columns'].items():
                if _tokens(meta['name'])-{'id','key'}=={target}: values.append(((name,col),))
    return list(dict.fromkeys(values))


def _metric(request,tree,tables):
    scopes,root=_scopes(tree)
    if root is None: return None,'ROOT_SCOPE_UNRESOLVED'
    if request['mode']=='plural_extreme':
        order=root.expression.args.get('order')
        if not order or not order.expressions: return None,'QUANTITY_ORDER_METRIC_UNRESOLVED'
        ordered=order.expressions[0];node=v1._expand_alias(ordered.this,root,tables)
        direction='DESC' if ordered.args.get('desc') else 'ASC'
    else:
        aggregates=[a for a in root.expression.find_all(exp.AggFunc) if a.find_ancestor(exp.Select) is root.expression]
        if len(aggregates)!=1: return None,'QUANTITY_COUNT_METRIC_UNRESOLVED'
        node=aggregates[0];direction=None
    lineage=v1._resolve(node.this if isinstance(node,exp.AggFunc) else node,root,tables)
    return {'scope':root,'scope_id':scopes.index(root),'node':node,'direction':direction,
            'lineage':lineage,'groups':v1._groups(root,tables),'sources':_selected_tables(root)},None


def _quantity(question,tree,tables):
    request=_request(question)
    if not request: return [],[]
    noun=request['unit'];quantities=[];rows=[]
    for name,table in tables.items():
        fks=table.get('fk_columns',set())
        for col,meta in table['columns'].items():
            words=_tokens(meta['name'])
            if meta.get('pk') or col in fks or words&{'id','key','code','identifier','index'}: continue
            if not re.search(r'INT|REAL|NUM|DEC|FLOAT|DOUBLE',meta.get('type',''),re.I): continue
            if noun in words and words-{noun}<={'count','number','of','quantity','total'}:
                quantities.append((name,col))
        if noun in _tokens(table['name']): rows.append(name)
    records_request=noun in {'row','record'}
    hypotheses=[];signals=[];metric,error=_metric(request,tree,tables)
    if quantities:
        hypotheses.append({'id':'QUANTITY_COLUMN','kind':'stored_quantity','bindings':[list(x) for x in quantities],
            'binding_unique':len(quantities)==1,'unit':noun,
            'assumption':'Named nonkey numeric column represents requested quantity; additivity is conditional, never inferred from INTEGER.'})
    if rows or records_request:
        hypotheses.append({'id':'ROW_ENTITY','kind':'row_records' if records_request else 'entity_rows',
            'owners':rows,'unit':noun,'binding_unique':len(rows)==1 or records_request,
            'assumption':'Each counted row/identity represents the requested entity; quantity fact rows need separate evidence.'})
    if not hypotheses:
        return [],[_signal('QUANTITY','UNRESOLVED_UNIT','conditional_unknown','NO_NONKEY_QUANTITY_OR_ENTITY_BINDING',
            request,[_assumption('UNIT_BINDING','Numeric identifiers/codes do not establish quantity.')])]
    for h in hypotheses:
        assumptions=[_assumption('TARGET_POPULATION','Current filters/relations preserve the population requested.'),
            _assumption('NULL_POLICY','NULL handling agrees with the requested quantity/entity interpretation.'),
            _assumption('MULTIPLICITY','No unaccounted fanout or duplicate fact contribution.'),
            _assumption('TIE_POLICY','Winner/output tie policy agrees with the request.')]
        facts={'request':request,'hypothesis':h,'scalar_selector_score':None}
        status='conditional_unknown';reason=error or 'UNIT_OR_SCOPE_UNRESOLVED'
        source={'question_span':request['question_span'],'requested_unit':noun,'mechanism':'plural_quantity_or_row_entity'}
        if metric:
            scope=metric['scope'];node=metric['node'];function=node.key.upper() if isinstance(node,exp.AggFunc) else 'COLUMN' if isinstance(node,exp.Column) else 'TRANSFORM'
            facts.update(actual_function=function,actual_metric=node.sql(dialect='sqlite'),
                actual_lineage=list(metric['lineage']) if metric['lineage'] else None,
                actual_group=[list(g) for g in metric['groups']] if metric['groups'] is not None else None)
            source['SQL_scope']=metric['scope_id']
            if h['id']=='QUANTITY_COLUMN' and len(quantities)==1:
                owner,col=quantities[0];table=tables[owner];keys=_keys(table);targets=_targets(request,owner,tables)
                native_key=tuple((owner,k) for k in keys)
                same_target=any(t==native_key for t in targets) and bool(keys)
                spans_target=any(set(t)<set(native_key) for t in targets) and bool(keys)
                groups=tuple(metric['groups']) if metric['groups'] is not None else None
                group_matches=any(set(t)==set(groups or []) for t in targets)
                facts.update(quantity_binding=[owner,col],declared_native_key=list(keys),
                    native_key_nonnull=bool(keys) and all(table['columns'][k].get('not_null') for k in keys),
                    target_bindings=[[list(x) for x in t] for t in targets],declared_target_key_equals_native_key=same_target,
                    declared_fact_key_has_extra_target_dimensions=spans_target)
                assumptions += [_assumption('COLUMN_UNIT_BINDING','This named column stores the requested unit, not an undocumented code.'),
                    _assumption('DECLARED_KEY_NONNULL_IDENTITY','Declared native key provides the intended nonnull fact identity.',
                        'OBSERVED_NONNULL_KEY' if keys and all(table['columns'][k].get('not_null') for k in keys) else 'UNPROVEN'),
                    _assumption('ADDITIVITY','Quantity cells across fact rows may be summed for the requested target.'),
                    _assumption('NONOVERLAPPING_FACTS','Fact records contribute nonoverlapping requested quantities.')]
                if function=='COUNT':
                    status='conditional_conflict';reason='COUNT_ROWS_DOES_NOT_CONSUME_STORED_QUANTITY'
                    assumptions.append(_assumption('ROW_IS_NOT_ONE_UNIT','Fact row is not known to be exactly one requested quantity unit.'))
                elif metric['lineage']==(owner,col) and function in {'SUM','COLUMN'}:
                    if request['mode']=='quantity_count' and function=='SUM' and not metric['groups']:
                        status='conditional_match';reason='SUM_NAMED_QUANTITY_UNDER_ADDITIVE_INTERPRETATION'
                    elif request['mode']=='plural_extreme' and targets:
                        if function=='SUM' and group_matches:
                            status='conditional_match';reason='SUM_QUANTITY_AT_REQUESTED_TARGET_GRAIN'
                        elif function=='COLUMN' and same_target and v1._plain_source(scope)==owner:
                            status='conditional_match';reason='DIRECT_COLUMN_AT_SINGLE_ROW_TARGET_GRAIN'
                        elif function=='COLUMN' and spans_target:
                            status='conditional_conflict';reason='DIRECT_FACT_VALUE_DOES_NOT_COMBINE_TARGET_FACT_ROWS'
                        else: reason='TARGET_AGGREGATION_GRAIN_NOT_PROVEN'
                    else: reason='REQUESTED_TARGET_GRAIN_UNRESOLVED'
                elif function=='TRANSFORM' or metric['lineage'] is None:
                    reason='QUANTITY_EXPRESSION_LINEAGE_UNRESOLVED'
                else: reason='OTHER_QUANTITY_REPRESENTATION_NOT_PROVEN'
            elif h['id']=='ROW_ENTITY':
                owner=rows[0] if len(rows)==1 else v1._plain_source(scope) if records_request else None
                table=tables.get(owner);keys=_keys(table) if table else []
                own_quantity=any(o==owner for o,_ in quantities)
                key_names=set().union(*(_tokens(k) for k in keys)) if keys else set()
                entity_evidence=bool(records_request or table and keys and (noun in key_names or not own_quantity))
                facts.update(row_owner=owner,declared_identity=keys,row_entity_evidence=entity_evidence,
                             same_owner_has_quantity_fact_column=own_quantity)
                if function=='COUNT' and owner and (owner in metric['sources'] or v1._plain_source(scope)==owner):
                    operand=node.this;distinct=isinstance(operand,exp.Distinct)
                    if distinct:
                        operand=operand.expressions[0] if len(operand.expressions)==1 else operand
                    bound=v1._resolve(operand,scope,tables) if isinstance(operand,exp.Column) else None
                    left_or_full=any(j.args.get('side') in ('LEFT','FULL','RIGHT') for j in scope.expression.args.get('joins',[]))
                    targets=_targets(request,owner,tables);groups=tuple(metric['groups'] or [])
                    target_ok=request['mode']=='quantity_count' and not groups or any(set(t)==set(groups) for t in targets)
                    if not entity_evidence: reason='FACT_ROWS_NOT_PROVEN_REQUESTED_ENTITIES'
                    elif not target_ok: reason='ROW_ENTITY_TARGET_SCOPE_UNRESOLVED'
                    elif isinstance(operand,exp.Star) and left_or_full: reason='NULL_EXTENSION_ROW_COUNT_UNRESOLVED'
                    elif isinstance(operand,exp.Star): status='conditional_match';reason='COUNT_ROWS_UNDER_ENTITY_ROW_INTERPRETATION'
                    elif bound and bound[0]==owner:
                        column=table['columns'][bound[1]]
                        completed={bound}|set(metric['groups'] or [])
                        identity={(owner,k) for k in keys}
                        if distinct and not identity<=completed: reason='DISTINCT_COMPONENT_DOES_NOT_PROVE_IDENTITY'
                        elif not column.get('not_null') and bound[1] not in keys: reason='NULLABLE_NONIDENTITY_COUNT_UNRESOLVED'
                        else: status='conditional_match';reason='COUNT_BOUND_IDENTITY_UNDER_ROW_INTERPRETATION'
                    else: reason='ROW_ENTITY_COUNT_OPERAND_UNRESOLVED'
                else: reason='ROW_ENTITY_REPRESENTATION_UNRESOLVED'
            if status=='conditional_match' and request['direction'] and metric['direction']!=request['direction']:
                status='conditional_conflict';reason='QUANTITY_OR_ENTITY_EXTREME_DIRECTION_DIFFERENT'
        signals.append(_signal('QUANTITY',h['id'],status,reason,source,assumptions,facts))
    return hypotheses,signals


def analyze(question,actual_schema,sql_ast):
    """Pure conditional report; global certificates remain exactly frozen V1."""
    base=v1.analyze(question,actual_schema,sql_ast)
    signals=[];hypotheses=[]
    tables,error=v1._schema(actual_schema)
    if not error:
        for table in actual_schema:
            tables[table['name'].casefold()]['fk_columns']={fk['from'].casefold() for fk in table.get('foreign_keys',[]) if 'from' in fk}
    if isinstance(question,str) and len(question)<=8192 and isinstance(sql_ast,exp.Select) and len(list(sql_ast.walk()))<=2000:
        tree=sql_ast.copy()
        if not error or actual_schema is None:
            signals.extend(_bounds(question,tree,tables))
        if not error:
            hypotheses,quantity=_quantity(question,tree,tables);signals.extend(quantity)
        for h in base['hypothesis_results']:
            for slot in h['slots']:
                if slot['slot']!='measure_function_owner_unit': continue
                relation=slot.get('conditional_relation_without_missing_proofs',slot['status'])
                state={'SUPPORTED':'conditional_match','EXPLICIT_CONFLICT':'conditional_conflict'}.get(relation,'conditional_unknown')
                signals.append(_signal('QUANTITY','EXPLICIT_'+h['hypothesis'],state,'EXPLICIT_MEASURE_CONDITIONAL_RELATION',
                    {'SQL_scope':slot.get('scope'),'source':'frozen_V1_explicit_measure_slot'},
                    [_assumption(r,r) for r in slot.get('missing_proof_reasons',[])],slot))
    summary={family:{state:sum(s['family']==family and s['status']==state for s in signals)
                    for state in ('conditional_match','conditional_conflict','conditional_unknown')}
             for family in ('COUNT_BOUND','QUANTITY')}
    return {'version':'PURE-V2','global':base['global'],'global_source':'unchanged_V1_certificate',
        'unknown_explanations':base['unknown_explanations'],'base_hypothesis_results':base['hypothesis_results'],
        'quantity_hypotheses':hypotheses,'conditional_signals':signals,'conditional_summary':summary,
        'conditional_hypotheses_are_alternatives_not_additive_votes':True,
        'selector_score':None,'requested_action':'NONE','tool_requests':[]}
