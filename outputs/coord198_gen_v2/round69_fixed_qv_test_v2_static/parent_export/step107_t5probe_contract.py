"""Question/schema evidence for an independent plan, separate from AST retention."""
import re
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from repair_rules import parse
from step107_slots_fields import words,family,output_focus
from step4_agent import continuation_conflicts
from step107_feedback_review import feedback,proof

def inspect(sql,obs):
    tree=parse(sql);q=obs['task']['question'];ql=q.lower();hard=[];soft=[];evidence=[]
    result={'hard_conflicts':hard,'uncertainties':soft,'evidence':evidence}
    if tree is None:hard.append('not_one_read_only_query');return result
    if len(tree.selects)!=len(obs['example']['columns']):hard.append('explicit_output_width')
    schema=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
    if schema is None:soft.append('schema_not_observed');return result
    tables={t['name'].lower():t for t in schema};bindings={};dqs=[]
    scopes=list(traverse_scope(tree) or [])
    for sc in scopes:
        local={}
        for alias,source in sc.sources.items():
            if isinstance(source,exp.Table):
                meta=tables.get(source.name.lower())
                if meta is None:hard.append('unknown_relation:'+source.name)
                else:local[alias.lower()]=meta
            else:soft.append('derived_relation_context')
        for c in sc.columns:
            owners=[(a,t,col) for a,t in local.items() if not c.table or c.table.lower()==a for col in t['columns'] if col['name'].lower()==c.name.lower()]
            outer=sc.parent
            while not owners and c.table and outer is not None:
                for a,source in outer.sources.items():
                    if a.lower()==c.table.lower() and isinstance(source,exp.Table) and source.name.lower() in tables:
                        t=tables[source.name.lower()];owners=[(a,t,col) for col in t['columns'] if col['name'].lower()==c.name.lower()]
                outer=outer.parent
            if len(owners)==1:bindings[id(c)]=owners[0]
            elif len(owners)>1:hard.append('ambiguous_column:'+c.sql())
            elif not c.table and c.this.args.get('quoted') and len(c.name)>1 and c.name.lower() in ql:
                # SQLite DQS fallback, admitted only as a question-anchored
                # comparison value, never as an invented output field.
                if isinstance(c.parent,exp.EQ) and c is c.parent.expression:
                    dqs.append(c.name.lower());soft.append('question_anchored_DQS_value')
                else:hard.append('unbound_quoted_output_or_operand:'+c.name)
            elif not c.table and any(c.name.lower()==e.alias.lower() for e in sc.expression.selects if e.alias):
                soft.append('select_alias_reference')
            elif any(not isinstance(s,exp.Table) for s in sc.sources.values()):soft.append('derived_column_binding')
            else:hard.append('unbound_column:'+c.sql())
    literals=[x.this.strip('%').lower() for x in tree.find_all(exp.Literal) if x.is_string]+dqs
    original=parse(obs['issue_sql']);required=[]
    if original is not None:
        for x in original.find_all(exp.Literal):
            if not x.is_string:continue
            value=x.this.strip('%').lower()
            if len(value)<=1 or value not in ql:continue
            quoted=any(mark+value+mark in ql for mark in ('\"',"'"))
            grammatical=re.search(r'\b(?:the|a)\s+'+re.escape(value)+r'\s+(?:number|amount|quantity|value)\b',ql) or re.search(r'\b(?:can|may|must|should|to)\s+'+re.escape(value)+r'\s+(?:with|by|for|in|at)\b',ql)
            if grammatical and not quoted:soft.append('source_literal_not_independently_anchored')
            else:required.append(value)
    for value in required:
        if not any(set(words(value))<=set(words(v)) for v in literals):hard.append('missing_explicit_constant:'+value)
    if required:evidence.append('question_anchored_named_constants')
    hard.extend('question_intent:'+x for x in (continuation_conflicts(sql,q) or ()))
    # Explicit output roles can contradict a same-type, different attribute.
    focus=words(output_focus(q));expected_roles=focus&{'name','topic','category','title','code','date','time','year','id'}
    mentions=list(re.finditer(r'\b(names?|topics?|categor(?:y|ies)|titles?|codes?|dates?|times?|years?|ids?|numbers?)\b',output_focus(q),re.I))
    role_order=[words(m[0])&{'name','topic','category','title','code','date','time','year','id','number'} for m in mentions]
    if mentions:expected_roles=role_order[0]
    if not expected_roles:
        lexical={w for t in schema for c in t['columns'] for w in words(c['name']) if w in focus and w not in set().union(*(words(t['name']) for t in schema))}
        if len(lexical)==1:expected_roles=lexical
    expected_entities={n for n,t in tables.items() if words(n)&focus}
    projections=list(tree.selects)
    if isinstance(tree,exp.SetOperation):projections=[e for s in tree.find_all(exp.Select) for e in s.selects]
    for position,expression in enumerate(projections):
        roles=expected_roles
        if len(tree.selects)>1:
            if len(role_order)==len(tree.selects):roles=role_order[position%len(tree.selects)]
            else:roles=set();soft.append('multi_output_role_mapping_unresolved')
        cols=list(expression.find_all(exp.Column));bound=[bindings[id(c)] for c in cols if id(c) in bindings]
        if isinstance(expression.unalias(),exp.Column) and bound:
            _,table,column=bound[0];cw=words(column['name']);tw=words(table['name'])
            target=roles-{'id'} if 'name' in roles else roles
            owner_has_named_field=any('name' in words(c['name']) for c in table['columns'])
            direct_entity_name=family(column['type'])=='text' and bool(cw&focus) and not owner_has_named_field and 'name' in target
            grounded=any(words(c['name'])&target for t in schema for c in t['columns'])
            if target and not cw&target and not direct_entity_name:
                if grounded:hard.append('explicit_output_attribute:'+column['name'])
                else:soft.append('requested_output_role_not_lexically_unique')
            if 'first' in focus and 'name' in focus and not {'first','name'}<=cw:hard.append('explicit_first_name_role')
            if 'last' in focus and 'name' in focus and not {'last','name'}<=cw:hard.append('explicit_last_name_role')
            if expected_entities and table['name'].lower() not in expected_entities and any(words(tables[e]['name'])&focus for e in expected_entities):
                # Table ownership is hard only when the requested field is
                # present on an explicitly requested table, not every bridge.
                requested=[t for n,t in tables.items() if n in expected_entities and any(words(c['name'])&cw for c in t['columns'])]
                if requested:hard.append('explicit_output_entity:'+table['name'])
            if bound:evidence.append('schema_bound_output:'+table['name']+'.'+column['name'])
    # Join evidence: bridge tables may be necessary without appearing in q.
    for j in tree.find_all(exp.Join):
        on=j.args.get('on');supported=False
        for eq in on.find_all(exp.EQ) if on is not None else []:
            a,b=bindings.get(id(eq.this)),bindings.get(id(eq.expression))
            if not a or not b:continue
            _,ta,ca=a;_,tb,cb=b
            edge=any(k['from'].lower()==ca['name'].lower() and k['table'].lower()==tb['name'].lower() and k['to'].lower()==cb['name'].lower() for k in ta['foreign_keys']) or any(k['from'].lower()==cb['name'].lower() and k['table'].lower()==ta['name'].lower() and k['to'].lower()==ca['name'].lower() for k in tb['foreign_keys'])
            classroom=words(ca['name'])==words(cb['name'])=={'classroom'} and bool(re.search(r'\bteach(?:ers?|ing)?\b',ql))
            if edge or classroom:supported=True;evidence.append('declared_FK_path' if edge else 'teaching_classroom_relation')
            else:soft.append('unproved_join_relation')
            if family(ca['type'])!=family(cb['type']) and 'unknown' not in (family(ca['type']),family(cb['type'])):
                if edge:soft.append('declared_FK_affinity_mismatch')
                else:hard.append('incompatible_join_types')
        if not supported:soft.append('join_needs_relationship_evidence')
    if re.search(r'\bboth\b',ql):
        if isinstance(tree,exp.Intersect):evidence.append('two_relation_intersection')
        elif tree.find(exp.And) or tree.find(exp.Having) or len(list(tree.find_all(exp.Exists)))>=2:soft.append('both_scope_requires_interpretation')
        else:hard.append('explicit_both_not_represented')
        for sc in tree.find_all(exp.Select):
            where=sc.args.get('where')
            if where and not where.find(exp.Or):
                values={}
                for eq in where.find_all(exp.EQ):
                    if isinstance(eq.this,exp.Column) and isinstance(eq.expression,exp.Literal):values.setdefault(eq.this.sql().lower(),set()).add(eq.expression.this)
                if any(len(v)>1 for v in values.values()):hard.append('same_row_conflicting_both_values')
    if re.search(r'\b(?:without|never|neither|do(?:es)? not have)\b',ql) and not (tree.find(exp.Not) or isinstance(tree,exp.Except)):
        hard.append('explicit_negative_scope_absent')
    # A named numeric predicate must constrain that attribute, not another
    # equally typed field. Candidate literal presence by itself is insufficient.
    for table in schema:
        for column in table['columns']:
            cw=words(column['name'])
            if len(cw)!=1:continue
            word=next(iter(cw))
            match=re.search(r'\b'+re.escape(word)+r'\s+(\d+)\b',ql)
            if not match:continue
            predicates=[eq for eq in tree.walk() if isinstance(eq,(exp.EQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.NEQ)) and isinstance(eq.this,exp.Column) and words(eq.this.name)==cw and isinstance(eq.expression,exp.Literal) and eq.expression.this==match[1]]
            if not predicates:hard.append('explicit_predicate_attribute:'+word)
    count_match=re.search(r'\b(?:how many|number of)\s+([a-z_]+)',ql)
    target_tables={n for n in tables if count_match and words(n)&words(count_match[1])}
    if target_tables:
        for count in tree.find_all(exp.Count):
            sc=count.find_ancestor(exp.Select)
            if sc is None:continue
            count_cols=[bindings[id(c)] for c in count.find_all(exp.Column) if id(c) in bindings]
            if count_cols and any(b[1]['name'].lower() not in target_tables for b in count_cols):hard.append('explicit_counted_entity')
            if isinstance(count.this,exp.Distinct):continue
            joins=list(sc.args.get('joins',[]))
            for join in joins:
                on=join.args.get('on');others=[]
                for c in on.find_all(exp.Column) if on is not None else []:
                    b=bindings.get(id(c))
                    if b and b[1]['name'].lower() not in target_tables:others.append(b)
                for _,other,_ in others:
                    keys={c['name'].lower() for c in other['columns'] if c.get('pk')}
                    bound={b[2]['name'].lower() for b in others if b[1] is other}
                    if not keys or not keys<=bound:hard.append('counted_entity_multiplied_by_association')
    if 'respectively' in ql:
        order=[m[0].lower() for m in re.finditer(r'\b(?:maximum|minimum|highest|lowest)\b',ql)]
        actual=[('maximum' if e.find(exp.Max) else 'minimum' if e.find(exp.Min) else 'unknown') for e in tree.selects]
        desired=[{'highest':'maximum','lowest':'minimum'}.get(x,x) for x in order]
        if desired and len(desired)==len(actual) and desired!=actual:hard.append('explicit_respectively_output_mapping')
        elif not desired:soft.append('respectively_mapping_unresolved')
    if re.search(r'\bnull(?: values?)? (?:last|first)\b',ql):
        want_first=bool(re.search(r'\bnull(?: values?)? first\b',ql));ordering=tree.args.get('order')
        if not ordering or any(bool(x.args.get('nulls_first'))!=want_first for x in ordering.expressions):hard.append('explicit_NULL_placement')
    if tree.find(exp.Sum):
        measured=[c for s in tree.find_all(exp.Sum) for c in s.find_all(exp.Column)]
        if any(words(c.name)&words(q) for c in measured):evidence.append('question_named_additive_measure')
        soft.append('additive_interpretation_not_declared_in_schema')
    if tree.args.get('limit') and re.search(r'\bmost|highest|lowest|greatest|fewest\b',ql):soft.append('unspecified_extreme_ties')
    if not expected_roles:soft.append('output_role_not_lexically_unique')
    result.update(hard_conflicts=sorted(set(hard)),uncertainties=sorted(set(soft)),evidence=sorted(set(evidence)))
    return result

def approve(parent,candidate,obs,mode='soft',uncertainty_policy='schema_supported'):
    if mode=='local':
        ok,why=proof(parent,candidate,obs);return ok,{'mode':mode,'reason':why}
    fb=feedback(candidate,obs)
    new=inspect(candidate,obs);old=inspect(parent,obs)
    record={'mode':mode,'candidate':new,'parent':old,'policy':uncertainty_policy}
    if not fb or not fb.get('matches_example'):return False,{**record,'reason':'not_observed_public_pass'}
    if new['hard_conflicts']:return False,{**record,'reason':'explicit_semantic_conflict'}
    oldfb=feedback(parent,obs)
    if oldfb and oldfb.get('matches_example') and not old['hard_conflicts']:
        return False,{**record,'reason':'passing_uncontradicted_parent_retained'}
    if mode=='strict' and new['uncertainties']:return False,{**record,'reason':'strict_uncertainty_abstention'}
    if uncertainty_policy=='abstain' and new['uncertainties']:return False,{**record,'reason':'calibrated_uncertainty_abstention'}
    if uncertainty_policy=='schema_supported' and any(x in new['uncertainties'] for x in ('unproved_join_relation','join_needs_relationship_evidence','derived_column_binding','schema_not_observed')):
        return False,{**record,'reason':'relationship_or_binding_not_supported'}
    return True,{**record,'reason':'independent_plan_evidence_with_recorded_uncertainty'}
