"""Conservative, provenance-bearing runtime features; not a semantic oracle.

No scorer, task identity, database identity, reference SQL or DB access here.
Positive lexical/relationship features are soft. Only syntax/output arity is
hard. A partial question analysis never produces a complete certificate.
"""
import re
from dataclasses import asdict
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from repair_rules import parse
from step107_qwen_proactive_protocol import Evidence, UNKNOWN, HARD, certify_simple_count


def tokens(s):
    return {w.rstrip('s') for w in re.findall(r'[a-z]+',s.lower().replace('_',' '))}


def temporal_conflicts(question,tree,schema):
    """Hard only for one unnegated, explicit grounded numeric-year constraint.

    No synonym inference, temporal encoding conversion, subquery or negation
    inversion. Ambiguous column owners/phrasing and missing predicates UNKNOWN.
    """
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return []
    if re.search(r'\b(not|no|neither|without|except|unless)\b',question,re.I):return []
    anchors=list(re.finditer(r'\b(after|before)\s+(\d{4})\b',question,re.I))
    if len(anchors)!=1:return []
    anchor=anchors[0];prefix=question[:anchor.start()]
    matches=[]
    for table in schema:
        for column in table['columns']:
            if not re.search(r'year',column['name'],re.I) or not re.search(r'INT|REAL|NUM|DEC',column.get('type',''),re.I):continue
            pattern=r'\b'+re.escape(column['name']).replace('_',r'[ _]')+r'\b'
            if re.search(pattern,prefix,re.I):matches.append((table,column))
    if len(matches)!=1:return []
    table,column=matches[0]
    sources=[]
    root=tree.args.get('from')
    if root and isinstance(root.this,exp.Table):sources.append(root.this)
    sources.extend(j.this for j in tree.args.get('joins',[]) if isinstance(j.this,exp.Table))
    aliases={t.alias_or_name.lower():t.name.lower() for t in sources}
    where=tree.args.get('where')
    if not where or any(where.find_all(exp.Not)):return []
    predicates=[]
    for predicate in where.find_all(exp.GT,exp.GTE,exp.LT,exp.LTE):
        left,right=predicate.this,predicate.expression;reverse=False
        if isinstance(right,exp.Column) and isinstance(left,exp.Literal):left,right=right,left;reverse=True
        if not isinstance(left,exp.Column) or not isinstance(right,exp.Literal) or right.is_string:continue
        if left.name.lower()!=column['name'].lower() or str(right.this)!=anchor[2]:continue
        owners=[name for alias,name in aliases.items() if (not left.table or alias==left.table.lower()) and
                any(t['name'].lower()==name and any(c['name'].lower()==left.name.lower() for c in t['columns']) for t in schema)]
        if owners!=[table['name'].lower()]:continue
        increasing=isinstance(predicate,(exp.GT,exp.GTE))!=reverse
        predicates.append((predicate,increasing))
    if len(predicates)!=1:return []
    predicate,increasing=predicates[0]
    if increasing==(anchor[1].lower()=='after'):return []
    return [{'kind':'hard','source':'question+schema+current_sql','detail':{
        'conflict':'explicit_year_direction','question_span':anchor[0],'table':table['name'],'column':column['name'],
        'sql_predicate':predicate.sql(),'scope':'single_select_positive_where','year':anchor[2]}}]


def analyze(obs, sql, schema):
    question=obs['task']['question']; q=tokens(question); tree=parse(sql)
    facts=[]; hard=[]; counts=dict(roles=0,values=0,relations=0,aggregates=0,unresolved=1)
    def fact(kind,source,detail):
        facts.append({'kind':kind,'source':source,'detail':detail})
    if tree is None:
        hard.append('unparseable_sql'); fact('hard','current_sql',sql)
    elif not any(isinstance(s.unalias(),exp.Star) or isinstance(s.unalias(),exp.Column) and s.unalias().is_star
                 for s in tree.selects) and len(tree.selects)!=len(obs['example']['columns']):
        hard.append('output_arity'); fact('hard','example.columns',{'expected':len(obs['example']['columns']),'actual':len(tree.selects)})
    if tree is not None:
        temporal=temporal_conflicts(question,tree,schema)
        if temporal:hard.append('explicit_year_direction');facts.extend(temporal)
        known={t['name'].lower():t for t in schema}
        # Traverse lexical SQL scopes. An unqualified ambiguous column is UNKNOWN.
        def resolve(col,scope):
            current=scope
            while current:
                sources={a.lower():v for a,v in current.sources.items() if isinstance(v,exp.Table) and v.name.lower() in known}
                matches=[(a,known[v.name.lower()],c) for a,v in sources.items()
                         for c in known[v.name.lower()]['columns']
                         if c['name'].lower()==col.name.lower() and (not col.table or a==col.table.lower())]
                if len(matches)==1:return matches[0]
                if matches or not col.table:return None
                current=current.parent
            return None
        scopes=list(traverse_scope(tree) or [])
        for scope in scopes:
            select=scope.expression
            if not isinstance(select,exp.Select):continue
            for item in select.selects:
                col=item.unalias()
                if not isinstance(col,exp.Column):continue
                bound=resolve(col,scope)
                if not bound:continue
                _,table,column=bound
                # Output is grounded only when both entity and attribute appear.
                entity=tokens(table['name']); attribute=tokens(column['name'])-entity-{'id'}
                if entity and entity<=q and attribute and attribute<=q:
                    counts['roles']+=1
                    fact('output_role','question+schema+current_sql',{'question':question,'table':table['name'],'column':column['name'],'sql':col.sql()})
            for eq in scope.expression.find_all(exp.EQ):
                # Do not attribute nested equalities to an outer scope.
                if eq.find_ancestor(exp.Select) is not select:continue
                a,b=eq.this,eq.expression
                if isinstance(a,exp.Column) and isinstance(b,exp.Column):
                    left,right=resolve(a,scope),resolve(b,scope)
                    if left and right and left[0]!=right[0]:
                        linked=any(f['from'].lower()==x[2]['name'].lower() and f['table'].lower()==y[1]['name'].lower()
                                   and f['to'].lower()==y[2]['name'].lower()
                                   for x,y in ((left,right),(right,left)) for f in x[1].get('foreign_keys',[]))
                        if linked:
                            counts['relations']+=1
                            fact('schema_fk_edge','schema.foreign_keys+current_sql',eq.sql())
                for column,value in ((a,b),(b,a)):
                    if not isinstance(column,exp.Column) or not isinstance(value,exp.Literal):continue
                    bound=resolve(column,scope)
                    if not bound:continue
                    # Only score an explicit question value at a predicate. Missing
                    # values can be legitimate encodings, so are never hard conflicts.
                    literal=str(value.this)
                    if re.search(r'(?<!\w)'+re.escape(literal)+r'(?!\w)',question,re.I):
                        counts['values']+=1
                        fact('predicate_value','question+schema+current_sql',{'question_span':literal,'predicate':eq.sql(),'table':bound[1]['name']})
            for agg in select.find_all(exp.AggFunc):
                if agg.find_ancestor(exp.Select) is not select:continue
                if isinstance(agg,exp.Count) and re.search(r'\b(how many|number of|count of)\b',question,re.I):
                    target=agg.this
                    if isinstance(target,exp.Column):
                        bound=resolve(target,scope)
                        if bound and tokens(bound[1]['name'])<=q:
                            counts['aggregates']+=1
                            fact('counted_entity_soft','question+schema+current_sql',{'question':question,'table':bound[1]['name'],'count':agg.sql(),'distinct_inferred':False})
            # Explicitly retain unresolved relation scope instead of asserting a
            # missing FK is wrong (cross products and alternate paths may be valid).
            for join in select.args.get('joins',[]):
                if not join.args.get('on') and not join.args.get('using'):
                    counts['unresolved']+=1
                    fact('unresolved_join','current_sql',join.sql())
        check=next((e for e in reversed(obs['history']) if e['tool']=='check' and e.get('sql')==sql),None)
        public=bool(check and check['observation'].get('ok') and check['observation'].get('matches_example'))
        cert=certify_simple_count(question,sql,schema,public)
        if cert.status!='UNKNOWN' and not hard:
            return cert, facts+[{'kind':'certificate','source':'question+schema+actual_history','detail':list(cert.provenance)}]
    return Evidence(HARD if hard else UNKNOWN,tuple(hard),**counts,
                    provenance=tuple(str(f) for f in facts)),facts


def compare(parent,candidate,parent_public,candidate_public,policy):
    if candidate.status==HARD:return False,'hard_candidate_conflict'
    if parent.status=='CERTIFIED':return False,'certified_parent_preserved'
    if not candidate_public:return False,'candidate_has_no_actual_public_pass'
    if candidate.status=='CERTIFIED':return True,'complete_certificate'
    if policy=='keep_unknown_public' and parent_public:return False,'keep_unknown_public'
    margin={'keep_unknown_public':2,'margin1':1,'margin2':2,'margin0_tie_candidate':0}[policy]
    if parent.status==HARD:return True,'hard_parent_replaced_by_checked_legal_candidate'
    eligible=candidate.score-parent.score>=margin and candidate.unresolved<=parent.unresolved
    return eligible,'soft_evidence_margin' if eligible else 'insufficient_soft_margin'


def record(evidence):
    return {**asdict(evidence),'score':evidence.score}
