"""Finite schema field slots with scope-local ownership and leaf-only edits."""
import re
from sqlglot import exp
from repair_rules import parse
from step107_frontier_search import sources


def words(s):
    s=re.sub(r'([a-z])([A-Z])',r'\1 \2',s)
    return {x.rstrip('s') for x in re.findall(r'[a-z]+',s.lower()) if x not in ('the','of','a','an','are','is','what','which','return','show','give','r')}


def family(kind):
    k=kind.lower()
    if any(x in k for x in ('char','text','clob')):return 'text'
    if any(x in k for x in ('int','real','float','double','numeric','number','decimal')):return 'number'
    if any(x in k for x in ('date','time')):return 'time'
    return 'unknown'


def output_focus(question):
    return re.split(r'\b(?:who|whose|where|which has|that have|that are|with|related to|from|in the)\b',question,1,flags=re.I)[0]


def role_score(column,question):
    return len(words(column)&words(output_focus(question)))


def build(sql,question,schema):
    tree=parse(sql)
    if tree is None:return [],['invalid_sql']
    slots=[];reject=[];known={t['name'].lower():t for t in schema}
    for scope_index,scope in enumerate(tree.find_all(exp.Select)):
        src=sources(scope,schema)
        local_tables=[]
        root=scope.args.get('from')
        if root and isinstance(root.this,exp.Table):local_tables.append(root.this)
        local_tables += [j.this for j in scope.args.get('joins',[]) if isinstance(j.this,exp.Table)]
        if any(t.name.lower() not in known for t in local_tables):reject.append('unknown_relation_in_scope');continue
        if not src or any(isinstance(j.this,exp.Subquery) for j in scope.args.get('joins',[])):reject.append('derived_scope');continue
        for expression_index,expression in enumerate(scope.selects):
            col=expression.unalias()
            if not isinstance(col,exp.Column):continue
            owners=[a for a,t in src.items() if (not col.table or a.lower()==col.table.lower()) and any(f['name'].lower()==col.name.lower() for f in t['columns'])]
            if not col.table and len(owners)>1:reject.append('ambiguous_bare_column_owner');continue
            owner=owners[0] if len(owners)==1 else None
            old=next((f for f in src[owner]['columns'] if f['name'].lower()==col.name.lower()),None) if owner else None
            expected=family(old['type']) if old else ('text' if words(question)&{'name','topic','category','title','code'} else 'unknown')
            pool=[]
            for alias,table in src.items():
                for field in table['columns']:
                    if owner==alias and col.name.lower()==field['name'].lower():continue
                    if expected!='unknown' and family(field['type'])!=expected:continue
                    if family(field['type'])=='unknown':continue
                    score=role_score(field['name'],question)
                    pool.append({'table':table['name'],'alias':alias,'column':field['name'],'type':field['type'],'pk':field.get('pk',0),
                                 'fk':[f for f in table.get('foreign_keys',[]) if f['from'].lower()==field['name'].lower()],
                                 'score':score,'role_evidence':sorted(words(field['name'])&words(output_focus(question)))})
            pool.sort(key=lambda c:(-c['score'],c['table'].lower(),c['column'].lower()))
            pool=pool[:4]
            if not pool:continue
            slots.append({'kind':'projection_field','scope_index':scope_index,'expression_index':expression_index,
                          'current':col.sql(dialect='sqlite'),'current_owner':owner,'current_column':col.name,
                          'current_score':role_score(col.name,question) if owner else -1,'expected_type':expected,
                          'question':question,'local_clause':expression.sql(dialect='sqlite'),'options':pool,
                          'owner_status':'declared_unique' if owner else 'unknown_column'})
    return slots[:3],reject


def symbolic(slot):
    scores=[c['score'] for c in slot['options']]
    if not scores:return 0,'no_options'
    top=max(scores)
    if top<=0 or top<=slot['current_score']:return 0,'current_not_contradicted'
    winners=[i+1 for i,s in enumerate(scores) if s==top]
    return (winners[0],'unique_output_role_margin') if len(winners)==1 else (0,'same_type_role_tie')


def compile_edit(sql,slot,choice):
    if not isinstance(choice,int) or choice<1 or choice>len(slot['options']):return None,'abstain_or_invalid_choice'
    tree=parse(sql);scopes=list(tree.find_all(exp.Select))
    if slot['scope_index']>=len(scopes):return None,'scope_address_mismatch'
    scope=scopes[slot['scope_index']];e=scope.selects[slot['expression_index']]
    if e.unalias().sql(dialect='sqlite')!=slot['current']:return None,'column_address_mismatch'
    option=slot['options'][choice-1];column=exp.column(option['column'],table=option['alias'])
    # Preserve aliases, sibling expressions and every WHERE/GROUP/ORDER/LIMIT node.
    if isinstance(e,exp.Alias):e.set('this',column)
    else:
        expressions=scope.selects;expressions[slot['expression_index']]=column;scope.set('expressions',expressions)
    return tree.sql(dialect='sqlite'),'one_projection_leaf_only'


def semantic_column(sql,schema):
    """Qualified/bare equality only when each scope proves one local owner."""
    tree=parse(sql)
    if tree is None:return None
    for scope in tree.find_all(exp.Select):
        src=sources(scope,schema)
        for c in scope.find_all(exp.Column):
            if c.find_ancestor(exp.Select) is not scope or c.table:continue
            owners=[a for a,t in src.items() if any(f['name'].lower()==c.name.lower() for f in t['columns'])]
            if len(owners)!=1:return None
            c.set('table',exp.to_identifier(owners[0]))
    return tree.sql(dialect='sqlite').lower()
