"""One charged value/schema probe; no hidden values, labels or retrieval IDs."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_semantic192_rules import tokens

def canon(s):return re.sub(r'[^a-z0-9]','',str(s).lower())
def qi(s):return '"'+s.replace('"','""')+'"'
def qs(s):return sql_text(exp.Literal.string(s))
def variants(value,column):
    values={value.lower()};key=canon(value);field=column.lower()
    if re.search(r'gender|sex',field) and key in ('male','female'):values.add(key[0])
    if 'degree' in field:
        if key in ('masters','master'):values.update(['master','masters','ms','msc','m.s.','ma','m.a.'])
        if key in ('phd','doctorate'):values.update(['phd','ph.d','ph.d.','doctorate'])
    if 'month' in field:
        months='january february march april may june july august september october november december'.split()
        if key in months:values.update([str(months.index(key)+1),f'{months.index(key)+1:02}'])
    return sorted(values)

def request(current,obs,schema,lexical=False):
    tree=parse(current)
    if not isinstance(tree,exp.Select):return None,'seed_not_single_select'
    src=sources(tree,schema);q=obs['task']['question'];bindings=[];parts=[]
    where=tree.args.get('where')
    if where is None:return None,'no_literal_predicate'
    for node in where.find_all((exp.EQ,exp.In)):
        if not isinstance(node.this,exp.Column):continue
        col=node.this;owners=[(a,t) for a,t in src.items() if (not col.table or col.table.lower()==a.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
        if len(owners)!=1:continue
        alias,table=owners[0];literals=node.expressions if isinstance(node,exp.In) and not node.args.get('query') else [node.expression] if isinstance(node,exp.EQ) else []
        for lit in literals:
            if not isinstance(lit,exp.Literal) or not lit.is_string or len(lit.this)<2 or canon(lit.this) not in canon(q):continue
            # Entity names require identity evidence, not a category rebind.
            if re.search(r'first|last|fname|lname|password|login|user.?name',col.name,re.I):continue
            text=[c['name'] for c in table['columns'] if re.search(r'CHAR|TEXT|CLOB',c['type'],re.I)]
            fields=[n for n in text if n.lower()==col.name.lower() or not lexical and not re.search(r'name|address|email|password|description|detail',n,re.I)]
            if len(fields)>6:return None,'too_many_unresolved_category_fields'
            for field in fields:
                idx=len(bindings);aliases=variants(lit.this,field) if not lexical else [lit.this.lower()]
                condition='LOWER('+qi(field)+') IN ('+','.join(qs(v) for v in aliases)+')'
                if not lexical:
                    key=canon(lit.this)
                    if key.endswith('ation'):condition+=' OR LOWER('+qi(field)+') LIKE '+qs(key[:-3]+'%')
                    elif key.endswith('ed'):condition+=' OR LOWER('+qi(field)+') LIKE '+qs(key[:-2]+'%')
                parts.append(f'SELECT DISTINCT {idx} AS binding, CAST({qi(field)} AS TEXT) AS value FROM {qi(table["name"])} WHERE '+condition)
                bindings.append({'alias':alias,'table':table['name'],'from_field':col.name,'to_field':field,'literal':lit.this,'aliases':aliases})
    if not parts:return None,'no_category_value_binding'
    return {'sql':' UNION ALL '.join(parts)+' LIMIT 9','bindings':bindings,'parent':current,'source':'ground438_value_probe','lexical':lexical},None

def apply(plan,feedback):
    if not feedback.get('ok') or feedback.get('truncated'):return [],'failed_or_truncated_value_probe'
    hits={}
    for idx,value in feedback['rows']:
        if not isinstance(idx,int) or not 0<=idx<len(plan['bindings']):return [],'invalid_probe_binding'
        b=plan['bindings'][idx];key=(b['alias'].lower(),b['from_field'].lower(),b['literal']);hits.setdefault(key,[]).append({**b,'value':value})
    tree=parse(plan['parent']);changes=[]
    for key,options in hits.items():
        unique={(o['to_field'].lower(),o['value']) for o in options}
        if len(unique)!=1:return [],'ambiguous_value_or_attribute'
        b=options[0]
        if b['to_field'].lower()==b['from_field'].lower() and b['value']==b['literal']:continue
        # A cross-field move needs an exact question literal, not an abbreviation.
        if b['to_field'].lower()!=b['from_field'].lower() and canon(b['value'])!=canon(b['literal']):return [],'cross_field_alias_has_insufficient_role_evidence'
        for lit in tree.find_all(exp.Literal):
            if not lit.is_string or lit.this!=b['literal']:continue
            n=lit.parent
            while n is not None and not isinstance(n,(exp.EQ,exp.In)):n=n.parent
            if n is None or not isinstance(n.this,exp.Column):continue
            if n.this.name.lower()!=b['from_field'].lower() or n.this.table and n.this.table.lower()!=b['alias'].lower():continue
            n.this.set('this',exp.to_identifier(b['to_field']));lit.set('this',b['value'])
        changes.append(b)
    sql=sql_text(tree)
    if not changes or sql==sql_text(parse(plan['parent'])):return [],'no_changed_binding'
    return [{'sql':sql,'source':'ground438_values','path':['observed_category_binding'],'evidence':{'changes':changes,'preserved':'all other AST clauses and original predicate operators','probe':'one public execute; bounded9 rows; reject truncation/ambiguity'},'confidence':12}],None
