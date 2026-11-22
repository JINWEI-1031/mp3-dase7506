"""Independent clause items with explicit semantic pre/postconditions."""
import re
from sqlglot import exp
from repair_rules import parse,sql_text
from step124_pool_search import sources
from step_graph439_edges import qi,family

def request(current,obs,schema,support=1):
    tree=parse(current);q=obs['task']['question']
    if not support:return None,'no_out_of_fold_item_support'
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return None,'nested_scope'
    if re.search(r'\b(?:both|neither|either)\b',q,re.I):return None,'compound_polarity'
    if re.search(r'\b(?:never|previously|before|after|since|ever)\b|\bno longer\b',q,re.I):return None,'historical_event_scope_not_current_state'
    if re.search(r'\b(?:have|has|had|was|were)\s+(?:not\s+)?logged\b',q,re.I):return None,'historical_event_scope_not_current_state'
    states=list(re.finditer(r'\blogged\s+(in|out)\b',q,re.I))
    if len(states)!=1:return None,'no_single_explicit_login_state'
    state=states[0]
    if re.search(r'\bnamed\b|\bcalled\b|\busername\b|\buser name\b|\blogin name\b|\blogged\s+in\s+as\b',q,re.I):return None,'identity_constraint_not_state_only'
    src=sources(tree,schema);where=tree.args.get('where')
    if not where or where.find(exp.Or):return None,'unsupported_predicate_scope'
    matches=[]
    for eq in where.find_all(exp.EQ):
        if not isinstance(eq.this,exp.Column) or not isinstance(eq.expression,exp.Literal) or not eq.expression.is_string:continue
        col=eq.this
        if not re.search(r'(?:^|_)login$',col.name,re.I):continue
        if re.sub(r'\W|_','',eq.expression.this).lower() not in ('loggedin','loggedout','true','false'):continue
        if re.search(r"['\"]"+re.escape(eq.expression.this)+r"['\"]",q,re.I):continue
        owners=[(a,t) for a,t in src.items() if (not col.table or a.lower()==col.table.lower()) and any(c['name'].lower()==col.name.lower() for c in t['columns'])]
        if len(owners)!=1:continue
        alias,table=owners[0]
        if any(c.get('pk') for c in table['columns'] if c['name'].lower()==col.name.lower()):continue
        if not any(re.search(r'user_?name|member_?name',c['name'],re.I) for c in table['columns']):continue
        matches.append({'alias':alias,'table':table['name'],'field':col.name,'literal':eq.expression.this})
    if len(matches)!=1:return None,'state_field_role_ambiguous'
    b=matches[0];prefix=q[:state.start()];negated=bool(re.search(r'(?:\bnot|\b(?:are|is)n[\u2019\']t)\s+(?:currently\s+)?$',prefix,re.I))
    if re.search(r'\b(?:not|without|no|none)\b|n[\u2019\']t',prefix,re.I) and not negated:return None,'state_negation_scope_ambiguous'
    positive=state[1].lower()=='in';positive=not positive if negated else positive;value='1' if positive else '0'
    sql=f'SELECT CAST({qi(b["field"])} AS TEXT) AS value, COUNT(*) AS frequency FROM {qi(b["table"])} GROUP BY {qi(b["field"])} LIMIT 9'
    return {'sql':sql,'parent':current,'binding':b,'value':value,'source':'graph439_item_probe','item':'binary_state_literal','support':support},None

def apply(plan,feedback):
    if not feedback.get('ok') or feedback.get('truncated'):return [],'incomplete_state_domain'
    values={str(r[0]) for r in feedback.get('rows',[])}
    if values!={'0','1'} or sum(r[1] for r in feedback['rows'])<3:return [],'not_repeated_complete_observed_binary_domain'
    tree=parse(plan['parent']);b=plan['binding'];changed=0
    for eq in tree.args['where'].find_all(exp.EQ):
        if isinstance(eq.this,exp.Column) and eq.this.name.lower()==b['field'].lower() and isinstance(eq.expression,exp.Literal) and eq.expression.this==b['literal']:
            eq.set('expression',exp.Literal.string(plan['value']));changed+=1
    if changed!=1:return [],'item_precondition_changed'
    return [{'sql':sql_text(tree),'source':'graph439_item','path':['binary_state_literal'],'confidence':12,'evidence':{'item':'binary_state_literal','binding':b,'target':plan['value'],'observed_domain':sorted(values),'observed_frequencies':{str(r[0]):r[1] for r in feedback['rows']},'contradiction':'explicit login-state pseudoliteral contradicts complete observed binary state domain; separate username column anchors identity elsewhere','preserved':'same field, table, every other predicate, projection, grouping, ordering and scope','limitation':'public domain evidence does not constrain hidden-instance values','out_of_fold_support':plan['support']}}],None

def generate(current,obs,schema,support):
    """Independent non-probing items; no arbitrary clause replacement."""
    tree=parse(current);q=obs['task']['question'];out=[];refusals=[]
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return [],['nested_scope']
    if re.search(r'\b(?:both|neither|either)\b',q,re.I):return [],['question_compound_scope_unsupported']
    if support.get('scalar_global_count',0):
        if tree.args.get('group') and not tree.args.get('having') and not tree.args.get('limit') and len(tree.expressions)==1 and isinstance(tree.expressions[0].unalias(),exp.Count) and re.search(r'\bhow many\b|\bnumber of\b',q,re.I) and not re.search(r'\beach\b|\bevery\b|\bper\b|\bgroup\b|\bby\b',q,re.I):
            t=tree.copy();t.set('group',None);out.append({'sql':sql_text(t),'source':'graph439_item','path':['scalar_global_count'],'confidence':10,'evidence':{'contradiction':'global quantity request and single count output cannot be answered as several partition counts','preserved':'original COUNT object/DISTINCT, joins, row filters and ordering; no HAVING or limit is permitted','support':support['scalar_global_count']}})
        else:refusals.append('scalar_count_precondition_not_met')
    if support.get('explicit_comparison_bound',0):
        cues=[('at least',exp.GTE),('more than',exp.GT),('at most',exp.LTE),('less than',exp.LT)]
        for cue,cls in cues:
            m=re.search(r'\b'+cue+r'\s+(\d+(?:\.\d+)?)\b',q,re.I)
            if not m:continue
            matches=[n for n in tree.find_all((exp.GT,exp.GTE,exp.LT,exp.LTE)) if isinstance(n.expression,exp.Literal) and not n.expression.is_string and n.expression.this==m[1] and not isinstance(n,cls)]
            if len(matches)!=1:refusals.append('comparison_object_ambiguous');continue
            n=matches[0]
            if n.find_ancestor((exp.Or,exp.Not)):refusals.append('comparison_boolean_scope_unsupported');continue
            if isinstance(n.this,exp.Column):
                src=sources(tree,schema);owners=[tb for a,tb in src.items() if (not n.this.table or n.this.table.lower()==a.lower()) and any(c['name'].lower()==n.this.name.lower() and family(c['type'])=='number' for c in tb['columns'])]
                if len(owners)!=1:refusals.append('comparison_type_or_owner_unproven');continue
            elif not isinstance(n.this,(exp.Count,exp.Sum,exp.Avg,exp.Min,exp.Max)):refusals.append('comparison_expression_unsupported');continue
            t=tree.copy();ns=[v for v in t.find_all(type(n)) if str(v)==str(n)]
            if len(ns)!=1:continue
            ns[0].replace(cls(this=ns[0].this.copy(),expression=ns[0].expression.copy()));out.append({'sql':sql_text(t),'source':'graph439_item','path':['explicit_comparison_bound'],'confidence':10,'evidence':{'contradiction':cue+' explicitly specifies comparison boundary','preserved':'same comparison object and numeric literal; all other clauses unchanged','support':support['explicit_comparison_bound']}})
    return out,refusals
