"""PK/FK-proved count-object and grouping repairs, with no DISTINCT blanket."""
import re
from sqlglot import exp
from repair_rules import parse
from step107_frontier_search import sources
from step107_slots_fields import words


def candidates(sql,question,schema):
    tree=parse(sql)
    if not isinstance(tree,exp.Select) or len(list(tree.find_all(exp.Select)))!=1:return [],'nested_scope_not_certified'
    if re.search(r'\b(?:not both|neither|without|not have|no )\b',question,re.I):return [],'negation_requires_quantifier_ir'
    src=sources(tree,schema);root=tree.args.get('from');joins=tree.args.get('joins',[])
    if not root or not isinstance(root.this,exp.Table) or len(joins)!=1:return [],'requires_one_proved_fk_join'
    ra=root.this.alias_or_name;j=joins[0]
    if not isinstance(j.this,exp.Table):return [],'derived_child'
    ca=j.this.alias_or_name
    if ra not in src or ca not in src or src[ra]['name'].lower()==src[ca]['name'].lower():return [],'self_or_unknown_role'
    rp=[c['name'] for c in src[ra]['columns'] if c.get('pk')];cp=[c['name'] for c in src[ca]['columns'] if c.get('pk')]
    fks=[f for f in src[ca].get('foreign_keys',[]) if f['table'].lower()==src[ra]['name'].lower()]
    if len(rp)!=1 or len(cp)!=1 or len(fks)!=1:return [],'ambiguous_key_or_direction'
    keymeta=[next(c for c in src[a]['columns'] if c['name']==key) for a,key in ((ra,rp[0]),(ca,cp[0]))]
    if any(not c.get('not_null') and c['type'].upper()!='INTEGER' for c in keymeta):return [],'nullable_sqlite_primary_key_not_certified'
    fk=fks[0];on=j.args.get('on')
    if not isinstance(on,exp.EQ) or not all(isinstance(x,exp.Column) for x in (on.this,on.expression)):return [],'non_simple_join'
    if {(on.this.table.lower(),on.this.name.lower()),(on.expression.table.lower(),on.expression.name.lower())}!={(ra.lower(),fk['to'].lower()),(ca.lower(),fk['from'].lower())}:return [],'join_not_declared_direction'
    counts=[x.unalias() for x in tree.selects if isinstance(x.unalias(),exp.Count)]
    if len(counts)!=1:return [],'requires_one_count_output'
    count=counts[0];q=question.lower();target=re.search(r'\b(?:how many|number of) ([a-z_]+)',q)
    per_child=bool(target and words(target[1])==words(src[ca]['name'])) and words(src[ra]['name'])<=words(q) and bool(re.search(r'\b(?:each|per|every)\b',q))
    root_target=bool(target and words(target[1])==words(src[ra]['name']))
    t=tree.copy();replacement=None;changed=[]
    if per_child and j.args.get('side')=='LEFT' and tree.args.get('group'):
        if tree.args.get('distinct'):return [],'distinct_projection_entity_scope_unresolved'
        where=tree.args.get('where')
        if where and any(c.table.lower()==ca.lower() for c in where.find_all(exp.Column)):return [],'child_where_would_erase_zero_population'
        columns=[e.unalias() for e in tree.selects if not isinstance(e.unalias(),exp.Count)]
        if any(not isinstance(c,exp.Column) or c.table.lower()!=ra.lower() for c in columns):return [],'nonroot_group_output'
        replacement=exp.Count(this=exp.column(cp[0],table=ca))
        group=t.args['group']
        if not any(isinstance(c,exp.Column) and c.table.lower()==ra.lower() and c.name.lower()==rp[0].lower() for c in group.expressions):
            if any(not isinstance(c,exp.Column) or c.table.lower()!=ra.lower() for c in group.expressions):return [],'unproved_group_scope'
            group.set('expressions',[*group.expressions,exp.column(rp[0],table=ra)]);changed.append('root_pk_group_functional_dependency')
    elif root_target and not tree.args.get('group') and not j.args.get('side') and j.args.get('kind') in (None,'INNER') and len(tree.selects)==1:
        if fk['from'].lower()==cp[0].lower():return [],'child_fk_unique_no_fanout_evidence'
        replacement=exp.Count(this=exp.Distinct(expressions=[exp.column(rp[0],table=ra)]))
    else:return [],'count_entity_or_population_not_explicit'
    if replacement.sql()!=count.sql():
        old=count.sql()
        for c in list(t.find_all(exp.Count)):
            if c.find_ancestor(exp.Select) is t and c.sql()==old:c.replace(replacement.copy())
        changed.append('count_declared_entity_key')
    if not changed:return [],'remaining_repairs_already_complete'
    return [{'sql':t.sql(dialect='sqlite'),'source':'scope_fd_count','changes':changed,
             'evidence':{'root_pk':rp[0],'child_pk':cp[0],'declared_fk':fk,'object':'child_per_root' if per_child else 'root_entity'},
             'preserved':'same joins, local predicates, output attributes, order directions, limits and explicit NULL policy'}],None
