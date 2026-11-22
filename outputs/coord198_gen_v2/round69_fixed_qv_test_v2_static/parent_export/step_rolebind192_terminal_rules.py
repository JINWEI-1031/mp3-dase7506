"""Current-plan-only proposals with type and directed-role constraints."""
from sqlglot import exp
from repair_rules import parse
from step_semantic192_rules import typed_literals
from step_rolebind192_rules import role_words,ROLE_GROUPS
from step_rolebind192_evidence import generate as evidence_generate
from step124_pool_search import sources

def generate(seeds,obs,schema,*,new=True,cap=24):
    out=[];q=obs['task']['question']
    for item in evidence_generate(seeds[:1],obs,schema,new=new,cap=cap):
        if not typed_literals(item['sql'],schema):continue
        tree=parse(item['sql'])
        if item['family']=='negative':
            root=tree.args.get('from');src=sources(tree,schema)
            if not root:continue
            parent=src.get(root.this.alias_or_name)
            if parent is None:continue
            ambiguous=False
            for exists in tree.find_all(exp.Exists):
                for child in sources(exists.this,schema).values():
                    links=[f for f in child['foreign_keys'] if f['table'].lower()==parent['name'].lower()]
                    if len(links)>1:
                        explicit=[f for f in links if any(role_words(f['from'])&g and role_words(q)&g for g in ROLE_GROUPS)]
                        if len(explicit)!=1:ambiguous=True
            if ambiguous:continue
        out.append(item)
    return out
