"""Explicit counted-noun conflicts are hard; other entity uncertainty remains soft."""
import re
import step_synth441_sketch as raw
from step_scope441_ir import table,key
from step124_pool_search import sources
from step_semantic192_rules import tokens
from repair_rules import parse
def extra_check(c,obs,schema):
    if c['family']!='count_membership':return True,None
    m=re.search(r'\b(?:how many|number of)\s+(?:(?:distinct|different)\s+)?([a-z_]+)',obs['task']['question'],re.I)
    if not m:return True,None
    noun=tokens(m[1]);root=table(schema,c['slots']['root']);pk=tokens(key(root) or '')-raw.STOP
    if not pk or pk&noun:return True,None
    src=sources(parse(c['parent']),schema)
    if any((tokens(key(tb) or '')-raw.STOP)&noun and tokens(tb['name'])&noun for tb in src.values()):return False,'explicit_counted_entity_changed_to_other_key_role'
    return True,None
def static_check(c,obs,schema):
    ok,why=raw.static_check(c,obs,schema)
    return extra_check(c,obs,schema) if ok else (ok,why)
def generate(current,obs,schema,family='combined'):
    pool,why=raw.generate(current,obs,schema,family)
    for c in pool:
        if c['static_admit']:
            ok,r=extra_check(c,obs,schema)
            if not ok:c['static_admit']=False;c['static_reason']=r
    return pool,why
