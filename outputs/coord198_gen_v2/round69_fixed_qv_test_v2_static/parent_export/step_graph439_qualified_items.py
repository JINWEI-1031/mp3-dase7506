"""Production admission: only the verified binary-state item, with typed roles."""
import re
import step_graph439_items as raw
from step_graph439_edges import family

def binding_guard(binding,schema):
    tables=[t for t in schema if t['name'].lower()==binding['table'].lower()]
    if len(tables)!=1:return 'unknown_state_table'
    cols=[c for c in tables[0]['columns'] if c['name'].lower()==binding['field'].lower()]
    if len(cols)!=1:return 'unknown_state_field'
    c=cols[0];name=c['name'];kind=c['type']
    if c.get('pk') or re.search(r'name|(?:^|_)id(?:_|$)|identifier|date|time|year|(?:first|last|recent|previous)_?login',name,re.I):return 'identity_or_time_field_not_state'
    if re.search(r'DATE|TIME|YEAR|INTERVAL',kind,re.I):return 'declared_temporal_type_not_state'
    if family(kind) not in ('text','number') and not re.search(r'BOOL',kind,re.I):return 'unknown_state_storage_affinity'
    return None

def request(current,obs,schema,support=1):
    p,reason=raw.request(current,obs,schema,support)
    if p is None:return None,reason
    reason=binding_guard(p['binding'],schema)
    return (None,reason) if reason else (p,None)

def apply(plan,feedback):return raw.apply(plan,feedback)

def generate(current,obs,schema,support):
    return [],['structural_item_families_failed_independent_boundaries']
