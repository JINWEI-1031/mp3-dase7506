"""Online gate: actual public history/schema only. No case0/DB/Gold/score access."""
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'MP3_SQL_student_package'),str(ROOT/'.venv/Lib/site-packages')]
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from mp3_sql.sqlite_tool import validate_sql
ALLOWED_NAMES=['Select','Union','Intersect','Except','Subquery','From','Join','Table','TableAlias','Identifier','Column','Alias','Literal','Null','Boolean','Star','Where','And','Or','Not','EQ','NEQ','GT','GTE','LT','LTE','Is','In','Between','Like','Paren','Neg','Add','Sub','Mul','Div','Mod','Distinct','Group','Count','Tuple']
ALLOWED={getattr(exp,n) for n in ALLOWED_NAMES}
def classify_receipt(history,sql):
 hits=[(i,e) for i,e in enumerate(history) if e.get('tool')=='check' and e.get('sql')==sql]
 if not hits:return {'status':'UNBOUND_NO_EXACT_CHECK','receipt_indices':[]}
 if len(hits)!=1:return {'status':'UNKNOWN_REPEATED_EXACT_CHECK','receipt_indices':[i for i,e in hits]}
 i,e=hits[0];o=e.get('observation',{});error=o.get('error');base={'receipt_indices':[i],'ok':o.get('ok'),'matches_example':o.get('matches_example'),'error':error}
 if o.get('ok') is True and error is None and type(o.get('matches_example')) is bool:return {**base,'status':'PUBLIC_PASS' if o['matches_example'] else 'PUBLIC_FINGERPRINT_FAIL'}
 if o.get('ok') is False and o.get('matches_example') is False and isinstance(error,dict):
  typ=error.get('type','');msg=str(error.get('message','')).casefold()
  if typ in ['Timeout','ResourceLimit','OutputLimit','MemoryError'] or any(w in msg for w in ['interrupted','out of memory','exceeds 5000','exceeds 2 mb','result cell exceeds']):cat='TECHNICAL_RESOURCE_UNKNOWN'
  elif typ=='InvalidSQL' or typ=='OperationalError' and any(w in msg for w in ['no such table:','no such column:','ambiguous column name:','syntax error']):cat='DETERMINISTIC_PARSE_BINDING_ERROR_NOT_FINGERPRINT_FAIL'
  else:cat='EXECUTION_ERROR_UNKNOWN'
  return {**base,'status':cat}
 return {**base,'status':'UNKNOWN_MALFORMED_OR_INCONSISTENT_FIELDS'}
def stable_subset(sql,history,ordered):
 def no(reason,**kw):return {'status':'UNKNOWN_REJECTED_STABILITY_NOT_PROVED','reason':reason,**kw}
 try:tree=validate_sql(sql)
 except ValueError:return no('invalid_SQL')
 if ordered is not False:return no('ordered_result_requires_order_proof_outside_V1')
 if any(tree.find_all(exp.Order,exp.Limit,exp.Offset)):return no('ordering_limit_offset_not_proved')
 if any(tree.find_all(exp.Window)):return no('window_not_proved')
 if any(tree.find_all(exp.With,exp.CTE)):return no('CTE_recursion_outside_V1')
 funcs=[n for n in tree.walk() if isinstance(n,exp.Func) and not isinstance(n,exp.Count)]
 if funcs:return no('unknown_or_volatile_function_outside_COUNT_only_V1',functions=[n.sql(dialect='sqlite') for n in funcs])
 unknown=sorted({type(n).__name__ for n in tree.walk() if type(n) not in ALLOWED})
 if unknown:return no('AST_operator_not_in_audited_V1',operators=unknown)
 tables=list(tree.find_all(exp.Table));schemas=[e for e in history if e.get('tool')=='schema' and isinstance(e.get('observation'),list)]
 if tables and not schemas:return no('no_actually_paid_schema')
 schema=schemas[0]['observation'] if schemas else [];names={t['name'].casefold() for t in schema}
 if any(t.db or t.catalog or t.name.casefold() not in names for t in tables):return no('external_or_unbound_table')
 try:
  bound=qualify(tree.copy(),dialect='sqlite',schema={t['name']:{c['name']:'UNKNOWN' for c in t['columns']} for t in schema},infer_schema=False,identify=False)
 except Exception as e:return no('column_binding_not_proved',binding_error=type(e).__name__)
 for s in bound.find_all(exp.Select):
  if s.args.get('having') is not None:return no('HAVING_outside_V1')
  group=s.args.get('group');keys={g.sql(dialect='sqlite',normalize=True) for g in group.expressions} if group else set()
  if group and any(not isinstance(g,(exp.Column,exp.Literal)) or isinstance(g,exp.Literal) and not g.is_string for g in group.expressions):return no('group_expression_or_ordinal_not_proved')
  # Count-only aggregate projection: each other expression is a literal or
  # exactly an explicit grouping key; no SQLite bare aggregate columns.
  local_counts=[n for n in s.find_all(exp.Count) if n.find_ancestor(exp.Select) is s]
  if local_counts or group:
   for projection in s.expressions:
    p=projection.unalias()
    if isinstance(p,(exp.Literal,exp.Null,exp.Boolean,exp.Count)):continue
    if p.sql(dialect='sqlite',normalize=True) not in keys:return no('bare_or_unproved_grouped_projection')
 return {'status':'SUPPORTED_STABLE_V1_CONDITIONAL_ON_FIXED_EXECUTOR','reason':'unordered_deterministic_relational_bag_COUNT_only_no_bare_aggregate','SQL_SHA256':hashlib.sha256(sql.encode()).hexdigest(),'paid_schema_used':bool(schemas),'does_not_prove_NL_correctness':True,'fixed_environment_reproducibility_assumption':True}
def eligibility_online(obs,parent_sql,completions):
 receipt=classify_receipt(obs['history'],parent_sql);stability=stable_subset(parent_sql,obs['history'],obs['ordered']);error_proof=static_error_proof(parent_sql,obs['history'],receipt);reason=None;route=None
 if receipt['status']=='PUBLIC_FINGERPRINT_FAIL':
  if stability['status']!='SUPPORTED_STABLE_V1_CONDITIONAL_ON_FIXED_EXECUTOR':reason=stability['reason']
  else:route='stable_successful_fingerprint_FAIL'
 elif receipt['status']=='DETERMINISTIC_PARSE_BINDING_ERROR_NOT_FINGERPRINT_FAIL':
  if not error_proof['proved']:reason=error_proof['reason']
  else:route='explicit_static_SQL_error'
 else:reason=receipt['status']
 if reason is None and (type(completions) is not int or not 0<=completions<=6):reason='UNKNOWN_UNBOUND_COMPLETION_COUNTER'
 if reason is None and completions>=6:reason='completion_cap'
 return {'eligible':reason is None,'route':route,'reason':reason or 'proved_parent_case0_failure_and_one_completion_available','receipt':receipt,'stability':stability,'static_error_proof':error_proof,'remaining_paid_tools':obs['remaining'],'zero_paid_remaining_is_allowed':True,'parent_SQL_SHA256':hashlib.sha256(parent_sql.encode()).hexdigest()}

def static_error_proof(sql,history,receipt):
 """Only source/AST/schema-provable errors; never blanket ok=false eligibility."""
 import re
 def no(reason):return {'proved':False,'reason':reason}
 if receipt['status']!='DETERMINISTIC_PARSE_BINDING_ERROR_NOT_FINGERPRINT_FAIL':return no('not_a_known_parse_binding_error_receipt')
 error=receipt['error'];typ=error['type'];message=error.get('message')
 if not isinstance(message,str):return no('missing_error_message')
 if typ=='InvalidSQL':
  try:validate_sql(sql)
  except ValueError as e:
   if str(e) in ['SQL must be a nonempty string','SQL exceeds the 16000-byte limit','Exactly one SELECT/WITH query is permitted','Only read-only SQL is permitted']:
    return {'proved':True,'reason':'same_official_validator_structurally_rejects_identical_SQL_before_DB_execution','route':'official_validator_structural_rejection','requires_same_validator_source':True}
  return no('parser_recursion_syntax_or_other_failure_not_proved_by_structural_whitelist')
 if typ!='OperationalError':return no('error_type_not_whitelisted')
 try:tree=validate_sql(sql)
 except ValueError:return no('current_official_validation_does_not_bind_this_SQLite_error')
 if not isinstance(tree,exp.Select) or any(tree.find_all(exp.With,exp.CTE,exp.Subquery,exp.Window)):return no('static_binding_error_scope_requires_single_flat_SELECT_no_CTE_window')
 if any(isinstance(n,exp.Func) and not isinstance(n,exp.Count) for n in tree.walk()):return no('unrecognized_or_volatile_function_rejected_even_for_error_route')
 schemas=[e['observation'] for e in history if e.get('tool')=='schema' and isinstance(e.get('observation'),list)]
 if not schemas:return no('no_actually_paid_schema_to_prove_static_binding_error')
 schema={t['name'].casefold():{c['name'].casefold() for c in t['columns']} for t in schemas[0]};tables=list(tree.find_all(exp.Table))
 if any(t.db or t.catalog or not isinstance(t.this,exp.Identifier) for t in tables):return no('external_or_dynamic_table_binding_unknown')
 missing=re.fullmatch(r'no such table:\s*([\w]+)',message,flags=re.I)
 if missing:
  name=missing[1].casefold()
  if name not in schema and any(t.name.casefold()==name for t in tables):return {'proved':True,'reason':'exact_SQLite_no_such_table_names_a_real_AST_table_absent_from_actual_paid_schema','route':'physical_table_absent','missing_table':missing[1],'fails_before_ORDER_LIMIT_aggregate_or_result_evaluation':True,'fixed_same_DB_schema_executor_assumption':True}
  return no('missing_table_error_not_bound_to_absent_real_table_AST')
 col=re.fullmatch(r'no such column:\s*([\w]+)\.([\w]+)',message,flags=re.I)
 if col:
  alias,name=col[1].casefold(),col[2].casefold();targets=[t for t in tables if t.alias_or_name.casefold()==alias]
  if len(targets)==1 and targets[0].name.casefold() in schema and name not in schema[targets[0].name.casefold()] and any(c.table.casefold()==alias and c.name.casefold()==name for c in tree.find_all(exp.Column)):
   return {'proved':True,'reason':'qualified_real_physical_column_absent_from_unique_AST_table_binding_and_paid_schema','route':'qualified_physical_column_absent','fixed_same_DB_schema_executor_assumption':True}
  return no('qualified_missing_column_binding_not_unique_or_not_absent')
 return no('unqualified_column_ambiguity_or_SQLite_syntax_error_not_yet_proved_whitelist')
