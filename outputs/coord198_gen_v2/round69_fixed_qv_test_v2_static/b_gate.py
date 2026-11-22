"""Exact frozen tierB pure extraction; no offline imports."""
import re
from gate_online import exp,validate_sql
SIMPLE=re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')

PREMISES=['The receipt is genuinely emitted by the pinned official check for byte-identical SQL, not synthesized or inferred from its text.','Case0 has the same immutable DB/schema, SQLite/compiler/build settings/authorizer and public binding contract.','Official query workers use fresh read-only connections, no SQL writes/ATTACH/temp setup, no injected custom functions/extensions, and no external or runtime-varying objects/state.','Paid schema is the actual complete protocol snapshot; no offline DB/DDL/gold or score evidence is added.']

def receipt_tier(sql,history):
 def no(reason):return {'proved':False,'route':None,'reason':reason,'tier':'B_CONDITIONAL_REAL_COMPILER_RECEIPT'}
 hits=[e for e in history if e.get('tool')=='check' and e.get('sql')==sql]
 if len(hits)!=1:return no('one_byte_exact_actual_check_required')
 o=hits[0].get('observation',{});err=o.get('error')
 if o.get('ok') is not False or o.get('matches_example') is not False or not isinstance(err,dict) or err.get('type')!='OperationalError' or not isinstance(err.get('message'),str):return no('not_genuine_known_OperationalError_shape')
 message=err['message']
 if any(x in message.lower() for x in ['timeout','interrupt','memory','resource','exceeds','too big','user-defined']):return no('resource_or_custom_runtime_error_excluded')
 schemas=[e['observation'] for e in history if e.get('tool')=='schema' and isinstance(e.get('observation'),list)]
 if not schemas or any(s!=schemas[0] for s in schemas):return no('actual_paid_complete_schema_missing_or_inconsistent')
 names=set()
 for t in schemas[0]:
  if not isinstance(t,dict) or not isinstance(t.get('name'),str) or not SIMPLE.fullmatch(t['name']) or not isinstance(t.get('columns'),list) or t['name'].lower() in names:return no('malformed_schema')
  names.add(t['name'].lower())
  if any(not isinstance(c,dict) or not isinstance(c.get('name'),str) for c in t['columns']):return no('malformed_schema_columns')
  if t.get('kind') in ['virtual','view','dynamic'] or t.get('dynamic') is True:return no('explicit_dynamic_or_virtual_object_not_admitted')
 try:tree=validate_sql(sql)
 except (ValueError,RecursionError):return no('parser_unsupported_never_reclassified_as_proved')
 if not isinstance(tree,exp.Select) or any(tree.find_all(exp.With,exp.CTE,exp.Subquery,exp.Window,exp.SetOperation)):return no('flat_single_SELECT_scope_required')
 allowed={'Select','From','Table','TableAlias','Identifier','Column','Star','Alias','Join','Where','And','Or','Not','EQ','NEQ','GT','GTE','LT','LTE','Is','In','Between','Like','Paren','Neg','Add','Sub','Mul','Div','Mod','Literal','Null','Boolean','Count','Group','Order','Ordered','Limit','Offset','Distinct','Tuple'}
 if any(type(n).__name__ not in allowed for n in tree.walk()):return no('unknown_AST_function_or_dynamic_expression')
 if any(not SIMPLE.fullmatch(n.name) for n in tree.find_all(exp.Identifier)):return no('complex_or_non_ASCII_identifier_not_bound')
 tables=list(tree.find_all(exp.Table));aliases=[]
 for t in tables:
  if not isinstance(t.this,exp.Identifier) or t.db or t.catalog or t.name.lower().startswith('sqlite_'):return no('external_temp_catalog_or_dynamic_table')
  aliases.append(t.alias_or_name.lower())
 if len(set(aliases))!=len(aliases):return no('duplicate_scope_alias')
 def yes(route):return {'proved':True,'route':route,'reason':'genuine_exact_SQLite_compile_binding_error_under_pinned_fixed_execution_contract','tier':'B_CONDITIONAL_REAL_COMPILER_RECEIPT','assumptions':PREMISES,'needs_extra_public_DDL_evidence':False,'does_not_statically_infer_name_absence_or_ambiguity':True,'case0_failure_conditional':True,'candidate_NL_correctness':'NOT_PROVED'}
 m=re.fullmatch(r'no such table:\s*([A-Za-z_][A-Za-z0-9_]*)',message,re.I)
 if m:
  return yes('receipt_missing_table') if any(t.name.lower()==m[1].lower() for t in tables) else no('error_table_not_in_AST_scope')
 m=re.fullmatch(r'(no such column|ambiguous column name):\s*([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?',message,re.I)
 if m:
  first,second=m[2].lower(),m[3].lower() if m[3] else None
  matches=[c for c in tree.find_all(exp.Column) if (c.table.lower()==first and c.name.lower()==second) if second] if second else [c for c in tree.find_all(exp.Column) if not c.table and c.name.lower()==first]
  if not matches:return no('error_column_not_in_AST_scope')
  return yes('receipt_ambiguous_column' if m[1].lower().startswith('ambiguous') else 'receipt_missing_column')
 m=re.fullmatch(r'near "([A-Za-z_][A-Za-z0-9_]*)": syntax error',message)
 if m:
  return yes('receipt_parser_supported_fixed_SQLite_syntax_error') if re.search(r'\b'+re.escape(m[1])+r'\b',sql,re.I) else no('syntax_token_not_in_actual_SQL')
 if message=='no tables specified' and not tables and any(isinstance(s,exp.Star) for s in tree.selects):return yes('receipt_star_without_FROM')
 if message=='aggregate functions are not allowed in the GROUP BY clause' and tree.args.get('group') is not None and any(tree.args['group'].find_all(exp.Count)):return yes('receipt_COUNT_in_GROUP_BY')
 return no('error_diagnostic_not_explicitly_whitelisted')
