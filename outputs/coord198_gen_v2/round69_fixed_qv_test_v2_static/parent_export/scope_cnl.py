"""Editable CNL AST records for a narrow independent-population scope contradiction."""
import re,json,hashlib
from sqlglot import exp,parse_one

def parse(s):
 try:return parse_one(s,read='sqlite')
 except Exception:return None

def digest_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
SUPPORTED={'Select','From','Table','Identifier','TableAlias','Column','Join','Where','And','Or','EQ','NEQ','GT','GTE','LT','LTE','Between','In','Literal','Boolean','Null','Distinct','Alias','Paren','Not','Is','Star'}
TEXT={'Select':'Collect projected tuples from input rows. Preserve duplicate rows unless DISTINCT is explicit.','Join':'For each matching input pair, retain the pair; inner joins remove unmatched rows and may multiply rows. NULL equality is UNKNOWN.','And':'Both child predicates must hold in the same bound row.','Or':'At least one child predicate holds in the same bound row.','Between':'In one bound row require lower <= value <= upper, inclusive. NULL is UNKNOWN.','Column':'Read one bound property; alias and column identity are fixed.','Where':'Retain only rows whose predicate is TRUE; FALSE and UNKNOWN are removed.','Distinct':'Collapse identical projected tuples, including NULL tuples under SQLite distinct semantics.','Literal':'Use exactly this literal value and type.','Table':'Read this relation with its explicit alias.','EQ':'Require operands equal in this variable scope; NULL gives UNKNOWN.','GT':'Require left greater than right in this variable scope; NULL gives UNKNOWN.','LT':'Require left less than right in this variable scope; NULL gives UNKNOWN.'}

def encode(tree):
 records=[];handles={}
 def visit(x,path):
  if isinstance(x,exp.Expression):
   h='h'+str(len(records));record={'handle':h,'path':path,'type':x.key,'class':type(x).__name__,'args':{},'editable':type(x).__name__ in SUPPORTED};records.append(record);handles[id(x)]=h
   for k,v in x.args.items():record['args'][k]=visit(v,path+[k])
   if isinstance(x,exp.Join) and (x.args.get('side') or x.args.get('kind') in ('CROSS','FULL')):record['editable']=False
   record['sentence']=h+': '+TEXT.get(type(x).__name__,'Structured '+type(x).__name__+' with exact child handles and scalar attributes; no implicit repair.')+' Children and attributes: '+json.dumps(record['args'],sort_keys=True)
   if not record['editable']:record['sentence']='UNEDITABLE '+record['sentence']
   return {'ref':h}
  if isinstance(x,list):return [visit(v,path+[i]) for i,v in enumerate(x)]
  return {'scalar':x}
 root=visit(tree,[])
 return {'root':root,'records':records,'fully_supported':all(x['editable'] for x in records),'handle_map':handles,'canonical':tree.sql(dialect='sqlite'),'ast_fingerprint':digest_obj(tree.dump())}

def decode(doc):
 records={r['handle']:r for r in doc['records']}
 def read(x):
  if isinstance(x,list):return [read(v) for v in x]
  if 'ref' in x:
   r=records[x['ref']];return getattr(exp,r['class'])(**{k:read(v) for k,v in r['args'].items()})
  return x['scalar']
 return read(doc['root'])

def forms(w):
 result={w,w.rstrip('s')}
 if w.endswith('ality'):result.add(w[:-5])
 if w.endswith('ies'):result.add(w[:-3]+'y')
 return result

def words(s):return re.findall(r'[a-z]+',re.sub(r'([a-z])([A-Z])',r'\1 \2',s).lower())
def qintent(q):
 if not re.search(r'\bboth\b.*\band\b',q,re.I):return None,'no_explicit_both_populations'
 # Independent set-of-values syntax, not "a person both tall and rich".
 prefix=q[:re.search(r'\bboth\b',q,re.I).start()]
 if not re.search(r'\b(?:hometowns|countries|nations|locations)\b',prefix,re.I):return None,'subject_value_set_unknown'
 comps=list(re.finditer(r'\b(older than|younger than|above|below|more than|less than)\s+(\d+(?:\.\d+)?)',q,re.I))
 if len(comps)!=2:return None,'need_two_explicit_numeric_conditions'
 a,b=comps
 op=lambda m:'GT' if m[1].lower() in ('older than','above','more than') else 'LT'
 if op(a)==op(b):return None,'not_disjoint_opposed_conditions'
 high=float(a[2]) if op(a)=='GT' else float(b[2]);low=float(a[2]) if op(a)=='LT' else float(b[2])
 if high<low:return None,'ranges_overlap_scope_ambiguous'
 # Require repeated population noun OR the explicit "from which towns did both people" population construction.
 middle=q[a.end():b.start()]
 repeated=bool(re.search(r'\band\s+[a-z]+(?:\s+(?:of|with))?\s+(?:age\s+)?$',middle,re.I))
 from_people=bool(re.search(r'^From which hometowns did both people\b',q,re.I))
 if not repeated and not from_people:return None,'shared_vs_independent_population_unknown'
 return {'kind':'independent population witnesses for the same projected value tuple','conditions':[{'op':op(m),'value':m[2],'span':list(m.span()),'text':m[0],'variable':'event'+str(i)} for i,m in enumerate(comps)],'value_scope_span':prefix,'repeated_population':repeated,'null_policy':'each predicate must be TRUE; NULL event values satisfy neither; INTERSECT groups projected NULL values using SQLite set semantics'},None

def find_at(t,path):
 x=t
 for k in path:x=x[k] if isinstance(k,int) else x.args[k]
 return x

def compile_edit(q,parent,schema):
 intent,w=qintent(q)
 if not intent:return None,w,None
 t=parse(parent)
 if not isinstance(t,exp.Select):return None,'non_select_or_existing_set_scope',None
 doc=encode(t);roundtrip=decode(doc);assert roundtrip.sql(dialect='sqlite')==t.sql(dialect='sqlite')
 if not doc['fully_supported']:return None,'unsupported_CNL_node',doc
 if len(list(t.find_all(exp.Select)))!=1 or len(t.expressions)!=1 or not isinstance(t.expressions[0].unalias(),exp.Column):return None,'output_or_nested_scope_unknown',doc
 if any(t.args.get(k) for k in ('group','having','order','limit','offset','with_')):return None,'nonlocal_modifier_unsupported',doc
 known={tb['name'].lower():tb for tb in schema};sources={x.alias_or_name.lower():x.name.lower() for x in t.find_all(exp.Table)}
 def owner(c):
  opts=[(a,tb,meta) for a,tb in sources.items() if (not c.table or c.table.lower()==a) and tb in known for meta in known[tb]['columns'] if meta['name'].lower()==c.name.lower()]
  return opts[0] if len(opts)==1 else None
 for col in t.find_all(exp.Column):
  if owner(col) is None:return None,'unbound_or_ambiguous_column_UNEDITABLE',doc
 output=t.expressions[0].unalias();bound=owner(output);colwords=words(output.name)
 # Explicit asked output key, not an inferred entity key or an arbitrary parent projection.
 qwords=words(intent['value_scope_span']);match=any(forms(a)&forms(b) or any(min(len(x),len(y))>=6 and (x.startswith(y) or y.startswith(x)) for x in forms(a) for y in forms(b)) for a in colwords for b in qwords)
 if not match:return None,'projected_value_not_question_key',doc
 where=t.args.get('where')
 if where is None:return None,'no_editable_predicate_scope',doc
 target_nodes=[];target_field=None;literal_by_value={}
 for node in where.walk():
  if isinstance(node,(exp.GT,exp.GTE,exp.LT,exp.LTE,exp.EQ)) and isinstance(node.this,exp.Column) and isinstance(node.expression,exp.Literal):
   value=node.expression.this
   if value in [x['value'] for x in intent['conditions']]:target_nodes.append(node);literal_by_value[value]=node.expression
  elif isinstance(node,exp.Between) and isinstance(node.this,exp.Column):
   lo,hi=node.args.get('low'),node.args.get('high')
   if isinstance(lo,exp.Literal) and isinstance(hi,exp.Literal) and {lo.this,hi.this}=={x['value'] for x in intent['conditions']}:target_nodes=[node];literal_by_value={lo.this:lo,hi.this:hi};break
 if len(target_nodes)==1 and isinstance(target_nodes[0],exp.Between):region=target_nodes[0];field=region.this
 elif len(target_nodes)==2:
  if target_nodes[0].this.sql(dialect='sqlite')!=target_nodes[1].this.sql(dialect='sqlite'):return None,'different_event_columns_unknown',doc
  if target_nodes[0].parent is not target_nodes[1].parent or not isinstance(target_nodes[0].parent,(exp.And,exp.Or)):return None,'nonlocal_predicate_pair_unknown',doc
  region=target_nodes[0].parent;field=target_nodes[0].this
 else:return None,'no_exact_two_threshold_predicate_handle',doc
 if re.search(r'older than|younger than|\bage\b',q,re.I):
  if 'age' not in words(field.name):return None,'event_attribute_not_question_age',doc
 elif not any(min(len(x),len(y))>=6 and (x.startswith(y) or y.startswith(x)) for x in words(field.name) for y in words(q)):
  return None,'event_attribute_not_question_supported',doc
 if owner(field)[1]!=bound[1]:return None,'projection_event_grain_requires03',doc
 if owner(field)[2].get('pk') or not re.search(r'INT|REAL|NUM|DEC|FLOAT',owner(field)[2]['type'],re.I):return None,'not_numeric_event_attribute',doc
 record=next(r for r in doc['records'] if r['handle']==doc['handle_map'][id(region)]);path=record['path'];branches=[]
 # One controlled scope sentence is replaced. Both child branches preserve every
 # non-target node, including accidental parent joins/filters; never silently fix them.
 for condition in intent['conditions']:
  branch=decode(doc);target=find_at(branch,path);pred=getattr(exp,condition['op'])(this=field.copy(),expression=literal_by_value[condition['value']].copy());target.replace(pred);branches.append(branch)
  check=branch.copy();find_at(check,path).replace(region.copy());assert check.sql(dialect='sqlite')==t.sql(dialect='sqlite')
 new=exp.Intersect(this=branches[0],expression=branches[1],distinct=True);sql=new.sql(dialect='sqlite');parse_new=parse(sql)
 if parse_new is None:return None,'compiled_set_parse_failed',doc
 if len(sql.encode('utf-8'))>16000:return None,'compiled_SQL_byte_budget_abstain',doc
 edit={'handle':record['handle'],'node_path':path,'old_sentence':record['sentence'],'new_sentence':'For each projected value tuple require one event0 satisfying '+intent['conditions'][0]['text']+' and independently one event1 satisfying '+intent['conditions'][1]['text']+'. Return the intersection of projected value sets.','question_spans':intent['conditions'],'scope_intent':intent,'compile_rule':'duplicate unchanged SELECT context into two INTERSECT branches; replace only target predicate region per branch','protected_context_proof':'Replacing each branch target with original region exactly restores original canonical AST; all other nodes are identical.','multiplicity_change':'explicit projected value-set intersection deduplicates output; not entity-name grouping or bag intersection','null_policy':intent['null_policy']}
 return {'sql':sql,'edit':edit,'cnl':{k:v for k,v in doc.items() if k!='handle_map'},'source':'training-derived deterministic CNL subset; no model or reference SQL'},None,doc
