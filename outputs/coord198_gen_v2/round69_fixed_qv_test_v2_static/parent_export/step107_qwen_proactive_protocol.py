"""Reviewable protocol primitives, not an enabled agent or SQL semantic oracle."""
from dataclasses import dataclass,field
import copy

UNKNOWN='UNKNOWN';CERTIFIED='CERTIFIED';HARD='HARD_CONFLICT'

@dataclass(frozen=True)
class Evidence:
    status:str=UNKNOWN
    hard_conflicts:tuple=()
    roles:int=0
    values:int=0
    relations:int=0
    aggregates:int=0
    unresolved:int=0
    provenance:tuple=()
    def __post_init__(self):
        assert self.status in (UNKNOWN,CERTIFIED,HARD)
        if self.hard_conflicts:assert self.status==HARD
        if self.status==HARD:assert self.hard_conflicts
        if self.status==CERTIFIED:assert not self.hard_conflicts and self.unresolved==0 and self.provenance
    @property
    def score(self):return 2*self.roles+2*self.values+2*self.relations+self.aggregates-2*self.unresolved

@dataclass
class GenerationLedger:
    total:int=0
    proactive:int=0
    expected_adapter_sha256:str=''
    model_object_id:int|None=None
    def can_extra(self):return self.total<6 and self.proactive<1
    def register(self,model,metadata,extra=False):
        assert metadata['repository']=='Qwen/Qwen3-0.6B'
        assert metadata['revision']=='c1899de289a04d12100db370d81485cdf75e47ca'
        assert metadata['parameters']<=1_000_000_000
        assert metadata.get('adapter') and metadata['adapter']['sha256']==self.expected_adapter_sha256
        if self.model_object_id is None:self.model_object_id=id(model)
        assert id(model)==self.model_object_id,'Never load another live model/adapter in an episode'
        assert self.total<6 and (not extra or self.proactive<1)
        self.total+=1;self.proactive+=int(extra)

def opportunity(obs,pending,parent_evidence,ledger,hint_first=True):
    """Reserve only costs of the one review and exact held action; no refund."""
    original_seen=any(e['tool'] in ('check','execute') and e.get('sql')==obs['issue_sql'] for e in obs['history'])
    if not original_seen:return False,{'reason':'original_draft_not_executed'}
    if parent_evidence.status==CERTIFIED:return False,{'reason':'parent_semantically_certified'}
    if not ledger.can_extra():return False,{'reason':'completion_cap'}
    if pending['tool']=='schema':return False,{'reason':'let_parent_schema_finish_first'}
    if hint_first and pending['tool']=='repair_hint' and not any(e['tool']=='repair_hint' for e in obs['history']):
        return False,{'reason':'let_first_cheap_hint_run'}
    schema_cost=int(not any(e['tool']=='schema' for e in obs['history']))
    held_cost=int(pending['tool']!='submit')
    required=schema_cost+1+held_cost
    return obs['remaining']>=required,{'schema_cost':schema_cost,'candidate_check_cost':1,'held_action_cost':held_cost,
                                     'required':required,'actual_remaining':obs['remaining'],'reason':'unknown_or_conflicted_has_real_budget'}

def choose(parent,candidate,parent_public,candidate_public,policy='evidence_margin',margin=2):
    """No grader label input. UNKNOWN is eligible, never automatically wrong."""
    if candidate.status==HARD or candidate.hard_conflicts:return False,'explicit_candidate_conflict'
    if parent.status==CERTIFIED:return False,'certified_parent_preserved'
    if not candidate_public:return False,'no_real_public_check_pass'
    if candidate.status==CERTIFIED:return True,'independent_certificate'
    if policy=='keep_unknown_public' and parent_public:return False,'registered_keep_unknown_control'
    known=candidate.roles+candidate.values+candidate.relations+candidate.aggregates
    if known==0:return False,'no_question_schema_evidence'
    if not parent_public and candidate.score>=parent.score:return True,'checked_repair_with_no_explicit_conflict'
    if candidate.score-parent.score>=margin and candidate.unresolved<=parent.unresolved:
        return True,'unknown_candidate_has_stronger_runtime_evidence'
    return False,'insufficient_margin_keep_or_abstain'

def action(event):return {k:v for k,v in event.items() if k!='observation'}

@dataclass
class ParentFrame:
    parent:object
    pending_action:dict
    owned_indices:set=field(default_factory=set)
    @classmethod
    def capture(cls,parent,pending,obs):return cls(parent,copy.deepcopy(pending),set(range(len(obs['history']))))
    def parent_view(self,actual):
        # Default design exposes the full genuine observation. No review event
        # is hidden, reordered, or relabelled; consequent flow changes are costs
        # to measure, not a claim of counterfactual behavioral equivalence.
        return copy.deepcopy(actual)
    def existing_pending(self,actual):
        for i,e in enumerate(actual['history']):
            if action(e)==self.pending_action:
                self.owned_indices.add(i);return i
        return None
    def restore_action(self):return copy.deepcopy(self.pending_action)

def prompt_context(question,schema,columns,parent_sql,mode,actual_feedback=None,original_issue_sql=None):
    if mode=='repair_original':
        assert isinstance(original_issue_sql,str)
        return ('Repair the SQLite query to answer the question. Output only the corrected SQL.',
                {'question':question,'draft':original_issue_sql,'columns':columns,'schema':schema})
    if mode=='plan':
        return ('Write one SQLite SELECT answering the question using the observed schema. Preserve all question conditions, entity roles and aggregation scope. Output SQL only.',
                {'question':question,'schema':schema,'columns':columns})
    assert mode=='edit'
    context={'question':question,'draft':parent_sql,'columns':columns,'schema':schema}
    if actual_feedback is not None:context['feedback']=copy.deepcopy(actual_feedback)
    return ('Repair the SQLite query to answer the question. Output only the corrected SQL.',context)

def certify_simple_count(question,sql,schema,actual_public_pass):
    """One deliberately narrow, fully consumed grammar; everything else UNKNOWN."""
    import re
    from sqlglot import exp
    from repair_rules import parse
    m=re.fullmatch(r'\s*how many ([a-z_ ]+?)(?: are there)?\s*\?\s*',question,re.I)
    t=parse(sql)
    if not actual_public_pass or not m or not isinstance(t,exp.Select):return Evidence(unresolved=1)
    token=lambda s:tuple(w.rstrip('s') for w in re.findall('[a-z]+',s.lower()))
    tables=[x for x in schema if token(x['name'])==token(m[1])]
    if len(tables)!=1 or len(t.selects)!=1:return Evidence(unresolved=1)
    if any(t.args.get(k) for k in ('joins','where','group','having','distinct','limit','offset','with','order')):return Evidence(unresolved=1)
    root=t.args.get('from');e=t.selects[0].unalias()
    if not root or not isinstance(root.this,exp.Table) or root.this.name.lower()!=tables[0]['name'].lower():return Evidence(unresolved=1)
    if not isinstance(e,exp.Count) or not (isinstance(e.this,exp.Star) or isinstance(e.this,exp.Literal) and not e.this.is_string and e.this.this=='1'):
        return Evidence(unresolved=1)
    return Evidence(CERTIFIED,roles=1,aggregates=1,provenance=('full_question_simple_count_grammar','schema_single_relation','actual_public_check'))

# Deliberately no create_agent: complete detector, meter hooks, controller loop,
# official trace replay and end-to-end pilot must be reviewed before activation.
