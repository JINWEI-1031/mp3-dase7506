"""Typed intentional module refusals only; unknown backend errors are hard."""
from sqlglot.errors import ParseError,TokenError,OptimizeError
class RuleRefusal(ValueError):
 pass
NORMAL=(RuleRefusal,ParseError,TokenError,OptimizeError)
def _causes(error,input_validation=False):
 seen=set();cause=error
 while cause is not None and id(cause) not in seen:
  seen.add(id(cause))
  if isinstance(cause,(MemoryError,RecursionError,AssertionError,RuntimeError,SystemError,OSError)):raise cause
  if cause is not error and type(cause) not in NORMAL:raise cause
  cause=cause.__cause__ or cause.__context__
def _ordinary(error):
 _causes(error)
 if type(error) not in NORMAL:raise error
def _input_error(error):
 # This handler covers validation only, never any compiler/backend invocation.
 if type(error) not in (ValueError,TypeError,KeyError):raise error
 _causes(error,True)
def _rejection_type(error):
 return 'ValueError' if type(error) is RuleRefusal else type(error).__name__
