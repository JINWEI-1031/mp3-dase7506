"""Portable entry for the frozen R69 q025/v075 compiler-first B controller.

Original decision modules are preserved byte-for-byte. This entry only binds paths,
initializes the original runtime observers once, and creates per-task state.
"""
import os,sys,json,hashlib
from pathlib import Path
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
SNAP=ROOT/'outputs/coord198_gen_v2/round69_fixed_qv_test_v2_static'
sys.path[:0]=[str(SNAP),str(SNAP/'parent_export')]

@lru_cache(maxsize=1)
def initialize():
    cfg=json.loads((ROOT/'config.json').read_text())
    for path,expected in json.loads((ROOT/'frozen-source-manifest.json').read_text()).items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen source changed: '+path)
    base=Path(os.environ['MP3_BASE_MODEL']).resolve();adapter=Path(os.environ['MP3_ADAPTER_DIR']).resolve()
    cfg.update(base_path=str(base),adapter_path=str(adapter))
    for path,expected in [(base/'model.safetensors',cfg['base_weights_SHA256']),(adapter/'adapter_model.safetensors',cfg['adapter_SHA256']),(adapter/'adapter_config.json',cfg['adapter_config_SHA256'])]:
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('Unexpected model identity: '+path.name)
    log=Path(os.environ.get('MP3_LOG_DIR',str(ROOT/'run-logs'))).resolve();log.mkdir(parents=True,exist_ok=True)
    # Fix all original behavior controls while leaving the declared external paths explicit.
    for k in list(os.environ):
        if k.startswith('MP3_'):del os.environ[k]
    os.environ.update(cfg['native_settings'],MP3_BASE_MODEL=str(base),MP3_ADAPTER_DIR=str(adapter),MP3_STEP63_MODEL_LOG=str(log/'original-model63.jsonl'),MP3_STEP125_MODEL_LOG=str(log/'original-model125.jsonl'),MP3_STEP5_MODEL_LOG=str(log/'original-model5.jsonl'),MP3_STEP32_MODEL_LOG=str(log/'original-model32.jsonl'))
    import torch,random
    random.seed(61);torch.manual_seed(61)
    from loader_guard import install
    install(cfg)
    import step63_agent
    from runtime import Runtime
    runtime=Runtime(cfg,log/'all-model-generations.jsonl');runtime.install(step63_agent)
    return runtime

class Episode:
    def __init__(self):self.agent=None
    def act(self,observation):
        if self.agent is None:
            runtime=initialize()
            if 'step125_agent' in sys.modules:sys.modules['step125_agent']._CACHE.clear()
            from runtime import State,ParentTap,ActiveAgent
            from scope_submission import create_agent as parent_factory
            from compiler_agent import Agent
            state=State(observation['task']['id'])
            self.agent=ActiveAgent(Agent(ParentTap(parent_factory(SNAP/'PARENT-CONFIG.json'),state),runtime,state),state)
        return self.agent.act(observation)

def create_agent(checkpoint=None):
    if checkpoint is not None:raise ValueError('Use bundled config.json; no checkpoint override')
    return Episode()
