"""Explicit immutable D/C3 source axis; terminal timing/Evidence are unchanged."""
import hashlib
import importlib
import json
import os
from pathlib import Path
from step107_terminal_qp2_agent import Agent,PARENTS
import step107_qwen_integration1_agent as frozen

CONFIG='outputs/step107-terminal-qp2-modelaxis-preregister.json'
SPEC_SHAS={
 'D':{'adapter_model.safetensors':'cb69a6c73ff0b4e05882b0b1fc680c7a1ddb63e41ca7c14b884dfa2de1c0c4b2',
      'adapter_config.json':'455cd0183de1a30c2bf0b7d4d1dd0be7b670191a506b76cc56117ef5cb9717a0'},
 'C3':{'adapter_model.safetensors':'8298a39b64eb62497943a11e49825aab90bfb0ada155cda98942021da0f83793',
       'adapter_config.json':'42bb4f1c4a1a2ba0d133f389b9d3f7f426fd77a67eb3e2ba4d74316275dbd60e'}}
ACTIVE_SPEC=None


def validate_spec(config,name):
    if name not in SPEC_SHAS:raise ValueError('Explicit model-spec D or C3 required, no default')
    spec=config['model_specs'][name]
    if spec['name']!=name or spec['adapter']['sha256']!=SPEC_SHAS[name]:raise ValueError('Immutable model-spec SHA mismatch')
    if config['base']['repo']!='Qwen/Qwen3-0.6B' or config['base']['revision']!='c1899de289a04d12100db370d81485cdf75e47ca':
        raise ValueError('D/C3 adapters can only use their original Qwen3 base')
    directory=Path(spec['adapter']['relative_path'])
    for filename,expected in SPEC_SHAS[name].items():
        with (directory/filename).open('rb') as file:actual=hashlib.file_digest(file,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Actual selected adapter file mismatch: '+filename)
    return spec


def model_identity(info,spec):
    if info.get('repository')!='Qwen/Qwen3-0.6B' or info.get('revision')!='c1899de289a04d12100db370d81485cdf75e47ca':
        raise ValueError('Actual base model identity mismatch')
    if info.get('weights_sha256')!='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b':
        raise ValueError('Actual base weight SHA mismatch')
    adapter=info.get('adapter') or {}
    if adapter.get('sha256')!=spec['adapter']['sha256']['adapter_model.safetensors'] or adapter.get('name')!=spec['name']:
        raise ValueError('Actual adapter SHA/name mismatch; never call D C3 or vice versa')
    if info.get('parameters',10**10)>1_000_000_000:raise ValueError('Parameter cap')
    return True


def configuration():
    global ACTIVE_SPEC
    config=json.loads(Path(os.environ.get('MP3_TQP_CONFIG',CONFIG)).read_text(encoding='utf-8'))
    name=os.environ.get('MP3_TQP_MODEL_SPEC')
    spec=validate_spec(config,name)
    if ACTIVE_SPEC is not None and ACTIVE_SPEC!=name:raise ValueError('One model-spec per fresh process; no active adapter switching')
    if Path(os.environ.get('MP3_ADAPTER_DIR','')).resolve()!=Path(spec['adapter']['relative_path']).resolve():
        raise ValueError('Selected directory does not match declared model-spec; no silent fallback')
    if os.environ.get('MP3_STEP63_ARM')!=name:raise ValueError('Actual adapter log label must match selected spec')
    ACTIVE_SPEC=name
    parent=os.environ.get('MP3_TQP_PARENT')
    if parent not in PARENTS:raise ValueError('Only two frozen parents permitted')
    module,expected=PARENTS[parent];path=Path('submission')/(module+'.py')
    if config['parents'][parent]['entry']!=path.as_posix() or config['parents'][parent]['entry_sha256']!=expected:
        raise ValueError('Parent config mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Actual parent source mismatch')
    return config,parent,module


def create_agent(checkpoint=None):
    config,parent,module=configuration();name=os.environ['MP3_TQP_MODEL_SPEC'];spec=config['model_specs'][name]
    frozen.install_meter()
    factory=importlib.import_module(module).create_agent
    # Original ledger validates actual model object/repository/revision/selected
    # weight SHA at every parent/extra generation; no D-only factory is called.
    return Agent(factory(checkpoint),spec['adapter']['sha256']['adapter_model.safetensors'],
                 os.environ['MP3_TQP_POLICY'])
