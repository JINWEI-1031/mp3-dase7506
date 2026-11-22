"""Fail-closed path binding for the independently packaged polarity/C4 pair.

Only the new deployment wrapper uses this module. No model is imported here.
"""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'deployment/polarity-c4-193-v1/runtime-config.json'
MANIFEST = ROOT / 'deployment/polarity-c4-193-v1/runtime-manifest.json'


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_member(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Unsafe deployment path')
    return path


def verify_sources(root=ROOT):
    manifest = read(root / MANIFEST.relative_to(ROOT))
    for name, expected in manifest['files'].items():
        if sha(safe_member(root, name)) != expected:
            raise ValueError('Frozen runtime source changed: ' + name)
    return manifest


def verify_model_files(base, adapter, config):
    base, adapter = Path(base).resolve(), Path(adapter).resolve()
    if base.name != config['base']['revision']:
        raise ValueError('Base snapshot directory must retain the pinned revision name')
    for directory, hashes in ((base, config['base']['files']),
                              (adapter, config['adapter']['files'])):
        for name, expected in hashes.items():
            if sha(directory / name) != expected:
                raise ValueError('Unexpected model file: ' + name)
    return base, adapter


@lru_cache(maxsize=1)
def bind():
    verify_sources()
    config = read(CONFIG)
    base, adapter = verify_model_files(os.environ['MP3_BASE_MODEL'],
                                       os.environ['MP3_ADAPTER_DIR'], config)
    allowed = set(config['environment']) | {'MP3_BASE_MODEL', 'MP3_ADAPTER_DIR',
                                           'MP3_STEP63_MODEL_LOG', 'MP3_STEP125_MODEL_LOG'}
    unknown = sorted(k for k in os.environ if k.startswith('MP3_') and k not in allowed)
    if unknown:
        raise ValueError('Unreviewed inherited MP3 settings: ' + ','.join(unknown))
    for name, value in config['environment'].items():
        if name in os.environ and os.environ[name] != value:
            raise ValueError('Frozen behavior setting differs: ' + name)
    os.environ.update(config['environment'])
    os.environ.update(MP3_BASE_MODEL=str(base), MP3_ADAPTER_DIR=str(adapter))
    for name in ('MP3_STEP63_MODEL_LOG', 'MP3_STEP125_MODEL_LOG'):
        if not os.environ.get(name):
            raise ValueError('A fresh output log path is required: ' + name)
    return config
