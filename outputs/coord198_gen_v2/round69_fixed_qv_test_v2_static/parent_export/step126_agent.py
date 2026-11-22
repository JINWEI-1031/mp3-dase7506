"""Load the pre-test selected E3/E4 configuration."""
import json
from pathlib import Path
from step125_agent import Agent
from step124_agent import create_agent as baseline


def create_agent(checkpoint=None):
    config=json.loads(Path(__file__).with_name('step126-selection.json').read_text(encoding='utf-8'))
    if config['arms']=='baseline':return baseline(checkpoint)
    # Construct without relying on a process-level experiment switch.
    import os
    previous=os.environ.get('MP3_STEP125_ARMS')
    os.environ['MP3_STEP125_ARMS']=config['arms']
    try:return Agent(checkpoint)
    finally:
        if previous is None:os.environ.pop('MP3_STEP125_ARMS',None)
        else:os.environ['MP3_STEP125_ARMS']=previous
