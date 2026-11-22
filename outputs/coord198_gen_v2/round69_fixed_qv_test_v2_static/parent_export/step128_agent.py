"""Only instantiated when the new training/validation selection gate has passed."""
import json
from pathlib import Path
from step127_agent import Agent


def create_agent(checkpoint=None):
    import os
    arm=json.loads(Path(__file__).with_name('step128-selection.json').read_text(encoding='utf-8'))['arm']
    previous=os.environ.get('MP3_STEP127_ARM');os.environ['MP3_STEP127_ARM']=arm
    try:return Agent(checkpoint)
    finally:
        if previous is None:os.environ.pop('MP3_STEP127_ARM',None)
        else:os.environ['MP3_STEP127_ARM']=previous
