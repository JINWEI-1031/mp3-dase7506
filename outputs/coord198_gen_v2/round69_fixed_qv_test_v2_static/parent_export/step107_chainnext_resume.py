"""Stateful parent suspension primitive, separate from terminal SQL fallback."""
import copy
class Suspension:
    def __init__(self,parent,action):
        self.parent=copy.deepcopy(parent)
        self.action=copy.deepcopy(action)
        self.delivered=False
    def resume(self,observation):
        if not self.delivered:
            if self.action['tool']!='submit' and observation.get('remaining',1)<=0:
                raise RuntimeError('Cannot replay a queued charged action without budget')
            self.delivered=True
            return copy.deepcopy(self.action)
        return self.parent.act(observation)
