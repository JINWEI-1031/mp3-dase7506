"""Independent deployment wrapper; original polarity Agent and C4 behavior.

The original 193 score belongs to the original entry/weight pair. This wrapper
requires a separate cold reproduction before an equivalence claim is made.
"""
from step107_polarity_c4_193_identity import bind


def create_agent(checkpoint=None):
    if checkpoint is not None:
        raise ValueError('The reviewed pair uses the original default checkpoint only')
    bind()
    from step107_polarity_selected import create_agent as original
    return original()
