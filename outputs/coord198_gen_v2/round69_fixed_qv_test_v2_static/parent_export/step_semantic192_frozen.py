from step_semantic192_selective import Agent
def create_agent(checkpoint=None):
    agent=Agent(checkpoint)
    agent.semantic_families=set(['event', 'relation', 'negative'])
    return agent
