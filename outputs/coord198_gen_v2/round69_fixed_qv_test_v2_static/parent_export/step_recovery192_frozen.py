from step_recovery192_balanced import Agent
def create_agent(checkpoint=None):
    agent=Agent(checkpoint)
    agent.features=set(['exit', 'priority', 'dedup'])
    return agent
