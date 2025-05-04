class MarioAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()
