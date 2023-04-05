import numpy as np
import gym


class Option:
    def __init__(self, policy, termination, id):
        self.policy = policy
        self.termination = termination
        self.id = id


class OptionAgent:
    def __init__(self, options):
        self.options = options
        self.current_option = None

    def act(self, state):
        if self.current_option is None or self.current_option.termination(state):
            self.current_option = np.random.choice(self.options)
        return self.current_option.policy(state)


env = gym.make('CartPole-v1', render_mode="rgb_array")
options = [Option(lambda s: 1 if s[2] < 0 else 0, lambda s: s[2] >= 0, 0),
           Option(lambda s: 1 if s[2] >= 0 else 0, lambda s: s[2] < 0, 1)]

agent = OptionAgent(options)
for i_episode in range(20):
    observation = env.reset()[0]

    for t in range(100):
        env.render()
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)

        if done:
            print(f'Episode finished after {t+1} steps')
            break
env.close()
