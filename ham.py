import gym


class AbstractMachine:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def step(self, inputs):
        raise NotImplementedError


class PrimitiveMachine(AbstractMachine):
    def __init__(self, policy):
        super().__init__([1], [1])
        self.policy = policy

    def step(self, inputs):
        return [self.policy(inputs[0])]


class CompositeMachine(AbstractMachine):
    def __init__(self, submachines, transition):
        inputs = sum([machine.inputs for machine in submachines], [])
        outputs = sum([machine.outputs for machine in submachines], [])
        super().__init__(inputs, outputs)
        self.submachines = submachines
        self.transition = transition

    def step(self, inputs):
        submachine_outputs = [submachine.step(inputs) for submachine in self.submachines]
        return self.transition(submachine_outputs)


class CartPoleMachine(CompositeMachine):
    def __init__(self):
        skill_0 = PrimitiveMachine(lambda s: 1 if s[2] < 0 else 0)
        skill_1 = PrimitiveMachine(lambda s: 1 if s[2] >= 0 else 0)
        skills = [skill_0, skill_1]
        super().__init__(skills, lambda submachine_outputs: submachine_outputs[0] if submachine_outputs[1][0] == 0 else submachine_outputs[1])


class HAMAgent:
    def __init__(self, machine):
        self.machine = machine

    def act(self, state):
        return self.machine.step([state])


env = gym.make('CartPole-v1', render_mode="rgb_array")
machine = CartPoleMachine()
agent = HAMAgent(machine)

for i_episode in range(20):
    observation = env.reset()[0]

    for t in range(100):
        env.render()
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action[0])

        if done:
            print(f'Episode finished after {t+1} steps')
            break

env.close()
