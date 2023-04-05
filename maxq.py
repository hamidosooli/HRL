import gym


class Skill:
    def __init__(self, policy, id):
        self.policy = policy
        self.id = id


class CompositeTask:
    def __init__(self, skills, id):
        self.skills = skills
        self.id = id

    def subtask(self, state):
        raise NotImplementedError

    def act(self, state):
        skill = self.skills[self.subtask(state)]
        return skill.policy(state)


class CartPoleTask(CompositeTask):
    def __init__(self, skills, id):
        super().__init__(skills, id)

    def subtask(self, state):
        if state[2] < 0:
            return 0
        else:
            return 1


class MAXQAgent:
    def __init__(self, tasks, root):
        self.tasks = tasks
        self.current_task = root

    def act(self, state):
        while isinstance(self.current_task, CompositeTask):
            subtask = self.current_task.subtask(state)
            self.current_task = self.tasks[self.current_task.id].skills[subtask]

        return self.current_task.policy(state)


env = gym.make('CartPole-v1', render_mode="rgb_array")
skills = [Skill(lambda s: 1, 0), Skill(lambda s: 0, 1)]
tasks = [CartPoleTask(skills, 0)]
agent = MAXQAgent(tasks, tasks[0])

for i_episode in range(20):
    observation = env.reset()[0]

    for t in range(100):
        env.render()
        action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(action)

        if done:
            print(f'Episode finished after {t+1} steps')
            agent.current_task = tasks[0]
            break

env.close()
