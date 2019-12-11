import time
import gym
import random
import numpy as np
import math

env = gym.make('Pendulum-v0')
MAX_EPISODE_STEPS = 200 

def find_theta(cos, sin):
    th = math.asin(sin)
    if sin > 0 > cos:
        return th + (math.pi / 2)
    elif sin < 0 and cos < 0:
        return th - (math.pi / 2)
    else:
        return th


class Agent():

    def __init__(self):

        pass

    def Train(self):
        PI = math.pi
        start_time = time.time()
        deadline = 15 * 60
        step = 0.2
        theta_range = np.arange(-PI, PI, step)
        omega_range = np.arange(-8, 8, step)
        action_range = np.arange(-2, 2, step)
        q = np.zeros([theta_range.size, omega_range.size, action_range.size])
        alpha = 0.7
        while time.time() - start_time < deadline:
            state = env.reset()
            step_count = 0
            done = False
            while not done and step_count < MAX_EPISODE_STEPS:
                theta = find_theta(state[0], state[1])
                th_pos = int((theta + PI) // step)
                om_pos = int((state[2] + 8) // step)
                best_actions = np.argmax(q, axis=2)
                best_action_pos = best_actions[th_pos][om_pos]
                action = best_action_pos * step - 2
                current_q = q[th_pos][om_pos][best_action_pos]
                state, reward, done, _ = env.step([action])
                step_count += 1
                new_theta = find_theta(state[0], state[1])
                new_th_pos = int((new_theta + PI) // step)
                new_om_pos = int((state[2] + 8) // step)
                new_best_action_pos = best_actions[new_th_pos][new_om_pos]
                q[th_pos][om_pos][best_action_pos] = current_q + alpha * (
                        reward + 0.9 * q[new_th_pos][new_om_pos][new_best_action_pos] - current_q)
            alpha = 0.7 - (0.68 * (time.time() - start_time) / deadline)
        np.save('Qs', q)

    def Play(self, render=True):
        q = np.load('Qs.npy')
        scores = []
        PI = math.pi
        step = 0.2
        for episode_count in range(1000):
            episode_count += 1
            print('******Episode ', episode_count)
            state = env.reset()
            score = 0
            done = False
            step_count = 0
            while not (done) and step_count < MAX_EPISODE_STEPS:
                theta = find_theta(state[0], state[1])
                th_pos = int((theta + PI) // step)
                om_pos = int((state[2] + 8) // step)
                best_actions = np.argmax(q, axis=2)
                best_action_pos = best_actions[th_pos][om_pos]
                action = best_action_pos * step - 2
                current_q = q[th_pos][om_pos][best_action_pos]
                state, reward, done, _ = env.step([action])
                step_count += 1
                score += reward
                if render:
                    env.render()
                    time.sleep(0.04) 
            scores.append(score)
            print('Score:', score)
        print("Average score over 1000 run : ", np.array(scores).mean())
        return scores, np.array(scores).mean()

    def Test(self):
        START_TIME = time.time()
        self.Train()
        TOTAL_TIME = time.time() - START_TIME
        return self.Play(), TOTAL_TIME


agent = Agent()
agent.Test()

