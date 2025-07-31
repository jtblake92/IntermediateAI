# frozenlake_qlearning.py

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import time
import random
import keyboard
import os
import matplotlib.pyplot as plt

def initialize_environment(episode, total_episodes):
    record_points = [0, total_episodes // 2, total_episodes - 1]
    should_record = episode in record_points
    render_mode = "rgb_array" if should_record else None
    base_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=render_mode)

    if should_record:
        # Assign unique video names by setting name_prefix
        return RecordVideo(
            base_env,
            video_folder="./videos",
            episode_trigger=lambda ep: True,
            name_prefix=f"episode_{episode}"
        )
    else:
        return base_env

def initialize_q_table(env):
    return np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state, Q, env, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state, :])

def update_q_table(Q, state, action, reward, new_state, learning_rate, discount_factor):
    best_next = np.max(Q[new_state, :])
    td_target = reward + discount_factor * best_next
    td_error = td_target - Q[state, action]
    Q[state, action] += learning_rate * td_error

def train_q_learning(total_episodes=5000, learning_rate=0.8, discount_factor=0.95):
    Q = None
    rewards_per_episode = []

    for episode in range(total_episodes):
        env = initialize_environment(episode, total_episodes)
        if Q is None:
            Q = initialize_q_table(env)

        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            epsilon = 1.0 / (episode + 1)
            action = choose_action(state, Q, env, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            update_q_table(Q, state, action, reward, new_state, learning_rate, discount_factor)
            state = new_state
            episode_reward += reward

        rewards_per_episode.append(episode_reward)
        env.close()

    return Q, rewards_per_episode

def run_simulation(env, Q, max_attempts=4):
    attempts = 0
    user_quit = False

    while attempts < max_attempts and not user_quit:
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        visited_states = []

        print(f"\U0001F3AE Attempt #{attempts + 1}")
        time.sleep(0.5)

        while not done:
            if keyboard.is_pressed("esc"):
                print("\u274C Exited by user.")
                user_quit = True
                break

            print("\n\nRendering current state...")
            frame = env.render()
            time.sleep(0.5)

            q_vals = Q[state]
            best_action = np.argmax(q_vals)

            if visited_states.count(state) >= 2:
                action = env.action_space.sample()
                print(f"\u27F3 Loop breaker: tried a new action randomly from state {state}")
            else:
                action = best_action

            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            visited_states.append(state)
            print(f"Step {steps}: State={state}, Action={action}, Reward={reward}, Q-values={np.round(q_vals, 2)}, New State={new_state}", flush=True)
            state = new_state

        if user_quit:
            break

        attempts += 1
        print("Result: \u2705 Success!" if episode_reward == 1 else "\u274C Failed.", f"Total Reward: {episode_reward}, Steps: {steps}\n", flush=True)

        if episode_reward == 1:
            print(f"\u2728 Solved in {steps} steps after {attempts} attempts!")
            time.sleep(3)
            break

    env.close()

def plot_rewards(rewards_per_episode):
    def moving_average(data, window_size=100):
        return np.convolve(data, np.ones(window_size)/window_size, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(rewards_per_episode), label="Moving Average (100 episodes)")
    plt.title("Training Reward: Moving Average Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Q, rewards = train_q_learning()
    print("\u2705 Training complete. Final Q-table:")
    print(np.round(Q, 2))
    sim_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    run_simulation(sim_env, Q)
    plot_rewards(rewards)
