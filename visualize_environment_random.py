import time
import numpy as np
import gym

# This import registers your custom environment ID: DRLRobot-v0
import AutonomousRobotNavigation2


def main():
    # Create environment
    env = gym.make("DRLRobot-v0")

    # Reset environment
    obs = env.reset()

    print("Environment started.")
    print("Observation shape:", np.array(obs).shape)

    episode_count = 5
    max_steps_per_episode = 1500

    for ep in range(episode_count):
        obs = env.reset()
        done = False
        total_reward = 0.0

        print(f"\nEpisode {ep + 1} started")

        for step in range(max_steps_per_episode):
            # Random action
            action_id = env.action_space.spaces[0].sample()
            action_params = env.action_space.spaces[1].sample()

            action = (action_id, action_params)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Visualize environment
            env.render()

            # Slow down visualization slightly
            time.sleep(0.01)

            if done:
                print(f"Episode {ep + 1} finished at step {step + 1}")
                print(f"Total reward: {total_reward:.4f}")
                break

    env.close()
    print("\nVisualization finished.")


if __name__ == "__main__":
    main()
