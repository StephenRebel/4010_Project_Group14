from RLMusicEnv import RLMusicBotEnv
from final_visualization import show_final_sheet
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

env = RLMusicBotEnv(bars=4)

#PPO
# ppo_env = RLMusicBotEnv(bars=8, render_mode="none")
# ppo_model = PPO("MlpPolicy", ppo_env, verbose=1)
# ppo_model.learn(total_timesteps=100000)
# ppo_model.save("ppo_musicbot")

# # Test the trained model
# obs, _ = ppo_env.reset()
# done = False
# while not done:
#     action, _states = ppo_model.predict(obs)
#     obs, ppo_reward, done, _, info = ppo_env.step(action)
#     if ppo_env.render_mode == "human":
#         ppo_env.render()

# #DQN
# dqn_env = RLMusicBotEnv(bars=16, render_mode=None)
# dqn_model = DQN("MlpPolicy", dqn_env, verbose=1)
# dqn_model.learn(total_timesteps=100000)
# dqn_model.save("dqn_musicbot")

# # Test the trained model
# obs, _ = dqn_env.reset()
# done = False
# while not done:
#     action, _states = dqn_model.predict(obs)
#     obs, dqn_reward, done, _, info = dqn_env.step(action)
#     if dqn_env.render_mode == "human":
#         dqn_env.render()

# #Random
# env = RLMusicBotEnv(bars=8, render_mode=None)
# obs, _ = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     new_note = env._map_action_to_note(action)
#     obs, reward, done, _, _ = env.step(action)
#     if env.render_mode == "human":
#         env.render()
        
# # Save to MIDI
# env.save_to_midi("random_song.mid")
# # show_final_sheet(env._musical_score)

# print("\nPPO test completed.")
# print("musical score: ")
# print(ppo_env._musical_score)
# print("chord progression: ")
# print(", ".join(ppo_env.chord_progression))
# print("reward: ")
# print(ppo_reward)
# ppo_env.save_to_midi("random_song_ppo.mid")

# print("\nDQN test completed.")
# print("musical score: ")
# print(dqn_env._musical_score)
# print("chord progression: ")
# print(", ".join(dqn_env.chord_progression))
# print("reward: ")
# print(dqn_reward)
# dqn_env.save_to_midi("random_song_dqn.mid")

# print("\nRandom test completed.")
# print("musical score: ")
# print(env._musical_score)
# print("chord progression: ")
# print(", ".join(env.chord_progression))
# print("reward: ")
# print(reward)