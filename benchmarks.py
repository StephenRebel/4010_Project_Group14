from RLMusicEnv import RLMusicBotEnv
from final_visualization import show_final_sheet
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

# Change to 4 for RandomAgent 
env = RLMusicBotEnv(bars=8, debug=True)

twinkle_twinkle_score = [
    [(60, 1.0, 0.8), (60, 1.0, 0.8), (67, 1.0, 0.8), (67, 1.0, 0.8)],  # Bar 1: C C G G
    [(69, 1.0, 0.8), (69, 1.0, 0.8), (67, 2.0, 0.8)],                    # Bar 2: A A G
    [(65, 1.0, 0.8), (65, 1.0, 0.8), (64, 1.0, 0.8), (64, 1.0, 0.8)],  # Bar 3: F F E E
    [(62, 1.0, 0.8), (62, 1.0, 0.8), (60, 2.0, 0.8)],                    # Bar 4: D D C
    [(60, 1.0, 0.8), (60, 1.0, 0.8), (67, 1.0, 0.8), (67, 1.0, 0.8)],  # Bar 5: C C G G (repeat of Bar 1)
    [(69, 1.0, 0.8), (69, 1.0, 0.8), (67, 2.0, 0.8)],                    # Bar 6: A A G (repeat of Bar 2)
    [(65, 1.0, 0.8), (65, 1.0, 0.8), (64, 1.0, 0.8), (64, 1.0, 0.8)],  # Bar 7: F F E E (repeat of Bar 3)
    [(62, 1.0, 0.8), (62, 1.0, 0.8), (60, 2.0, 0.8)]                     # Bar 8: D D C (repeat of Bar 4)
]
# Chord progression should be: [C -> G7 -> C -> G7 -> C -> G7 -> C -> G7]

# Itsy Bitsy Spider is C Major with rests for testing
itsy_bitsy_spider_score = [
    [(60, 0.5, 0.8), (64, 0.5, 0.8), (64, 1.0, 0.8), (64, 0.5, 0.8), (65, 0.5, 0.8), (64, 1.0, 0.8)],  # Bar 1: C E E E F E (itsy bitsy spider)
    [(62, 0.5, 0.8), (60, 0.5, 0.8), (60, 1.0, 0.8), (None, 1.0, 0.8), (60, 1.0, 0.8)],                # Bar 2: D C C [rest] C (went up the)
    [(64, 0.5, 0.8), (64, 0.5, 0.8), (64, 1.0, 0.8), (65, 0.5, 0.8), (67, 0.5, 0.8), (67, 1.0, 0.8)],  # Bar 3: E E E F G G (water spout)
    [(None, 0.5, 0.8), (67, 0.5, 0.8), (65, 0.5, 0.8), (64, 0.5, 0.8), (62, 1.0, 0.8), (60, 1.0, 0.8)],# Bar 4: [rest] G F E D C (down came the)
    [(62, 0.5, 0.8), (64, 0.5, 0.8), (65, 1.0, 0.8), (64, 0.5, 0.8), (62, 0.5, 0.8), (60, 1.0, 0.8)],  # Bar 5: D E F E D C (rain and washed the)
    [(60, 0.5, 0.8), (64, 0.5, 0.8), (64, 1.0, 0.8), (64, 0.5, 0.8), (65, 0.5, 0.8), (64, 1.0, 0.8)],  # Bar 6: C E E E F E (spider out)
    [(62, 0.5, 0.8), (60, 0.5, 0.8), (60, 1.0, 0.8), (None, 1.0, 0.8), (67, 1.0, 0.8)],                # Bar 7: D C C [rest] G (out came the)
    [(67, 0.5, 0.8), (65, 0.5, 0.8), (64, 0.5, 0.8), (62, 0.5, 0.8), (60, 2.0, 0.8)]                    # Bar 8: G F E D C (sun and dried up)
]
# Chord progression should be: [C → F → C → G7 → C → G7 → C → G7 → C]

reward_twinkle = env._compute_reward(musical_score=twinkle_twinkle_score)
print("Reward for Twinkle Twinkle:", reward_twinkle)

print("\n" + "="*70 + "\n")

reward_itsy = env._compute_reward(musical_score=itsy_bitsy_spider_score)
print("Reward for Itsy Bitsy Spider:", reward_itsy)