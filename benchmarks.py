from RLMusicEnv import RLMusicBotEnv
from final_visualization import show_final_sheet
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

# Change to 4 for RandomAgent 
env = RLMusicBotEnv(bars=8, debug=True)

twinkle_twinkle_score = [
    [(60, 1.0), (60, 1.0), (67, 1.0), (67, 1.0)],  # Bar 1: C C G G
    [(69, 1.0), (69, 1.0), (67, 2.0)],              # Bar 2: A A G
    [(65, 1.0), (65, 1.0), (64, 1.0), (64, 1.0)],  # Bar 3: F F E E
    [(62, 1.0), (62, 1.0), (60, 2.0)],              # Bar 4: D D C
    [(60, 1.0), (60, 1.0), (67, 1.0), (67, 1.0)],  # Bar 5: C C G G (repeat of Bar 1)
    [(69, 1.0), (69, 1.0), (67, 2.0)],              # Bar 6: A A G (repeat of Bar 2)
    [(65, 1.0), (65, 1.0), (64, 1.0), (64, 1.0)],  # Bar 7: F F E E (repeat of Bar 3)
    [(62, 1.0), (62, 1.0), (60, 2.0)]               # Bar 8: D D C (repeat of Bar 4)
]
# Chord progression should be: [C -> G7 -> C -> G7 -> C -> G7 -> C -> G7]

# Itsy Bitsy Spider is C Major with rests for testing
itsy_bitsy_spider_score = [
    [(60, 0.5), (64, 0.5), (64, 1.0), (64, 0.5), (65, 0.5), (64, 1.0)],  # Bar 1: C E E E F E (itsy bitsy spider)
    [(62, 0.5), (60, 0.5), (60, 1.0), (None, 1.0), (60, 1.0)],             # Bar 2: D C C [rest] C (went up the)
    [(64, 0.5), (64, 0.5), (64, 1.0), (65, 0.5), (67, 0.5), (67, 1.0)],    # Bar 3: E E E F G G (water spout)
    [(None, 0.5), (67, 0.5), (65, 0.5), (64, 0.5), (62, 1.0), (60, 1.0)],   # Bar 4: [rest] G F E D C (down came the)
    [(62, 0.5), (64, 0.5), (65, 1.0), (64, 0.5), (62, 0.5), (60, 1.0)],    # Bar 5: D E F E D C (rain and washed the)
    [(60, 0.5), (64, 0.5), (64, 1.0), (64, 0.5), (65, 0.5), (64, 1.0)],    # Bar 6: C E E E F E (spider out)
    [(62, 0.5), (60, 0.5), (60, 1.0), (None, 1.0), (67, 1.0)],             # Bar 7: D C C [rest] G (out came the)
    [(67, 0.5), (65, 0.5), (64, 0.5), (62, 0.5), (60, 2.0)]                 # Bar 8: G F E D C (sun and dried up)
]
# Chord progression should be: [C → F → C → G7 → C → G7 → C → G7 → C]

# Ode to Joy in C Major is technically more complex than a nursery rhyme. Tests harmony, repeated motifs, and long note phrasing all at once
ode_to_joy_score = [
    [(64, 1.0), (64, 1.0), (65, 1.0), (67, 1.0)],  # Bar 1: E E F G
    [(67, 1.0), (65, 1.0), (64, 1.0), (62, 1.0)],  # Bar 2: G F E D
    [(60, 1.0), (60, 1.0), (62, 1.0), (64, 1.0)],  # Bar 3: C C D E
    [(64, 1.5), (62, 0.5), (62, 2.0)],             # Bar 4: E (long), D (short), D (long)
    [(64, 1.0), (64, 1.0), (65, 1.0), (67, 1.0)],  # Bar 5: E E F G (repeat of Bar 1)
    [(67, 1.0), (65, 1.0), (64, 1.0), (62, 1.0)],  # Bar 6: G F E D (repeat of Bar 2)
    [(60, 1.0), (60, 1.0), (62, 1.0), (64, 1.0)],  # Bar 7: C C D E (repeat of Bar 3)
    [(62, 1.5), (60, 0.5), (60, 2.0)]              # Bar 8: D (long), C (short), C (long)
]
# Chord progression should be: [C -> G7 -> C → G7 -> C -> G7 -> C -> G7]

# Happy Birthday in C Major (converted to 4 4) tests leaps, pickup note, and phrase variation
happy_birthday_score = [
    [(67, 1.0), (67, 1.0), (69, 1.0), (67, 1.0)],  # Bar 1: G G A G
    [(72, 2.0), (71, 2.0)],                        # Bar 2: C (2 beats), B (2 beats)
    [(67, 1.0), (67, 1.0), (69, 1.0), (67, 1.0)],  # Bar 3: G G A G
    [(74, 2.0), (72, 2.0)],                        # Bar 4: D (2 beats), C (2 beats)
    [(67, 1.0), (67, 1.0), (67, 1.0), (76, 1.0)],  # Bar 5: G G G E
    [(72, 1.0), (71, 1.0), (69, 2.0)],             # Bar 6: C B A (A held 2 beats)
    [(65, 1.0), (65, 1.0), (64, 1.0), (72, 1.0)],  # Bar 7: F F E C
    [(74, 1.0), (72, 1.0), (72, 2.0)]              # Bar 8: D C C (final C held)
]
# Chord progression should be: [G -> C -> G -> C -> G -> C -> F -> C]


reward_twinkle = env._compute_reward(musical_score=twinkle_twinkle_score)
print("Reward for Twinkle Twinkle:", reward_twinkle)

print("\n" + "="*70 + "\n")

reward_itsy = env._compute_reward(musical_score=itsy_bitsy_spider_score)
print("Reward for Itsy Bitsy Spider:", reward_itsy)

print("\n" + "="*70 + "\n")

reward_ode = env._compute_reward(musical_score=ode_to_joy_score)
print("Reward for Ode to Joy:", reward_ode)

print("\n" + "="*70 + "\n")

reward_birthday = env._compute_reward(musical_score=happy_birthday_score)
print("Reward for Happy Birthday:", reward_birthday)
