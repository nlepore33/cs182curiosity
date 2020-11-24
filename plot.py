import numpy as np
import matplotlib.pyplot as plt

runs = [
    'PPO_ICM_MountainCarContinuous-v0_-37.07607583799019.npy',
    'PPO_ICM_MountainCarContinuous-v0_-66.22553486191602_0.9_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-68.08634495153602_0.75_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-76.5178613270446.npy',
    'PPO_ICM_MountainCarContinuous-v0_-97.24413804480204.npy',
    'PPO_MountainCarContinuous-v0_-71.65956278338766.npy',
    'PPO_MountainCarContinuous-v0_-74.31046497716287.npy',
    'PPO_MountainCarContinuous-v0_-110.17387717608476.npy'
]

seed_runs = [
    'PPO_ICM_MountainCarContinuous-v0_-50.71670728565125_0.0_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.2.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.5.npy',
    'PPO_ICM_MountainCarContinuous-v0_-53.10911658906612_1.0_0.9.npy',
    'PPO_MountainCarContinuous-v0_-50.27348071147965.npy'
]

seeded = True

if seeded:
    directory = './results_seeded/'
    savefile = './plots/ppo_vs_ppo_icm_seeded'
    loop_runs = seed_runs
else:
    directory = './results/'
    savefile = './plots/ppo_vs_ppo_icm'
    loop_runs = runs

plot_range = 30

for run in loop_runs:
    y = np.load(directory+run)
    y = y[:plot_range]
    # if run.startswith('PPO_ICM'):
    #     color = 'b'
    # else:
    #     color = 'r'
    plt.plot(np.arange(len(y)), y, label=run)
plt.legend()
plt.savefig(savefile)
plt.show()