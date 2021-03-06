import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")


if args.mode == 'train':
	s='Training Data: Histogram of final values for each episode'
else:
	s='Testing Data: Histogram of final values for each episode'

plt.hist(a, bins=20)
plt.title(s)
plt.xlabel('Values in dollars')
plt.ylabel('Number of episodes')
plt.show()
