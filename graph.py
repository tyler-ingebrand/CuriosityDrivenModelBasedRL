import numpy as np
from matplotlib import pyplot as plt

names = ["Known", "Unknown", "Curious-0.1", "Curious-1", "Curious-2", "Curious-5", "Curious-10"]
            # "Curious-50", "Curious-100", "Curious-150", "Curious-200"]

for n in names:
    try:
        with open('data/{}.npy'.format(n), 'rb') as f:
            means = np.load(f)
            stds = np.load(f)
            times = np.load(f)
        plt.fill_between(x=times, y1=means + stds, y2=means - stds, alpha=0.2)
        plt.plot(times, means, label=n)
    except:
        pass


plt.xlabel("Environment Interactions")
plt.ylabel("Test Episode Reward")
plt.title("Comparing MPC")
plt.legend(loc="lower right")
# plt.show()
plt.savefig("data/ComparingMPC.png")
