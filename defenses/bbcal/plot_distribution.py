import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme()

attack = 'hamock'
with open(f"{attack}_activations.pkl", 'rb') as f:
    poison_activation, clean_activation = pickle.load(f)

# breakpoint()
plt.hist(poison_activation, density = True, label = 'Backdoor samples')
plt.hist(clean_activation, density = True, label = 'Clean samples')

plt.legend()
plt.savefig(f'{attack}.png', dpi = 100)