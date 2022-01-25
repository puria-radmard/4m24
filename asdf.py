import pickle
import matplotlib.pyplot as plt

with open('qB_results_2021-12-19 14:22:43.235307.pkl', 'rb') as handle:
    results = pickle.load(handle)




for r in results:
    plt.plot(r['convergence'], label = min(r['convergence']))
    print([r['acc_rates']])

plt.legend()
plt.show()

import pdb; pdb.set_trace()