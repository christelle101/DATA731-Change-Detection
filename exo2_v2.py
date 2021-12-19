import scipy.io
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('X_pluv.mat')
matrix = mat['X_pluv']

X1 = matrix[0]
X2 = matrix[1]
X3 = matrix[2]
pas = 200

for X in [X1,X2,X3]:
    entropie_totale = []
    i = 0
    while i + pas*2 != len(X1)+1 :
    
        set_1 = X[i : i + pas-1]
        set_2 = X[i + pas : i + pas*2]
        
        moyenne_set_1 = np.mean(set_1)
        moyenne_set_2 = np.mean(set_2)
        
        variance_set_1 = (np.std(set_1))
        variance_set_2 = (np.std(set_2))
        
        part1 = ((moyenne_set_1 - moyenne_set_2)**2)/(variance_set_1**2 + variance_set_2**2)
        part2 = ((variance_set_1**2)/(variance_set_2**2))+((variance_set_2**2)/(variance_set_1**2))
        
        entropie = (1/2)*part1 + (1/2)*part2 - 1
        
        entropie_totale.append(entropie)
        i += 1
    plt.plot(entropie_totale, label ="entropie totale")

plt.legend()
plt.show()

