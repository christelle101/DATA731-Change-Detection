from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy as sp
from numpy import asarray
import matplotlib.pyplot as plt

"""
    ANALYSE 1
"""
"""#Question 1 - Generer 100 realisations pour N = 100
mu, sigma = 0, 1
s = np.random.normal(mu, sigma, 10000)
print(s)

#Question 2 - Tracé de l'histogramme pour b = 12, b = 24, b = 36
#plt.hist(s, 12)
#plt.hist(s, 24)
plt.hist(s,36)
plt.show()


#Question 4 - Tracé de la ddp
count, bins, ignored = plt.hist(s, 36, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

#compare avec la répartition théorique
#x = np.linspace(-30, 30, 100)
#y = sp.stats.norm.pdf(x, mu, sigma)
#plt.plot(x,y)
#plt.show()
"""

"""
    ANALYSE 2
"""
"""mu_2, sigma_2 = 2, 3
l = np.random.normal(mu_2, sigma_2, 10000)
plt.hist(l, 12)
#plt.hist(l, 24)
#plt.hist(l,36)
plt.show()

count_2, bins_2, ignored_2 = plt.hist(l, 12, density=True)
plt.plot(bins_2, 1/(sigma_2 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins_2 - mu_2)**2 / (2 * sigma_2**2) ),
         linewidth=2, color='r')
plt.show()"""

"""
    ANALYSE 3
"""
#Question 1 
"""On fait varier les valeurs de N entre 100 et 10 000 avec un pas de 50"""
for n in range(100, 10000, 50):
    realisations = np.random.normal(1, 3, n)
    print(realisations)
    
"""N = []
i = 100
imax = 10000
while i<imax:
    N.append(i)
    i = i+50
#print(N)


#Tableau de moyennes et variances empiriques
Emean = []
Evar = []
Moy = []


for m in N:
    datatemp = np.random.normal(mu, sigma, m)
    mutemp = np.mean(datatemp)
    Emean.append(mutemp)
    Moy.append(mu)


plt.plot(N, Emean, label = 'Moyenne empirique')
plt.plot(N,Moy, label = 'Moyenne theorique')
plt.legend()
plt.show()"""
mu_3, sigma_3 = 1, 3
Emean = []
Evar = []
for i in range(100, 10050, 50):
    L = np.random.normal(mu_3, sigma_3, i)
    m = sum(L)/len(L)
    varRes = sum([(i - m)**2 for i in L])/(len(L) - 1)
    Emean.append(m)
    Evar.append(varRes)
print(Emean)
print(Evar)

x = np.linspace(-5, 5, 199)
plt.plot(x, Emean, label = 'Moyenne empirique')
plt.legend()
plt.plot(x, [1 for i in range(199)])
plt.show()

y = np.linspace(-5, 5, 199)
plt.plot(y, Evar, label = 'Variance')
plt.legend()
plt.plot(y, [9 for i in range(199)])
plt.show()
