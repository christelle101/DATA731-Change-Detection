import matplotlib.pyplot as plt
import math
import scipy.io
import numpy as np
import matplotlib.mlab as mlab

from scipy.stats import norm
from scipy.integrate import quad

mat = scipy.io.loadmat('X_pluv.mat')
#print(mat)

matrix = mat["X_pluv"]

def p(x):
    return norm.pdf(x, np.mean(matrix[0]), math.sqrt(1.05935))

def q(x):
    return norm.pdf(x, np.mean(matrix[1]), math.sqrt(1.04411))

def KL(x):
    return p(x) * np.log( p(x) / q(x) )

range = np.arange(-10, 10, 0.001)

KL_int, err = quad(KL, -10, 10) 
#print( 'KL: ', KL_int )

fig = plt.figure(figsize=(18, 8), dpi=100)

#---------- First Plot

ax = fig.add_subplot(1,2,1)
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.set_xlim(0,12)


ax.text(-2.5, 0.17, 'p(x)', horizontalalignment='center',fontsize=17)
ax.text(4.5, 0.17, 'q(x)', horizontalalignment='center',fontsize=17)

plt.plot(range, p(range))
plt.plot(range, q(range))

#---------- Second Plot

ax = fig.add_subplot(1,2,2)
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.set_xlim(-10,10)
ax.set_ylim(-0.001,0.001)

ax.text(3.5, 0.0005, r'$DK_{KL}(p||q)$', horizontalalignment='center',fontsize=17)

ax.plot(range, KL(range))

ax.fill_between(range, 0, KL(range))

plt.savefig('KullbackLeibler.png',bbox_inches='tight')
plt.show()