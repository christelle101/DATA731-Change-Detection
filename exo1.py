import scipy.io
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('X_pluv.mat')
#print(mat)


matrix = mat["X_pluv"]
#print(matrix)

X1 = matrix[0]
X2 = matrix[1]
X3 = matrix[2]

Y1 = (X1,X2)
Y2 = (X2,X3)
Y3 = (X3,X1)

"""
    QUESTION 1 - MATRICES DE COVARIANCE
"""
"""data_y1 = np.array([matrix[0], matrix[1]])
covMatrix_y1 = np.cov(data_y1, bias = True)
#sn.heatmap(covMatrix_y1, annot = True, fmt = 'g')

data_y2 = np.array([matrix[1], matrix[2]])
covMatrix_y2 = np.cov(data_y2, bias = True)
#sn.heatmap(covMatrix_y2, annot = True, fmt = 'g')

data_y3 = np.array([matrix[2], matrix[0]])
covMatrix_y3 = np.cov(data_y3, bias = True)
sn.heatmap(covMatrix_y3, annot = True, fmt = 'g')

c1=np.cov(matrix[0],matrix[1], bias = True)
c2=np.cov(matrix[1],matrix[2], bias = True)
c3=np.cov(matrix[2],matrix[0], bias = True)

#plt.hist2d(matrix[0],matrix[1])
#plt.hist2d(matrix[1],matrix[2])
plt.show()"""


"""
    QUESTION 2 - HISTOGRAMME BIVARIE
"""
"""
# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y = matrix[0], matrix[1]
hist, xedges, yedges = np.histogram2d(x, y)

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()"""

"""
    QUESTION 3 - AFFICHAGE DES DDP THEORIQUES GAUSSIENNES
"""


"""moyenne = np.mean(Y1)
sigma = np.std(Y1)
count,bins,ignored = plt.hist(Y1,51,normed=True)
plt.plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-moyenne)**2/(2*sigma**2)), linewidth=2, color="r")"""


"""moyenne = np.mean(Y2)
sigma = np.std(Y2)
count,bins,ignored = plt.hist(Y2,51,normed=True)
plt.plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-moyenne)**2/(2*sigma**2)), linewidth=2, color="r")"""

moyenne = np.mean(Y3)
sigma = np.std(Y3)
count,bins,ignored = plt.hist(Y3,51,normed=True)
plt.plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-moyenne)**2/(2*sigma**2)), linewidth=2, color="r")


plt.show()

