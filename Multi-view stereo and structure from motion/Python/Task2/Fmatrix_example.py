import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from utils import draw_eplines, vgg_F_from_P
from estimateF import estimateF
from estimateFnorm import estimateFnorm



# The given image coordinates were originally localized manually.
# That is, 11 points (A,B,C,D,E,F,G,H,L,M,N) are marked from both images.
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'l', 'm', 'n']

x1 = 1.0e+03 * np.array([0.7435, 3.3315, 0.8275, 3.2835, 0.5475, 3.9875,
                         0.6715, 3.8835, 1.3715, 1.8675, 1.3835])
                      
y1 = 1.0e+03 * np.array([0.4455, 0.4335, 1.7215, 1.5615, 0.3895, 0.3895, 
                         2.1415, 1.8735, 1.0775, 1.0575, 1.4415])

x2 = 1.0e+03 * np.array([0.5835, 3.2515, 0.6515, 3.1995, 0.1275, 3.7475, 
                         0.2475, 3.6635, 1.1555, 1.6595, 1.1755])

y2 = 1.0e+03 * np.array([0.4135, 0.4015, 1.6655, 1.5975, 0.3215, 0.3135, 
                         2.0295, 1.9335, 1.0335, 1.0255, 1.3975])

# Load images and their corresponding camera matrices
im1 = np.array(Image.open('im1.jpg'))
im2 = np.array(Image.open('im2.jpg'))
P1 = loadmat('P1.mat')['P1']
P2 = loadmat('P2.mat')['P2']

# The fundamental matrix F can be computed from the projection matrices if they are known
FfromPs = vgg_F_from_P(P1, P2)

# Implement the 8 point and the normalized 8 point methods in estimateF.py and estimateFnorm.py
# for F-matrix estimation
F = estimateF(np.vstack((x1, y1, np.ones(11))), np.vstack((x2, y2, np.ones(11))))
Fnorm = estimateFnorm(np.vstack((x1, y1, np.ones(11))), np.vstack((x2, y2, np.ones(11))))

# Visualize
plt.figure(1)
plt.subplot(2, 1, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(im1)
plt.plot(x1, y1, 'c+', markersize=10)
for i in range(np.size(x1)):
    plt.annotate(labels[i], (x1[i], y1[i]), color='c', fontsize=15)
plt.subplot(2, 1, 2)
plt.title('Cyan: Projection matrices,  Magenta: 8-point,  Yellow: Normalized 8-point')
plt.xticks([])
plt.yticks([])
plt.imshow(im2)
plt.plot(x2, y2, 'c+', markersize=10)
for i in range(np.size(x1)):
    plt.annotate(labels[i], (x2[i], y2[i]), color='c', fontsize=15)

# Draw epipolar lines
# Fx is the epipolar line associated with x, (l'= Fx)
# a = eplines[0,i]
# b = eplines[1,i]
# c = eplines[2,i]
# ax+by+c=0
eplines = np.dot(FfromPs, np.vstack((x1, y1, np.ones(11))))
draw_eplines(eplines, im2, 'c-')

if F is not None:
    eplinesA = np.dot(F, np.vstack((x1, y1, np.ones(11))))
    draw_eplines(eplinesA, im2, 'm-')

    eplinesB = np.dot(Fnorm, np.vstack((x1, y1, np.ones(11))))
    draw_eplines(eplinesB, im2, 'y-')

plt.show()
