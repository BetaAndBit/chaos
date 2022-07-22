from random import Random, randint
from turtle import color
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()




import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.add_patch(plt.Rectangle((0, 0),1, 1,color="black"))
E = 0.05
plt.xlim([0-E, 1+E])
plt.ylim([0-E, 1+E])
plt.show()






import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.add_patch(plt.Rectangle((0, 0),1/3, 1/3,color="black"))
ax.add_patch(plt.Rectangle((0+2/3, 0),1/3, 1/3,color="black"))
ax.add_patch(plt.Rectangle((0, 0+2/3),1/3,1/3,color="black"))
ax.add_patch(plt.Rectangle((0+2/3, 0+2/3),1/3, 1/3,color="black"))
E = 0.05
plt.xlim([0-E, 1+E])
plt.ylim([0-E, 1+E])




import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.add_patch(plt.Rectangle((0, 0),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/9, 0),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0, 0+2/9),1/9,1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/9, 0+2/9),1/9, 1/9,color="black"))

ax.add_patch(plt.Rectangle((0+2/3, 0),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/3+2/9, 0),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/3, 0+2/9),1/9,1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/3+2/9, 0+2/9),1/9, 1/9,color="black"))

ax.add_patch(plt.Rectangle((0, 0+2/3),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/9, 0+2/3),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0, 0+2/3+2/9),1/9,1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/9, 0+2/3+2/9),1/9, 1/9,color="black"))

ax.add_patch(plt.Rectangle((0+2/3, 0+2/3),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/3+2/9, 0+2/3),1/9, 1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/3, 0+2/3+2/9),1/9,1/9,color="black"))
ax.add_patch(plt.Rectangle((0+2/3+2/9, 0+2/3+2/9),1/9, 1/9,color="black"))
E = 0.05
plt.xlim([0-E, 1+E])
plt.ylim([0-E, 1+E])

plt.show()








import numpy as np
import matplotlib.pyplot as plt

def trojkat(x, y, bok):
    E = np.array([[x+bok*0,y+bok*0], [x+bok*1,y+bok*0], [x+bok*1/2,y+bok*np.sqrt(2)/2], [x+bok*0,y+bok*0]])
    return plt.Polygon(E, facecolor = 'k')






import numpy as np
import matplotlib.pyplot as plt

def function1(x,y):
  return (x/2, y/2)

def function2(x,y):
  return (x/2 + 1/2, y/2)

def function3(x,y):
  return (x/2 + 1/4, y/2 + np.sqrt(3)/4)

functions = [function1,function2,function3]

N = 10000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  function = np.random.choice(functions, p=list(map(lambda x: x/3,[1, 1, 1])))
  x, y = function(x,y)
  x_value.append(x)
  y_value.append(y)

plt.scatter(x_value,y_value, s=0.2, color="black")

plt.show()






import numpy as np
import matplotlib.pyplot as plt

def function1(x,y):
  return (0., 0.16*y)

def function2(x,y):
  return (0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6)

def function3(x,y):
  return (0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6)

def function4(x,y):
  return (-0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44)

functions = [function1,function2,function3,function4]

N = 10000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  function = np.random.choice(functions, p=[0.01, 0.85, 0.07, 0.07])
  x, y = function(x,y)
  x_value.append(x)
  y_value.append(y)

plt.scatter(x_value,y_value, s=0.2, color="black")

plt.show()










import random
import numpy as np
import matplotlib.pyplot as plt

def fu(x,p):
    m1 = np.array([[0, 0], [0, 0.16]])
    f1 = np.array([0, 0])
    m2 = np.array([[0.85, 0.04], [-0.04, 0.85]])
    f2 = np.array([0, 1.6])
    m3 = np.array([[0.20, -0.26], [0.23, 0.22]])
    f3 = np.array([0, 1.6])
    m4 = np.array([[-0.15, 0.28], [0.26, 0.24]])
    f4 = np.array([0, 0.44])
    if   p <= 0.01: \
    return np.dot(m1, x) + f1
    elif p <= 0.86: \
    return np.dot(m2, x) + f2
    elif p <= 0.93: \
    return np.dot(m3, x) + f3
    elif p > 0.93: \
    return np.dot(m4, x) + f4

N = 50000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  x, y = fu(np.array([x,y]),random.random())
  x_value.append(x)
  y_value.append(y)

plt.scatter(x_value,y_value, s=0.2, color="black")









import random
import numpy as np
import matplotlib.pyplot as plt

def gu(x,p):
    m1 = np.array([[-0.4, 0], [0, -0.4]])
    f1 = np.array([-1, 0.1])
    m2 = np.array([[0.76, -0.4], [0.4, 0.76]])
    f2 = np.array([0, 0])
    if   p <= 0.5: \
    return np.dot(m1, x) + f1
    else: \
    return np.dot(m2, x) + f2

N = 50000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  x, y = gu(np.array([x,y]),random.random())
  x_value.append(x)
  y_value.append(y)

plt.scatter(x_value,y_value, s=0.2, color="black")

plt.show()



# Spiral



import random
import numpy as np
import matplotlib.pyplot as plt

def gu(x,p):
    m1 = np.array([[0.25, 0], [0, 0.25]])
    f1 = np.array([0, 0.5])
    m2 = np.array([[.823, -.475], [.475, .823]])
    f2 = np.array([.301, -.172])
    if   p <= 0.073: \
    return np.dot(m1, x) + f1
    else: \
    return np.dot(m2, x) + f2

N = 50000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  x, y = gu(np.array([x,y]),random.random())
  x_value.append(x)
  y_value.append(y)

plt.scatter(x_value,y_value, s=0.2, color="black")

plt.show()












#  McWorter's Pentigree



import random
import numpy as np
import matplotlib.pyplot as plt

def gu(x,p):
    m1 = np.array([[0.309, -0.255], [0.255, 0.309]])
    f1 = np.array([0, 0])
    m2 = np.array([[-0.118, -0.363], [0.363, -0.118]])
    f2 = np.array([0.309, 0.225])
    m3 = np.array([[0.309, 0.225], [-0.225, 0.30]])
    f3 = np.array([0.191, 0.588])
    m4 = np.array([[-0.118, 0.363], [-0.363, -0.118]])
    f4 = np.array([0.500, 0.363])
    m5 = np.array([[0.309, 0.225], [-0.225, 0.309]])
    f5 = np.array([0.382, 0])
    m6 = np.array([[0.309, -0.225], [0.225, 0.309]])
    f6 = np.array([0.691, -0.225])
    if   p <= 0.16: \
    return np.dot(m1, x) + f1
    elif p <= 0.32: \
    return np.dot(m2, x) + f2
    elif p <= 0.48: \
    return np.dot(m3, x) + f3
    elif p <= 0.64: \
    return np.dot(m4, x) + f4
    elif p <= 0.80: \
    return np.dot(m5, x) + f5
    elif p > 0.80: \
    return np.dot(m6, x) + f6
    
N = 50000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  x, y = gu(np.array([x,y]),random.random())
  x_value.append(x)
  y_value.append(y)

plt.scatter(x_value,y_value, s=0.2, color="black")

plt.show()










#  Sierp



import random
import numpy as np
import matplotlib.pyplot as plt

def gu(x,p):
    m1 = np.array([[0.5, 0], [0, 0.5]])
    f1 = np.array([0, 0])
    m2 = np.array([[0.5, 0], [0, 0.5]])
    f2 = np.array([0.5, 0])
    m3 = np.array([[0.5, 0], [0, 0.5]])
    f3 = np.array([0, 0.5])
    if   p <= 0.33: \
    return np.dot(m1, x) + f1
    elif p <= 0.66: \
    return np.dot(m2, x) + f2
    elif p > 0.66: \
    return np.dot(m3, x) + f3
    
N = 50000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  x, y = gu(np.array([x,y]),random.random())
  x_value.append(x)
  y_value.append(y)

plt.scatter(x_value,y_value, s=0.2, color="black")

plt.show()





#  Sierp2



import random
import numpy as np
import matplotlib.pyplot as plt

def gu(x,p):
    m1 = np.array([[0.5, 0], [0, 0.5]])
    f1 = np.array([0, 0.5])
    m2 = np.array([[0.5, 0], [0, 0.5]])
    f2 = np.array([0.5, 0])
    m3 = np.array([[0, 0.5], [0.5, 0]])
    f3 = np.array([0,.5 0.5])
    if   p <= 0.33: \
    return np.dot(m1, x) + f1
    elif p <= 0.66: \
    return np.dot(m2, x) + f2
    elif p > 0.66: \
    return np.dot(m3, x) + f3
    
N = 50000
x, y = 0, 0
x_value = []
y_value = []

for i in range(N):
  x, y = gu(np.array([x,y]),random.random())
  x_value.append(x)
  y_value.append(y)



plt.scatter(x_value,y_value, s=0.2, color="black")

plt.show()











# Spirala - transformacje


import random
import math
import numpy as np
import matplotlib.pyplot as plt


def shift(x,p):
    return x + p

def scale(x,d):
    return np.multiply(x,d)

def rotate(x,alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

def gu(x,p):
    if   p <= 0.9: \
    return scale(rotate(x,  -math.pi * 55 / 180), [0.9, 0.9])
    else: \
    return shift(scale(x, [0.4, 0.1]), [1, 0])

N = 50000
x, y = 0, 0
x_value = []
y_value = []
c_value = []

for i in range(N):
  c = random.random()
  x, y = gu(np.array([x,y]),c)
  x_value.append(x)
  y_value.append(y)
  if   c <= 0.9: \
   c_value.append("red")
  else: \
   c_value.append("black")

plt.scatter(x_value,y_value, s=0.2, c=c_value, alpha=0.2)

plt.show()






# Spirala - klasycznie

import random
import math
import numpy as np
import matplotlib.pyplot as plt


def shift(x,p):
    return x + p

def scale(x,d):
    return np.multiply(x,d)

def rotate(x,alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

def spirala(x,depth):
    res = []
    if depth > 1: 
        x1 = scale(rotate(x,  -math.pi * 20 / 180), [0.9, 0.9])
        res1 = spirala(x1,depth - 1)
        x2 = shift(scale(x, [0.4, 0.1]), [1, 0])
        res2 = spirala(x2,depth - 1)
    else:
        plt.plot(x[0],x[1], marker='o', color = "black", markersize=3)

fig = plt.figure(figsize=(8, 6), dpi=80)
spirala([0,0],14)

plt.show()




# Trojkat - klasycznie

# pyplot - biblioteka do rysowania
import matplotlib.pyplot as plt
# np - operacje na wektorach i macierzach
import numpy as np

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# trojkat Sierpinskiego o głębokości depth
def trojkat(x, depth):
    if depth > 1: 
        x1 = scale(shift(x,  [0, 0]), [0.5, 0.5])
        trojkat(x1,depth - 1)
        x2 = scale(shift(x,  [0.5, 0]), [0.5, 0.5])
        trojkat(x2,depth - 1)
        x3 = scale(shift(x,  [0.25, 0.5]), [0.5, 0.5])
        trojkat(x3,depth - 1)
    else:
        plt.plot(x[0],x[1], marker='o', color = "black", markersize=3)

# inicjacja rysunku i narysowanie trójkąta Sierpińskiego
plt.figure(figsize=(8, 6), dpi=80)
trojkat([0,0], depth = 8)
plt.show()










import matplotlib.pyplot as plt
import numpy as np

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# trojkat Sierpinskiego o głębokości depth
def sierpinski(x, depth):
    if depth > 1: 
        x1 = scale(shift(x,  [0, 0]), [0.5, 0.5])
        sierpinski(x1, depth - 1)
        x2 = scale(shift(x,  [0.5, 0]), [0.5, 0.5])
        sierpinski(x2, depth - 1)
        x3 = scale(shift(x,  [0.25, 0.5]), [0.5, 0.5])
        sierpinski(x3, depth - 1)
    else:
        plt.plot(x[0],x[1], marker='o', color = "black", markersize=3)

# inicjacja rysunku i narysowanie trójkąta Sierpińskiego
plt.figure()
sierpinski([0,0], depth = 7)
plt.show()











# prosty T Sierpińskiego


# pyplot - biblioteka do rysowania
import matplotlib.pyplot as plt
# np - operacje na wektorach i macierzach
import numpy as np
# math - operacje matematyczne
import math

def triangle(x, scale):
    plt.fill([x[0], x[0]+scale, x[0]+scale/2], [x[1],x[1],x[1]+scale*math.sqrt(3)/2], color = "black")

# trojkat Sierpinskiego o boku scale i głębokości depth
def sierpinski(x, scale, depth):
    if depth > 1: 
        sierpinski(x, scale / 2, depth - 1)
        sierpinski(np.add(x, [scale/2, 0]), scale / 2, depth - 1)
        sierpinski(np.add(x, [scale/4, scale*math.sqrt(3)/4]), scale / 2, depth - 1)
    else:
        triangle(x, scale)

# inicjacja rysunku i narysowanie trójkąta Sierpińskiego
plt.figure()
sierpinski([0,0], scale = 1, depth = 4)
plt.show()







# prosty Dywan Sierpińskiego


# pyplot - biblioteka do rysowania
import matplotlib.pyplot as plt
# np - operacje na wektorach i macierzach
import numpy as np

def square(x, scale):
    plt.fill(np.add(x[0], [0, scale, scale, 0]), np.add(x[1], [0, 0, scale, scale]), color = "black")

# dywan Sierpinskiego o boku scale i głębokości depth
def carpet(x, scale, depth):
    if depth > 1: 
        carpet(x, scale / 3, depth - 1)
        carpet(np.add(x, [scale/3, 0]), scale / 3, depth - 1)
        carpet(np.add(x, [2*scale/3, 0]), scale / 3, depth - 1)
        carpet(np.add(x, [0, scale/3]), scale / 3, depth - 1)
        carpet(np.add(x, [2*scale/3, scale/3]), scale / 3, depth - 1)
        carpet(np.add(x, [0, 2*scale/3]), scale / 3, depth - 1)
        carpet(np.add(x, [scale/3, 2*scale/3]), scale / 3, depth - 1)
        carpet(np.add(x, [2*scale/3, 2*scale/3]), scale / 3, depth - 1)
    else:
        square(x, scale)

# inicjacja rysunku i narysowanie trójkąta Sierpińskiego
plt.figure()
carpet([0,0], scale = 1, depth = 4)
plt.show()







# prosty Kurz Cantora


# pyplot - biblioteka do rysowania
import matplotlib.pyplot as plt
# np - operacje na wektorach i macierzach
import numpy as np

# dykurz Cantora o boku scale i głębokości depth
def dust(x, scale, depth):
    if depth > 1: 
        dust(x, scale / 3, depth - 1)
        dust(x + scale*2/3, scale / 3, depth - 1)
    else:
        plt.plot([x, x+scale], [0,0], color = "black")

# inicjacja rysunku i narysowanie trójkąta Sierpińskiego
plt.figure()
dust(0, scale = 1, depth = 5)
plt.show()






import math
import numpy as np
import matplotlib.pyplot as plt

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# obrót wokół punktu 0,0 o kąt alpha
def rotate(x, alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

# rysujemy spiralę
def spiral(x,depth):
    res = []
    if depth > 1: 
        x1 = scale(rotate(x, -math.pi * 20 / 180), [0.9, 0.9])
        spiral(x1, depth - 1)
        x2 = shift(scale(x, [0.4, 0.4]), [1, 0])
        spiral(x2, depth - 1)
    else:
        plt.plot(x[0],x[1], marker='o', color = "black", markersize=3)

fig = plt.figure()
spiral([0,0], depth = 18)
plt.show()








import math
import numpy as np
import matplotlib.pyplot as plt

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# obrót wokół punktu 0,0 o kąt alpha
def rotate(x, alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

# rysujemy paproć
def fern(x,depth):
    if depth > 1: 
        x1 = shift(scale(rotate(x, -math.pi * 10 / 180), [0.3, 0.3]), [0, 0])
        fern(x1, depth - 1)
        x2 = shift(scale(rotate(x, math.pi * 15 / 180), [0.25, 0.25]), [0, 0])
        fern(x2, depth - 1)
        x3 = shift(scale(rotate(x, -math.pi * 1 / 180), [0.9, 0.9]), [0.01, 0])
        fern(x3, depth - 1)
    else:
        plt.plot(x[0],x[1], marker='o', color = "black", markersize=3)

fig = plt.figure()
fern([0,0], depth = 13)
plt.show()






# Random



import numpy as np
import matplotlib.pyplot as plt

# transformacje składające się na paproć
def trans1(x,y):
  return (0., 0.16*y)

def trans2(x,y):
  return (0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6)

def trans3(x,y):
  return (0.2*x - 0.26*y, 0.23*x + 0.22*y + 0.8)

def trans4(x,y):
  return (-0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44)

# lista funkcji
transformations = [trans1,trans2,trans3,trans4]
# prawdopodobieństwa wylosowania poszczególnych funkcji
ps = [0.01, 0.79, 0.1, 0.1]

# rozpocznij symulacje
N = 200000
x, y = 0, 0
x_vec = []
y_vec = []

for i in range(N):
  transformation = np.random.choice(transformations, p=ps)
  x, y = transformation(x,y)
  x_vec.append(x)
  y_vec.append(y)

plt.scatter(y_vec, x_vec, s=0.2, color="black")


def trans0(x,y):
  return (x, y)
transformations.append(trans0)

a_vec = []
b_vec = []
for ind in range(0,4):
  trans = transformations[ind]
  a_vec = []
  b_vec = []
  for i, j in [[-2.2,0], [-2.2,10], [2.8,10], [2.8,0], [-2.2, 0]]:
    a, b = trans(i, j)
    a_vec.append(a)
    b_vec.append(b)
  plt.plot(b_vec, a_vec, color = "red")

plt.show()







# Drzewo

import numpy as np
import matplotlib.pyplot as plt

# transformacje składające się na drzewo
def trans1(x,y):
  return(0.195*x -0.488*y + 0.4431, 0.344*x + 0.443*y + 0.2452)
  
def trans2(x,y):
  return(0.462*x + 0.414*y + 0.2511, -0.252*x + 0.361*y + 0.5692)
  
def trans3(x,y):
  return(-0.637*x + 0.8562, 0.501*y + 0.2512)
 
def trans4(x,y):
  return(-0.035*x + 0.07*y + 0.4884, -0.469*x + 0.022*y + 0.5069)
 
def trans5(x,y):
  return(-0.058*x -0.07*y + 0.5976, 0.453*x -0.111*y + 0.0969)
  
# lista funkcji
transformations = [trans1,trans2,trans3,trans4, trans5]
# prawdopodobieństwa wylosowania poszczególnych funkcji
ps = [0.2, 0.2, 0.2, 0.2, 0.2]

# rozpocznij symulacje
N = 200000
x, y = 0, 0
x_vec = []
y_vec = []

for i in range(N):
  transformation = np.random.choice(transformations, p=ps)
  x, y = transformation(x,y)
  x_vec.append(x)
  y_vec.append(y)

plt.scatter(x_vec, y_vec, s=0.2, color="black")

plt.show()

def trans0(x,y):
  return (x, y)
transformations.append(trans0)

a_vec = []
b_vec = []
for ind in range(0,4):
  trans = transformations[ind]
  a_vec = []
  b_vec = []
  for i, j in [[0,0], [0,1], [1,1], [1,0], [0, 0]]:
    a, b = trans(i, j)
    a_vec.append(a)
    b_vec.append(b)
  plt.plot(a_vec, b_vec, color = "red")

plt.show()











## Heighway
## https://larryriddle.agnesscott.org/ifs/heighway/heighway.htm


import math
import numpy as np
import matplotlib.pyplot as plt

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# obrót wokół punktu 0,0 o kąt alpha
def rotate(x, alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

t = 0.5

# rysujemy spiralę
def heighway(x,depth,color = "black"):
    if depth > 1: 
        x1 = scale(rotate(x, -math.pi * 45 / 180), [math.sqrt(0.5), math.sqrt(0.5)])
        heighway(x1, depth - 1, color = "blue")
        x2 = shift(scale(rotate(x, -math.pi * (135 - 180*t) / 180), [math.sqrt(0.5), math.sqrt(0.5)]), [1 - 0.5*t, 0.5*t])
        heighway(x2, depth - 1, color = "red")
    else:
        plt.plot(x[0],x[1], marker='o', color = color, markersize=3)

fig = plt.figure()
heighway([0,0], depth = 14)
plt.show()





import math
import numpy as np
import matplotlib.pyplot as plt

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# obrót wokół punktu 0,0 o kąt alpha
def rotate(x, alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

r = 0.7
theta = 45

# rysujemy spiralę
def sbt(x,depth,color = "black"):
    if depth > 1: 
        x1 = shift(scale(rotate(x, -math.pi * theta / 180), [r, r]), [0, 1])
        sbt(x1, depth - 1, color = "blue")
        x2 = shift(scale(rotate(x, math.pi * theta / 180), [r, r]), [0, 1])
        sbt(x2, depth - 1, color = "red")
        x3 = x
        sbt(x3, depth - 1, color = "black")
    else:
        plt.plot(x[0],x[1], marker='o', color = color, markersize=3)

fig = plt.figure()
sbt([0,0], depth = 12)
plt.show()








import math
import numpy as np
import matplotlib.pyplot as plt

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# obrót wokół punktu 0,0 o kąt alpha
def rotate(x, alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

# rysujemy pentigree
def pentigree(x,depth, color = "black"):
    if depth > 1: 
        x1 = scale(rotate(x, math.pi * 36 / 180), [0.381966, 0.381966])
        pentigree(x1, depth - 1, color="pink")
        x2 = shift(scale(rotate(x, math.pi * 108 / 180), [0.381966, 0.381966]), [0.309, 0.225])
        pentigree(x2, depth - 1, color="orange")
        x3 = shift(scale(rotate(x, -math.pi * 36 / 180), [0.381966, 0.381966]), [0.191, 0.588])
        pentigree(x3, depth - 1, color="black")
        x4 = shift(scale(rotate(x, -math.pi * 108 / 180), [0.381966, 0.381966]), [0.500, 0.363])
        pentigree(x4, depth - 1, color="blue")
        x5 = shift(scale(rotate(x, -math.pi * 36 / 180), [0.381966, 0.381966]), [0.382, 0])
        pentigree(x5, depth - 1, color="green")
        x6 = shift(scale(rotate(x, math.pi * 36 / 180), [0.381966, 0.381966]), [0.691, -0.225])
        pentigree(x6, depth - 1, color="red")
    else:
        plt.plot(x[0],x[1], marker='o', color = color, markersize=3)

fig = plt.figure()
pentigree([0,0], depth = 6)
plt.show()







import numpy as np
import matplotlib.pyplot as plt

def transform(x,y, affine):
  return(affine[0]*x + affine[1]*y + affine[2], affine[3]*x + affine[4]*y + affine[5])

# lista funkcji
affines = [[0.14, 0.01, -0.08, 0.0, 0.51, -1.31],
 [0.43, 0.52, 1.49, -0.45, 0.5, -0.75],
 [0.45, -0.49, -1.62, 0.47, 0.47, -0.74],
 [0.49, 0.0, 0.02, 0.0, 0.51, 1.62]]

# prawdopodobieństwa wylosowania poszczególnych funkcji
ps = [0.25, 0.25, 0.25, 0.25]

# rozpocznij symulacje
N = 500000
x, y = 0, 0
x_vec = []
y_vec = []
trans_vec = []

for i in range(N):
  trans = np.random.choice(range(0,4), p=ps)
  affine = affines[trans]
  x, y = transform(x,y, affine)
  x_vec.append(x)
  y_vec.append(y)
  trans_vec.append(["black", "blue", "green", "red"][trans])

plt.scatter(x_vec, y_vec, s=0.2, color=trans_vec)
plt.show()




affines = [[0.787879, -0.424242, 1.758647, 0.242424, 0.859848, 1.408065],
[-0.121212, 0.257576, -6.721654, 0.151515, 0.05303, 1.377236],
[0.181818, -0.136364, 6.086107, 0.090909, 0.181818, 1.568035]]

ps = [0.9, 0.05, 0.05]


# rozpocznij symulacje
N = 500000
x, y = 0, 0
x_vec = []
y_vec = []
trans_vec = []

for i in range(N):
  trans = np.random.choice(range(0,7), p=ps)
  affine = affines[trans]
  x, y = transform(x,y, affine)
  x_vec.append(x)
  y_vec.append(y)
  trans_vec.append(["black", "blue", "green", "red", "orange", "pink", "brown"][trans])

plt.scatter(x_vec, y_vec, s=0.2, color=trans_vec)
plt.show()




affines = [[0.202, -0.805, -0.373, -0.689, -0.342, -0.653],
[0.138, 0.665, 0.66, -0.502, -0.222, -0.277]]

ps = [0.5, 0.5, 0]

affines = [[0.05, 0.0, -0.06, 0.0, 0.4, -0.47],
[-0.05, 0.0, -0.06, 0.0, -0.4, -0.47],
[0.03, -0.14, -0.16, 0.0, 0.26, -0.01],
[-0.03, 0.14, -0.16, 0.0, -0.26, -0.01],
[0.56, 0.44, 0.3, -0.37, 0.51, 0.15],
[0.19, 0.07, -0.2, -0.1, 0.15, 0.28],
[-0.33, -0.34, -0.54, -0.33, 0.34, 0.39]]

ps = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]








## gra w chaos na trzy punkty

import math
import numpy as np
import matplotlib.pyplot as plt

triangle = [[0,0], [1, 2], [2,1]]
point = [0.1, 0]

# rozpocznij symulacje
N = 200000
x_vec = []
y_vec = []

for i in range(N):
  ind = np.random.choice(range(0,3))
  point = np.add(triangle[ind], np.multiply(np.subtract(point, triangle[ind]), 0.5))
  x_vec.append(point[0])
  y_vec.append(point[1])

plt.scatter(x_vec, y_vec, s=0.2)
plt.show()









import math
import numpy as np
import matplotlib.pyplot as plt

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# rysujemy pentagon
def pentagon(x, depth, color = "black"):
    if depth > 1: 
        x1 = shift(scale(x, [0.382, 0.382]), [0, 0])
        pentagon(x1, depth - 1, color = "red")
        x2 = shift(scale(x, [0.382, 0.382]), [0.618, 0])
        pentagon(x2, depth - 1, color = "green")
        x3 = shift(scale(x, [0.382, 0.382]), [0.809, 0.588])
        pentagon(x3, depth - 1, color = "blue")
        x4 = shift(scale(x, [0.382, 0.382]), [0.309, 0.951])
        pentagon(x4, depth - 1, color = "black")
        x5 = shift(scale(x, [0.382, 0.382]), [-0.191, 0.588])
        pentagon(x5, depth - 1, color = "orange")
    else:
        plt.plot(x[0],x[1], marker='o', color = color, markersize=3)

fig = plt.figure()
pentagon([0,0], depth = 6)
plt.show()









import math
import numpy as np
import matplotlib.pyplot as plt

# przesunięcie punktu x o delta
def shift(x, delta):
    return np.add(x, delta)

# przeskalowanie punktu x razy ratio
def scale(x, ratio):
    return np.multiply(x, ratio)

# obrót wokół punktu 0,0 o kąt alpha
def rotate(x, alpha):
    return np.matmul(x,[[math.cos(alpha), -math.sin(alpha)],[math.sin(alpha), math.cos(alpha)]])

# rysujemy paproć
def fern(x, depth, color = "black"):
    if depth > 1: 
        x1 = shift(scale(rotate(x, -math.pi * 10 / 180), [0.5, 0.3]), [0, 0])
        fern(x1, depth - 1, color = "red")
        x2 = shift(scale(rotate(x, math.pi * 15 / 180), [0.45, 0.25]), [0, 0])
        fern(x2, depth - 1, color = "blue")
        x3 = shift(scale(rotate(x, -math.pi * 1 / 180), [0.9, 0.9]), [0.01, 0])
        fern(x3, depth - 1, color = "black")
        x4 = scale(rotate(x, -math.pi * 1 / 180), [0.25, 0.01])
        fern(x4, depth - 1, color = "orange")
    else:
        plt.plot(x[0],x[1], marker='o', color = color, markersize=3)

fig = plt.figure()
fern([0,0], depth = 11)
plt.show()



