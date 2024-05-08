# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 08:36:28 2024

@author: UX425
"""

import numpy as np
import matplotlib.pyplot as plt

###########EJERCICIO 1#######

f = lambda x: (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2


###########EJERCICIO 2#######

def MN(f, x0, r=1, itmax=1000, tol=1e-7):
    n = len(x0)
    X = r * np.identity(n)
    for i in range(n):
        X[i] += x0
    X = np.vstack([x0, X])
    X = X.T

    saveX = np.zeros((itmax + 1, n, n + 1))  

    v = []
    for i in range(n + 1):
        v.append(f(X[:, i]))
    fX = np.array(v)
    orden = np.argsort(fX)

    xmin = X[:, orden[0]]
    fxmin = fX[orden[0]]
    xmax = X[:, orden[n]]
    fxmax = fX[orden[n]]

    M = 1 / (n + 1) * np.sum(fX)
    test = (1 / n * np.sum((fX - M) ** 2)) ** 0.5
    it = 0
    saveX[0, :, :] = X

    while it <= itmax and test >= tol:
        xb = 1 / n * (-xmax + np.sum(X, axis=1))
        xref = 2 * xb - xmax
        if fxmin > f(xref):
            xexp = 2 * xref - xb
            if f(xexp) < f(xref):
                xnew = xexp
            else:
                xnew = xref
        elif fxmin <= f(xref) and f(xref) < fX[orden[n - 1]]:
            xnew = xref
        else:
            if f(xmax) <= f(xref):
                xnew = 0.5 * (xmax + xb)
            else:
                xnew = 0.5 * (xref + xb)

        if f(xnew) > f(xmax):
            X = 0.5 * (X.T + xmin).T
            v = []
            for i in range(n + 1):
                v.append(f(X[:, i]))
            fX = np.array(v)
            
        else:
            X[:, orden[n]] = xnew
            fX[orden[n]] = f(xnew)
            
        
        orden = np.argsort(fX)
        fxmax = fX[orden[n]]
        fxmin = fX[orden[0]]

        xmin = X[:, orden[0]]
        xmax = X[:, orden[n]]

        M = 1 / (n + 1) * np.sum(fX)
        test = (1 / n * np.sum((fX - M) ** 2)) ** 0.5
        it += 1
        saveX[it, :, :] = X  
    return xmin,f(xmin),it,saveX[:it+1]
    

    
###########EJERCICIO 3#######

x0=np.array([-0.3, 1.8])
sol = MN(f,x0)  
print(sol)


###########EJERCICIO 4#######

[X,Y]=np.meshgrid(np.linspace(-4.5,4.5),np.linspace(-4.5,4.5))
fig=plt.figure()
plt.contour(X,Y,f([X,Y]),40)
plt.axis('scaled')

saveX = sol[3]

for X in saveX:
    plt.plot(X[0, :], X[1, :], 'r-')

plt.title('Triángulos del algoritmo sobre las curvas de nivel')
plt.show()
    