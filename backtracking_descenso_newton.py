# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:36:29 2024

@author: UX425
"""

import numpy as np
from numpy import linalg
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


f = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
g = lambda x : np.array([400*x[0]**3-400*x[0]*x[1]+2*x[0]-2,-200*x[0]**2+200*x[1]])
H = lambda x : np.array([[-400*x[1]+1200*x[0]**2 +2,-400*x[0]],[-400*x[0] , 200]])

#######EJERCICIO 1##########

[X,Y]=np.meshgrid(np.linspace(-2,2),np.linspace(-3,6))
con = plt.contour(X,Y,f([X,Y]),np.linspace(0,500,20))
plt.clabel(con)



det = linalg.det(H([1,1]))
print(det)

##La funcion no es convexa pues DET(H([(0,6)]))<0

#######EJERCICIO 2##########

#def backtracking(f,g,xk,pk,alpha=1,rho=0.5,c=1e-4):
    

#######EJERCICIO 3##########

def descenso_mas_rapido(H, g, x_0):
    x = [x_0]
    alpha = [1]
    for i in range(0, 500):
        a_opt = lambda a : f(x[i]-a*g(x[i]))
        alpha_i = backtracking()
        alpha.append(alpha_i)
        xi = x[i]-g(x[i])*alpha_i
        x.append(xi)
    return(f(x[500]))




#######EJERCICIO 4##########

def metodo_Newton_Puro_grafica(H,g):
    [X,Y]=np.meshgrid(np.linspace(-2,2),np.linspace(-3,6))
    plt.contour(X,Y,f([X,Y]),np.linspace(0,500,20))
    x = [[-0.1,-0.3]]
    for i in range(0,6):
        p = linalg.solve(H(x[i]),-g(x[i]))
        xi = x[i] + p
        x.append(xi)
        plt.plot([x[i-1][0], x[i][0]], [x[i-1][1], x[i][1]], '-r')
        plt.plot(x[i][0], x[i][1], 'ok')
    plt.savefig('Pr2.png')
    return(x[5],f(x[5]))

print(metodo_Newton_Puro_grafica(H,g))

#######EJERCICIO 5##########explica a q se debe la lenta convengerncia del método del descenso mas rapido
