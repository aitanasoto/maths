# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:45:31 2024

@author: UX425
"""

import numpy as np
from numpy import linalg


######PROBLEMA 1##########################################
g=lambda x: np.array([x[0]**2+x[1]*x[2]*x[3]-2,
                    x[0]+x[0]*x[1]-x[2]*x[3]-1,
                    -x[0]+x[0]**2+2*x[1]-x[2]**3+x[3]**2-2,
                    x[0]*x[1]*x[2]+x[1]*x[2]*x[3]-2,
                    x[0]*x[1]-x[2]*np.exp(x[3]-1)])

grad_g= lambda x: np.array([[2*x[0],x[2]*x[3],x[1]*x[3],x[2]*x[1]],
                            [1+x[1],x[0],-x[3],-x[2]],
                            [-1+2*x[0],2,-3*x[2]**2,2*x[3]],
                            [x[1]*x[2],x[2]*(x[0]+x[3]),x[1]*(x[0]+x[3]),x[1]*x[3]],
                            [x[1],x[0],-np.exp(x[3]-1),-x[2]*np.exp(x[3]-1)]]).T

f=lambda x: .5*sum(g(x)**2)

grad_f = lambda x: grad_g(x)@g(x)



######PROBLEMA 2##########################################

def Gauss_Newton(x0,f,grad_f,grad_g,tol,N):
    norma = lambda x: (sum(grad_f(x)**2))**.5
    i = 0
    
    while True:
        xk = x0+linalg.solve( grad_g(x0)@grad_g(x0).T ,-grad_f(x0))
        norma_grad_f = norma(xk)
        i+=1
        if norma_grad_f < tol or i == N:
            break
        x0 = xk
        
    return i,xk,f(xk),norma_grad_f

x0 = np.array([-0.2,0.5,0.3,0.3])

gn = Gauss_Newton(x0, f, grad_f, grad_g, 1e-8, 1000) 

print('En el método de Gauss-Newton puro obtenemos: \n',
      '\n Número de iteraciones: ', gn[0],'\n',
      'Solución obtenida: ', gn[1], '\n',
      'Valor de f: ', gn[2],'\n',
      'Norma del gradiente de f: ', gn[3],'\n')



######PROBLEMA 3###########################################


def wolfe(f, grad_f, x, p, c1=1e-4, c2=0.9):
    a = 0
    b = 2
    continua = True
    cota = c2*np.dot(grad_f(x), p)
    fx=f(x)
    
    
    while f(x+b*p) <= fx+c1*b*np.dot(grad_f(x), p):
        b=2*b
        
    while continua == True:
        alpha=.5*(a+b)
        if f(x+alpha*p) > fx+c1*alpha*np.dot(grad_f(x), p):
            b=alpha
        elif np.dot(grad_f(x + alpha*p), p) < cota:
            a=alpha
        else:
            continua=False
    return alpha


def Gauss_Newton_BL(x0,grad_f,grad_g,tol,N):
    norma = lambda x: (sum(grad_f(x)**2))**.5
    i = 0
    
    while True:
        p = linalg.solve(grad_g(x0)@grad_g(x0).T,-grad_f(x0))
        alpha=wolfe(f,grad_f,x0,p)
        xk = x0+alpha*p
        norma_grad_f = norma(xk)
        i+=1
        if norma_grad_f < tol or i == N:
            break
        x0 = xk
        
    return i,xk,f(xk),norma_grad_f

gn_bl = Gauss_Newton_BL(x0, grad_f, grad_g, 1e-8, 1000) 

print('En el método de Gauss-Newton con búsqueda lineal obtenemos: \n',
      '\n Número de iteraciones: ', gn_bl[0],'\n',
      'Solución obtenida: ', gn_bl[1], '\n',
      'Valor de f: ', gn_bl[2],'\n',
      'Norma del gradiente de f: ', gn_bl[3],'\n')


        
######PROBLEMA 4############################################


def Gauss_Newton_DR(x0,grad_f,grad_g,tol,N):
    norma = lambda x: (sum(grad_f(x)**2))**.5
    i = 0
    
    while True:
        p = -grad_f(x0)
        alpha=wolfe(f,grad_f,x0,p)
        xk = x0+alpha*p
        norma_grad_f = norma(xk)
        i+=1
        if norma_grad_f < tol or i == N:
            break
        x0 = xk
        
    return i,xk,f(xk),norma_grad_f
        
Gauss_Newton_DR(x0, grad_f, grad_g, 1e-8, 1000) 

'''Los tres métodos convergen pero si utilizamos la dirrección del método
del descenso más rápido necesitaremos más iteraciones. En los apartados 2 y 3 el método
converge en 13 y 7 iteraciones respectivamente y en el apartado 4 en 165'''



######PROBLEMA 5############################################

#Gauss_Newton(np.array([0,0,0,0]), f, grad_f, grad_g, 1e-8, 1000)
    
'''Utilizando el origen como punto inicial nos da un error porque obtenemos
una matriz singular (grad_g(x0)@grad_g(x0).T) pues el origen lo es'''

def Lebenberg_Marquardt(x0,grad_f,grad_g,tol,N):
    norma = lambda x: (sum(grad_f(x)**2))**.5
    norma2 = lambda x: (sum(grad_f(x)**2))
    I = np.identity(4)
    i = 0
    
    while True:
        p = linalg.solve( grad_g(x0)@grad_g(x0).T + np.dot(norma2(x0),I) ,-grad_f(x0))
        alpha=wolfe(f,grad_f,x0,p)
        xk = x0+alpha*p
        norma_grad_f = norma(xk)
        i+=1
        if norma_grad_f < tol or i == N:
            break
        x0 = xk
        
    return i,xk,f(xk),norma_grad_f

Lebenberg_Marquardt(np.array([0,0,0,0]), grad_f, grad_g, 1e-8, 1000) 



######PROBLEMA 6(OPCIONAL)##################################

g0 = lambda x: x[0]**2+x[1]*x[2]*x[3]-2
g1 = lambda x: x[0] + x[0]*x[1] - x[2]*x[3]- 1 
g2 = lambda x: -x[0]+x[0]**2+2*x[1]-x[2]**3+x[3]**2-2
g3 = lambda x: x[0]*x[1]*x[2]+x[1]*x[2]*x[3]-2
g4 = lambda x: x[0]*x[1]-x[2]*np.exp(x[3]-1)

H0 = lambda x: np.array([[2,0,0,0],[0,0,x[3],x[2]],[0,x[3],0,x[1]],[0,x[2],x[1],0]])
H1 = lambda x: np.array([[0,1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,-1,0]])
H2 = lambda x: np.array([[2,0,0,0],[0,0,0,0],[0,0,-6*x[2],0],[0,0,0,2]])
H3 = lambda x: np.array([[0,x[2],x[1],0],[x[2],0,x[0]+x[3],x[2]],[x[1],x[0]+x[3],0,x[1]],[0,x[2],x[1],0]])
H4 = lambda x: np.array([[0,1,0,0],[1,0,0,0],[0,0,0,-1*np.exp(x[3]-1)],[0,0,-1*np.exp(x[3]-1),-1*x[2]*np.exp(x[3]-1)]])

H = lambda x: np.dot(grad_g(x), grad_g(x).T) + g0(x)*H0(x) + g1(x)*H1(x) + g2(x)*H2(x) + g3(x)*H3(x) + g4(x)*H4(x)



def Gauss_Newton_Puro(x0,f,grad_f,grad_g,tol,N):
    norma = lambda x: (sum(grad_f(x)**2))**.5
    i = 0
    
    while True:
        p = linalg.solve( H(x0),-grad_f(x0))
        xk = x0 +p
        norma_grad_f = norma(xk)
        i+=1
        if norma_grad_f < tol or i == N:
            break
        x0 = xk
        
    return i,xk,f(xk),norma_grad_f

Gauss_Newton_Puro(np.array([0,0,0,0]), f,grad_f, grad_g, 1e-8, 1000)


"""Al ejecutar el siguiente método con la función wolfe anterior, esta última no paraba de iterar.
Por ello hemos creado la siguiente función del método de Wolfe con un número máximo de iteraciones"""

def wolfe_max_iter(f, grad_f, x, p, c1=1e-4, c2=0.9,max_iter=100):
    a = 0
    b = 2
    continua = True
    cota = c2*np.dot(grad_f(x), p)
    fx=f(x)
    i = 0
    
    while f(x+b*p) <= fx+c1*b*np.dot(grad_f(x), p) and max_iter>i:
        b=2*b
        i+=1
        
    while continua == True and i<=max_iter:
        alpha=.5*(a+b)
        if f(x+alpha*p) > fx+c1*alpha*np.dot(grad_f(x), p):
            b=alpha
        elif np.dot(grad_f(x + alpha*p), p) < cota:
            a=alpha
        else:
            continua=False
        i += 1
    return alpha


def Newton_wolfe(x0,f,grad_f,grad_g,tol,N):
    norma = lambda x: (sum(grad_f(x)**2))**.5
    i = 0
    
    while True:
        p = np.linalg.solve(H(x0),-grad_f(x0))
        alpha = wolfe_max_iter(f, grad_f, x0, p)
        xk = x0 +p*alpha
        norma_grad_f = norma(xk)
        i+=1
        if norma_grad_f < tol or i == N:
            break
        x0 = xk
        
    return i,xk,f(xk),norma_grad_f,alpha

nw = Newton_wolfe(np.array([0,0,0,0]), f,grad_f, grad_g, 1e-8, 1000) 
print('\n El valor de alpha en la iteración 1000 del método de Newton es: ', nw[4])
"""El método converge en un punto no estacionario pues el valor de alpha en cada iteracion
es muy próximo a 0 por lo tanto el valor de la fución apenas se modifica en cada iteracion"""





