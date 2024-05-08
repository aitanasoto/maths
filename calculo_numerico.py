#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:48:43 2023

@author: aitana
"""
######PAQUETES########
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy.integrate import quad
import numpy.polynomial.polynomial as npp
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep
import numpy.polynomial.polynomial as npp
import scipy.linalg as la







#########BÁSICOS###########

#polinomios

p=npp.Polynomial([2,-1,-2,1]) #orden creciente
p.roots() #raices polinomio p

#listas y arrays
x=np.linspace(0,2,7) #7 puntos equisespaciados entre 0 y 2
np.array([])
np.sort #ordenar menor a mayor
9*np.random.rand(6)-5 # 6 num aleatorios entre -5 y 4

#integrar

sci.quad(lambda x: funcion depende d x,lim_inf,lim_sup)[0]

#######POLINOMIO INTERPOLADOR######

np.polyfit(x, y, deg) #x,y listas con nodos orden decreciente


def DifDivididas(x,y):
    n=len(x)-1
    A=np.zeros((n+1,n+1))
    for i in range(len(x)):
        A[i,0]=y[i]
    for j in range(1,n+1):
        for i in range(j,n+1):
            A[i,j]=(A[i][j-1]-A[i-1][j-1])/(x[i]-x[i-j])
    return A

def PolNewton(x,y):
    n=len(x)-1
    d=np.diag(DifDivididas(x,y))
    m=np.array([1])
    p=np.array([0])
    for i in range(n+1):
        p=np.polyadd(p,d[i]*m)
        m=np.polymul(m , [1 , -x[i]])
    return p


def DifDivid_Hermite(x,y,dy):
    n=len(x)
    A=np.zeros((2*n,2*n))
    #Creamos array nodos duplicados
    z= np.reshape([[x[i],x[i]] for i in range(n)],(1,2*n))
    yz=np.reshape([[y[i],y[i]] for i in range(n)],(1,2*n))
    dyz=np.reshape([[0,dy[i]] for i in range(n)],(1,2*n))
    for j in range(2*n):
        for i in range(j,2*n):
            if j==0:
                A[i,j]=yz[0,i]
            elif j==1 and i%2!=0:
                A[i,j]=dyz[0,i]
            else:
                A[i,j]= (A[i,j-1]-A[i-1,j-1])/(z[0,i]-z[0,i-j])
    return A

def PolHermite(x,y,dy):
    n=len(x)
    z=np.zeros(2*n)
    DD=np.diag(DifDivid_Hermite(x,y,dy))
    monomio = np.array([1])
    Pol = np.array([0])
    for i in range(2*n):
        z[i]=x[int(i/2)]
        Pol = np.polyadd(Pol,DD[i]*monomio)
        monomio = np.polymul(monomio,[1,-z[i]])
    return Pol

nodos= np.array([])
y = np.array([f(i) for i in nodos])
dy = np.array([df(i) for i in nodos])


##SPLINES##
from scipy.interpolate import CubicSpline

sp_natural=CubicSpline(x, y, bc_type="natural")
print(sp_natural(1.5)) #evalua el polinomio en ese punto

sp_frontera = CubicSpline(x, y, bc_type=((1,1/10),(1,1/10))) 
print(sp_frontera(2))
#devuelve matri filas son an,...,a0 y columnas s0,s1

#Splines
#construir el spline natural:
#sp_natural=CubicSpline(x,y,bc_type='natural')

#construir el spline de frontera fija
#sp_ffija=CubicSpline(x,y,bc_type=((1,f'(x0)),(1,f'(xn))))

#para pedir los coefs hacemos sp_natural.c
#nos devuelve un array de 4 filas 
#orden decreciente
#cada columna es un Si 
#para evaluarlo ponemos spline(x) (como si fuera una funcion)
#sp_natural=CubicSpline(x3,y3,bc_type='natural')
#sp=sp_natural.c
#len(sp) #da las filas

#Error
np.abs(np.polyval(p,x)-f(x))



########APROXIMACIÓN DE POLINOMIOS#######

##CHEBYSHEV###
TCh = np.polynomial.chebyshev.Chebyshev([0,0,1])
#grado 2: 0,0,1 grado3: 0,0,0,1
#devuelve dominio y algo mas

nodCheby=T2Ch.roots() 
Tch=npp.polyfromroots(nodCheby2)[::-1]  #POLINOMIO

def Chebyshev(n):
    if n==0:
        T = np.array([1])
        return T
    elif n==1:
        T = np.array([1,0])
        return T
    else:
        Tn1= Chebyshev(n-1)
        Tn2 = Chebyshev(n-2)
        Tn = np.polyadd(np.polymul(np.array([2,0]),Tn1),-Tn2)
           
    return Tn

def aprxChebyshev(f,n):
    Sol = np.array([0])
    
    for i in range(n+1):
        T = Chebyshev(i)
        a = quad(lambda x: (1/(np.sqrt(1-x**2)))*f(x)*np.polyval(T,x),-1,1)[0]
        b = quad(lambda x: (1/(np.sqrt(1-x**2)))*(np.polyval(T,x))**2,-1,1)[0]
        c = a/b
        Sol=np.polyadd(Sol,c*T)
    
    return Sol


###LEGENDRE###

def Legendre(n):
    if n==0:
        T = np.array([1])
        return T
    elif n==1:
        T = np.array([1,0])
        return T
    else:
        Tn1 = Legendre(n-1)
        Tn2 = Legendre(n-2)
        Tn  = np.polyadd(np.polymul(np.array([(2*n-1)/(n),0]),Tn1),np.polymul(np.array([(-n+1)/n]),Tn2))
        return(Tn)
    
def aprxLegendre(f,n):
    Sol = np.array([0])
    
    for i in range(n+1):
        T = Legendre(i)
        a = quad(lambda x: f(x)*np.polyval(T,x),-1,1)[0]
        b = quad(lambda x: (np.polyval(T,x))**2,-1,1)[0]
        c = a/b
        Sol=np.polyadd(Sol,c*T)
    
    return Sol   

#devuelve el polinomio

####PRACTICA UTIL###

def cv(x,a,b): #[-1,1] a [a,b]
    return -1+(2.0/(b-a)*(x-a))

def des_cv(x,a,b): #[a,b] a [-1,1]
    return a+(b-a)*(x+1)/2.0

fcv=lambda x: f(des_cv(x,3,6))
aLcv=aprxLegendre(fcv,6)
P3=lambda x: np.polyval(aLcv,cv(x,3,6))

#Error

error_L = (quad(lambda x: (f(x)-np.polyval(aL,x))**2,-1,1)[0])**0.5

####GRAFICAS#####
plt.plot()
plt.legend((" ", " ", " "), loc="best")
plt.xlabel("Abcisas")
plt.ylabel("Ordenadas")
plt.title("Gráfica ejercicio")
plt.show





"DERIVADA CON 3 Y 5 PUNTOS"

#Con función
def df3puntos(f,x0,h):
    df = (1/(2*h))*(f(x0+h)-f(x0-h))
    return df
    
def df5puntos(f,x0,h):
    df = (1/(12*h))*(f(x0-2*h)-8*f(x0-h)+8*f(x0+h)-f(x0+2*h))
    return df

#Con f(x) ya calculada

def dy_3ptos(y,h):
    dy=(y[2]-y[0])/(2*h)
    return dy

def dy_5puntos(y,h):
    return (y[0]-8*y[1]+6*y[3]-y[4])/(12*h)

"INTEGRALES NEWTON-COTES"

def trapecio(f,a,b):
    h = b - a
    i = (h/2)*(f(a)+f(b))
    return i

def trapecio_puntos(x,y):
    h = x[1]-x[0]
    i = (h/2)*(y[0]+y[1])
    return i

def simpson(f,a,b):
    x = np.linspace(a,b,3)
    h = (b - a)/2
    i = (h/3)*(f(x[0])+4*f(x[1])+f(x[2]))
    return i

def simpson_puntos(x,y):
    h = (x[2]-x[0])/2
    i = (h/3)*(y[0]+4*y[1]+y[2])
    return i
    
    
def simpson38(f,a,b):
    h = (b - a)/3
    x = np.linspace(a,b,4)
    i = ((3*h)/8)*(f(x[0])+3*f(x[1])+3*f(x[2])+f(x[3]))
    return i

def simpson38_puntos(x,y):
    h = (x[3]-x[0])/3
    i = ((3*h)/8)*(y[0]+3*y[1]+3*y[2]+y[3])
    return i

"INTEGRALES COMPUESTA (intervalos +grandes +grado)"

def simpson_compuesta(f,a,b,n):
    x = np.linspace(a,b,n+1)
    h = (b-a)/n
    integral = 0
    
    for i in range(0,n//2):
        suma = f(x[2*i])+4*f(x[2*i+1])+f(x[2*i+2])
        integral += suma
        
    integral = (h/3) * integral
    
    return integral

def simpson_compuesta_puntos(x,y):
    integral = 0
    for i in range(0,(len(x)-1)//2):
        suma = simpson_puntos(x[2*i:2*i+3],y[2*i:2*i+3])
        integral += suma
    return integral
    

def trapecio_compuesta(f,a,b,n):
    x = np.linspace(a,b,n+1)
    h = (b-a)/n
    suma = 0
    for i in range(1,n):
        suma += f(x[i])
    integral = (h/2)*(f(a)+f(b)+2*suma)
    return integral

def trapecio_compuesta_puntos(x,y):
    integral = 0
    for i in range(0,len(x)-1):
        suma = trapecio_puntos(x[i:i+2],y[i:i+2])
        integral += suma
    return integral




"GRAFICAR LAS ÁREAS (EJEMPLO)"

plt.plot(x,np.polyval(np.polyfit(x31,f2(x31),1),x))
plt.fill_between(x,0,np.polyval(P_trap,x), alpha = 0.45)

"DESCOMPOSICIÓN LU"
P, L, U = la.lu(A) #devuelve P(identidad) ,L,U 

"MÉTODOS ITERATIVOS SIST.LINEALES"

A3 = np.array(#matriz,dtype=float)
b3 = np.array(#vector)
x0=np.zeros(#n)
            
norma = np.inf #norma infinito
k= 100 #está puesto este valor en todos los ejemplos

#LAS FUNCIONES DEVUELVEN [SOL,ITERACIONES,RADIO ESPECTRAL]

def Jacobi(A,b,x0,norma,error,k):
    D = np.diag(np.diag(A))
    L = -np.tril(A,-1)
    U = -np.triu(A,1)
    M = D
    N = L+U
    B=la.inv(M)@N
    c = la.inv(M)@b
    r0=max(abs(la.eig(B)[0]))
    if r0>=1:
        print("La matriz no converge")
    i=1
    while True:
        if i>=k:
            print("El método no converge en",k,"pasos")
            return[x0,k,r0]
        x1 = np.dot(B,x0)+c
        if la.norm(x1-x0,norma)<error:
            return[x1,i,r0]
        i = i+1
        x0=x1.copy()
     
def Gauss_seidel(A,b,x0,norma,error,k):
    D = np.diag(np.diag(A))
    L = -np.tril(A,-1)
    U = -np.triu(A,1)
    M = D-L
    N = U
    B=la.inv(M)@N
    c = la.inv(M)@b
    r0=max(abs(la.eig(B)[0]))
    if r0>=1:
        print("La matriz no converge")
    i=1
    while True:
        if i>=k:
            print("El método no converge en",k,"pasos")
            return[x0,k,r0]
        x1 = np.dot(B,x0)+c
        if la.norm(x1-x0,norma)<error:
            return[x1,i,r0]
        i = i+1
        x0=x1.copy()
        
def Sor(A,b,x0,norma,error,k,w):
    D = np.diag(np.diag(A))
    L = -np.tril(A,-1)
    U = -np.triu(A,1)
    M = 1/w*(D-w*L)
    N = 1/w*((1-w)*D+w*U)
    B=la.inv(M)@N
    c = la.inv(M)@b
    r0=max(abs(la.eig(B)[0]))
    if r0>=1:
        print("La matriz no converge")
        return([],0,r0)
    i=1
    while True:
        if i>=k:
            print("El método no converge en",k,"pasos")
            return[0,k,r0]
        x1 = np.dot(B,x0)+c
        if la.norm(x1-x0,norma)<error:
            return[x1,i,r0]
        i = i+1
        x0=x1.copy()
        
#ENCONTRAR EL VALOR DE w ÓPTIMO

a = np.linspace(0.1,1.9,19) #valores entres lo q tomamos w

iter=[]

for w in a:
    [x3,i3,r3]=Sor(A3,b3,x0,np.inf,10**(-3),100,w)
    iter.append(i3)
    
iter
k=np.argmin(iter) #return el índice del mínimo valor
w_opt=a[k]  #w óptimo

"ECUACIONES NO LINEALES"

"Primero hay q graficar la función para elegir los intervalos donde"
"estan los 0s pues las siguientes funciones no dan una única solución"

x = np.linspace(0,15,150)
plt.plot(x,f(x),label="Función")
plt.plot(x,0*x,label="Eje x")
plt.legend(loc = "best")
plt.xlabel("Abcisas")
plt.ylabel("Ordenadas")
plt.show()

def biseccion(f,a,b,tol,maxiter):
    #f funcion q determina la ecuacion
    #a ext inferior
    #b ext supr
    #tolerancia
    #maxiter:numero max d iteraciones
    if f(a)*f(b)>0:
        print("El intervalo no es adecuado")
    i = 0
    while(i<maxiter) and abs(b-a)>=tol:
        p = (a+b)/2.0
        if f(p)==0:
            return [p,i]
        else:
            if f(a)*f(p)>0:
                a = p
            else:
                b = p
        i = i+1
    return[p,i]

def dy_5pts(f,x0,h):
    dy = (f(x0-2*h)-8*f(x0-h)+8*f(x0+h)-f(x0+2*h))/(12*h)
    return dy


def newton(f,a,b,intermax,tol):
    x0=biseccion(f,a,b,.001,4)[0]
    df = dy_5pts(f,x0,.001)
    x1 = x0-(f(x0)/df)
    i = 1
    while(i<intermax) and abs(x1-x0)>tol:
        x0 = x1
        df = dy_5pts(f,x0,0.1)
        x1 = x0-(f(x0)/df)
        i = i+1
    return x0,x1,i

#Newton con orden de convergencia

def Newton(f,x0,tol,p,R):
    df = dy_5pts(f,x0,0.1)
    x1 = x0-(f(x0)/df)
    i = 1
    while abs(x1-x0)>tol:
        i = i+1
        x0 = x1
        df = dy_5pts(f,x0,0.1)
        x1 = x0-(f(x0)/df)
        e0 = np.abs(x0-p)
        e1 = np.abs(x1-p)
        A = e1/(e0)**R
    return [i,x1,x0,e1,A]

#R = orden de convergencia(1 lineal,2 cuadrática,>2 converge rapido)

def secante(f,a,b,tol,numiter):
    #Obtengo x0 por biseccion
    x0 = biseccion(f,a,b,0.01,1)[0]
    x1 = x0 - (f(x0))/(dy_5pts(f, x0,0.1))
    x = [x0,x1]
    for i in range(numiter):
        new = x[i+1] - ((x[i]-x[i+1])*f(x[i+1]))/(f(x[i])-f(x[i+1]))
        x.append(new)
        if np.abs(x[-1]-x[-2])<tol:
            return x[-1], i+1
    return x[-1], i+1

def regula_falsi(f,a,b,tol,numiter):
    x0 = biseccion(f,a,b,0.01,1)[0]
    x1 = x0 - (f(x0))/(dy_5pts(f, x0,0.1))
    x = [x0,x1]
    
    
    for i in range(numiter):
        new = x[i+1] - ((x[i]-x[i+1])*f(x[i+1]))/(f(x[i])-f(x[i+1]))
        x.append(new)
        if np.abs(x[-1]-x[-2])<tol:
            return x[-1], i+1
    #Aseguramos que la siguiente iteracion esté entra las dos anteriores
        z = x.copy()
        if f(x[-2])*f(x[-1])<0:
            z[-1]=x[-1]
            z[-2]=x[-2]
        else:
            z[-1]=x[-1]
            z[-2]=x[-3]
        x = z.copy()
    return x[-1], i+1

"SISTEMAS NO LINEALES"

def Jac(F,x0,h):
    n = len(x0)
    J = np.zeros([n,n])
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1
        J[:,j] = (F(x0-2*h*e)-8*F(x0-h*e)+8*F(x0+h*e)-F(x0+2*h*e))/(12*h)
    return J
    
    


def NewtonSist(F,X0,h,tol,norm,maxit):
    i = 1
    cond = False
    while i < maxit:
        X1 = X0 + np.linalg.solve(Jac(F,X0,h),-F(X0)) 
        if la.norm(X0-X1,norm) > tol:
            X0 = X1.copy()
            i+=1
        else:
            cond = True
            break
    if cond == True:
        return [X1,i]
    else:
        print('El método no converge en {0} iteraciones'.format(maxit))
        return [X1,maxit]
            
#Así se define un campo vectorial en Python:
    
def F1(X):
    f1 = np.sin(X[0]) + X[1]**2 + np.log(X[2]) - 7
    f2 = 3*X[0] + 2**X[1] - 1/((X[2])**3) + 1
    f3 = X[0] + X[1] + X[2] - 5
    return np.array([f1,f2,f3])
