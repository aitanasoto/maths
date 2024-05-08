from random import random


#FUNCIÓN QUE DEVUELVE PALABRA SIN ACENTOS NI MAYUSC.
def QuitaAcentos(palabra):
    PAL = str()
    palabra = palabra.lower()
    az = 'áâà';
    ez = 'éêè';
    iz = 'íîì';
    oz = 'óôò';
    uz = 'úûù'
    for ch in palabra:
        if ch in az:
            ch = 'a'
        elif ch in ez:
            ch = 'e'
        elif ch in iz:
            ch = 'i'
        elif ch in oz:
            ch = 'o'
        elif ch in uz:
            ch = 'u';
        PAL = PAL + ch
    return PAL


################################################

##LISTAS##
#SI HACEMOS L1+L2 PILLA LA CONCATENACIÓN DE LAS FILAS (UNIÓN)
#SI HACEMOS max(L1+L2) /min() nos devuelve directamente el máximo/mínimo de la unión de las listas
#para hacer unión sin repetidos

def Quitar_repetidos_lista(Lista):
    QQ = list(L)
    for i in range(len(L)-1, -1, -1):
        if L[i] in L[:i] : del QQ[i] #comando del elimina la posición i de la lista
    return QQ

#FUNCIÓN QUE PASA DE UNA PALABRA A UNA LISTA
def palabraAlista(palabra):
    lista=[]
    palabracad=str(palabra)
    for c in palabracad:
        lista.append(c)
    return lista


def divisores(n):
    l = []
    for i in range(1,n+1):
        if n % i == 0:
            l.append(i)
    return l


#hacer la unión e intersección de listas#
def insersect_listas(lista1, lista2):
    l = []
    for i in lista1:
        if i in lista1 and i in lista2:
            l.append(i)
    return l


def union_listas(lista1, lista2):
    if len(lista1)<len(lista2):
        for elem in lista2:
            if elem not in lista1:
                lista1.append(elem)
        return lista1
    else:
        for elem in lista1:
            if elem not in lista2:
                lista2.append(elem)
        return lista2

#devuelve el número más alto de una lista
def maximo_lista(lista):
    max = lista[0]
    for i in range(len(lista)):
        if lista[i]>max:
            max = lista[i]
    return max

#devuelve el mínimo valor de una lista
def minimo_lista(lista):
    min = lista[0]
    for i in range(len(lista)):
        if lista[i]<min:
            min = lista[i]
    return min

################################################

##MATRICES##

#de números reales#
def cadena_a_lista_num_float(cadena):
    lista = []
    try:
        for c in cadena.split(','):
            lista.append(float(c))

    except ValueError as e:
        raise ValueError('Uno de los elementos de la lista separados por coma no es un número')

    return lista


#de números enteros#
def cadena_a_lista_num_int(cadena):
    lista = []
    try:
        for c in cadena.split(','):
            lista.append(int(c))

    except ValueError as e:
        raise ValueError('Uno de los elementos de la lista separados por coma no es un número')

    return lista

#para pedir matriz o hacer una random#
def introduce_matriz():
    M = []
    fila = None
    i = 1
    while fila != []:
        try:
            cadena = input('Introduce la fila {0}: '.format(i))
            fila = cadena_a_lista_num_float(cadena)
            if fila != []:
                M.append(fila)
                i += 1
        except ValueError as e:
            print(e, 'o has puesto un espacio en blanco')
            print('Voy a imprimir la matriz que llevas hasta el momento')
            fila = []
    return M



def crea_matriz_random(n,m,max):
    M = []
    for i in range(n):
        lista = []
        for j in range(m):
            lista.append(random() * max)
        M.append(lista)
    return M

def imprime_matriz(M):
    for i in range(len(M)):
        for j in range(len(M[i])):
            print(M[i][j], end="   ")  #tambien se puede poner end = '\t'
        print()

#operaciones con matrices#
def suma_matrices(m1, m2):
    if len(m1) != len(m2):
        print('Las matrices no tienen el mismo números de filas')

    matriz = []
    for j in range(len(m1)):
        fila = []
        if len(m1[j]) != len(m2[j]):
            print('Las matrices no tienen el mismo número de columnas.')
        for i in range(len(m1[j])):
            fila.append(m1[j][i] + m2[j][i])
        matriz.append(fila)
    return matriz


def resta_matrices(m1, m2):
    if len(m1) != len(m2):
        print('Las matrices no tienen el mismo números de filas')

    matriz = []
    for j in range(len(m1)):
        fila = []
        if len(m1[j]) != len(m2[j]):
            print('Las matrices no tienen el mismo número de columnas.')
        for i in range(len(m1[j])):
            fila.append(m1[j][i] - m2[j][i])
        matriz.append(fila)
    return matriz


def multiplicacion_matrices(m1, m2):
    matriz=[]
    for i in range(len(m1)):
        matriz.append([])
        if len(m2) == 0 or len(m1[i]) != len(m2):
            print('El número de columnas de la primera matriz debe ser igual al número de filas de la segunda.')
        for j in range(len(m2[0])):
            matriz[i].append(0)
            for k in range(len(m2)):
                if k >= len(m1[i]) or j >= len(m2[k]):
                    print('El número de columnas de la primera matriz debe ser igual al número de filas de la segunda.')
                matriz[i][j] += m1[i][k] * m2[k][j]
    return matriz


#eliminar elementos de la matriz#

#funciona tomando fila 1 como la primera
def eliminar_fila(M, fila):
    nueva_M = []
    for i in range(len(M)):
        if i != fila - 1:    #hacemos que fila cuente como las matriz, en verdad en M la primera fila es la 0 no la 1
            nueva_M.append(M[i])
    return nueva_M

#funciona tomando fila 1 como la primera
def eliminar_columna(M, columna):
    nueva_M = []
    for i in range(len(M)):
        fila_sin_elem_col = []
        for j in range(len(M[i])):
            if j != columna - 1:
                fila_sin_elem_col.append(M[i][j])
        nueva_M.append(fila_sin_elem_col)
    return nueva_M


def quitar_elemento(M, elem):
    M_sin_elem = []
    for i in range(len(M)):
        fila_sin_elem = []
        for j in range(len(M[i])):
            if elem != M[i][j]:
                fila_sin_elem.append(M[i][j])
        M_sin_elem.append(fila_sin_elem)
    return M_sin_elem


#hacer traspuesta# #interesante a la hora de trabajar con columnas, ya que por fulas es más sencillo tratar las matrices (son listas)#
def transpuesta(M):
    t = list([])
    for i in range(len(M[0])):
        t.append([])
        for j in range(len(M)):
            t[i].append(M[j][i])
    return t

def traspuesta_ratoli(M):
    traspuesta = []
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i == 0:
                traspuesta.append([])
            traspuesta[j].append(M[i][j])

    return traspuesta

#unir las filas de una matriz (o columnas si trabajmos con trapuesta) interesante para comparar o ordenar de mayor a menor y eso#
def union_filas(M):
    union = []
    for i in range(len(M)):
        for j in range(len(M[i])):
            union.append(M[i][j])
    union_sin_repetir = list(set(union))
    return union_sin_repetir

#estudiar si un elemento etá en más de una fila#
def esta_en_otras_listas(matriz, elem):
    for lista in matriz:
        if elem not in lista:
            return False
    return True

def interseccion_filas(M):
    inter = []
    for lista in M:
        for elem in lista:
            if esta_en_otras_listas(M,elem) and elem not in inter:
                inter.append(elem)
    return inter

#hallar determinante hasta 3x3#
def determinante(M):
    if len(M) == 1:
        det = M[0][0]
    if len(M) == 2:
        det = M[0][0] * M[1][1] - M[1][0] * M[0][1]
    if len(M) == 3:
        det = M[0][0] * M[1][1] * M[2][2] + M[1][0] * M[2][1] * M[0][2] + M[0][1] * M[1][2] * M[2][0] - M[0][2] * M[1][1] * M[2][0] - M[0][0] * M[1][2] * M[2][1] - M[0][1] * M[1][0] * M[2][2]
    return det

#nxn por gauss#
def Triangulacion(Mat):
    n=len(Mat)
    F=Mat; a=Mat
    for i in range(n-1):
        for j in range(i+1,n):
            for h in range(n):
                F[j]=[F[j][h]-a[j][i] / a[i][i]*F[i][h]]
    return Mat

def Determinante_Gauss(Mat):
    T=Triangulacion(Mat)
    pronf=1
    for h in range(n):
        prod *= T[h][h]
    return prod


#crea lista con la columna i
def columnacompleta(numerodecolumna,M):
    col=[]
    for i in range(len(M)):
        if len(M[i])>=numerodecolumna:
            col.append((M[i][numerodecolumna]))
    return col


##########################################################

##VALUEERROR#

#A la hora de hacer una función. Si queremos que salga por la salida de ValueError cuando introduce algo distinto
#a lo que realmente pedimos, PERO no da valueerror como tal. Hacemos:

#if jskska:
#else:
# raise ValueError('Esa opción no es válida')       #por ejemplo si pide número del 1 al 4  y damos el 5



#Por otro lado, si ese ValueError lo pilla ya python.  #por ejemplo pide número y damos letra
#Haremos:
#try:
#jdklcndnlfnvf
#except ValueError as e:
# print('Esa opción no es Válida.', file=stderr)

#para ello, cargamos antes el paquete from sys import stderr














