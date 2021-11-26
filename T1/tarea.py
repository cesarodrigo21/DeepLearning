

#GRADIENTE DESCENDENTE PARA RESOLVER EL PROBLEMA DE REGRESION LINEAL
#Agregar al programa instrucciones que muestren lo siguiente gráficamente:

from numpy import *
from matplotlib import pyplot as plt
def gradient_descent(alpha, x, y, ep, max_iter):
  convergio = False
  iter = 0
  N = len(x) #-- número de ejemplos
  #-- valores iniciales de theta
  t0 = 0
  t1 = 0
  #-- error total, J(theta)
  J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(N)])
  #-- ciclo de iteraciones
  #lista_obtenida[0]=lista de t0
  #lista_obtenida[1]=lista de t1
  #lista_obtenida[2]=lista de error
  #lista_obtenida[3]=lista de iteraciones
  lista_obtenida = [[],[],[],[]]
  while not convergio:
    #-- para cada ejemplo de entrenamiento calcular el gradiente (d/d_theta)j(theta)
    grad0 = 1.0/N * sum([(t0 + t1*x[i] - y[i]) for i in range(N)])
    grad1 = 1.0/N * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(N)])
    #-- actualizar las thetas temporales
    temp0 = t0 - alpha * grad0
    temp1 = t1 - alpha * grad1
    #-- actualizar las theta
    t0 = temp0
    t1 = temp1
    lista_obtenida[0].append(t0)
    lista_obtenida[1].append(t1)
    #-- calcula el error cuadrado medio
    e = sum( [(t0 + t1*x[i] - y[i])**2 for i in range(N)] )
    if abs(J-e) <= ep:
      print ('Convergió con iteraciones: ', iter, '!!!')
      convergio = True
    J = e #-- actualizar error
    iter += 1 #-- incrementa iteraciones
    lista_obtenida[2].append(e)
    lista_obtenida[3].append(iter)
    if iter == max_iter: #-- si no converge
      print ('Se excedió del máximo de iteraciones!')
      convergio = True 
  return t0,t1,lista_obtenida #-- devuelve los valores calculados de theta0 y theta1
#------- programa principal--------------
#-- lee puntos de muestra del archivo data.csv
puntos = genfromtxt("data.csv", delimiter=",")
#-- las coordenadas x en el arreglo x, las coordenadas y en el arreglo y
x = []
y = []
N = len(puntos) #-- nro de puntos
print("Nro de datos leidos: ", N)
for i in range(N):
  x.append(puntos[i,0])
  y.append(puntos[i,1])
#-- parametros iniciales
alfa= 0.0001 #-- learning rate
ep = 0.00001 #-- tolerancia
max_itera= 1000000 #-- nro maximo de iteraciones
#-- comienza el gradiente descendente
tetha0, tetha1, lista_para_graficar = gradient_descent(alfa,x,y,ep,max_itera)
#-- mostrar resultados
print('Theta0: ',tetha0)
print('Theta1: ',tetha1)


#PARTE A : Error Cuadrado Medio versus el número de iteraciones.
import numpy as np
plt.figure(figsize=(10,6))
plt.plot(lista_para_graficar[3], lista_para_graficar[2])
plt.xlabel('Numero de iteraciones')
plt.ylabel('Error cuadrado medio')
plt.title('Error cuadrado medio VS Numero de iteraciones')
plt.grid(True)

#PARTE B : Los valores de θ1 y θ0 versus el número de iteraciones.
plt.figure(figsize=(10,6))
plt.plot(lista_para_graficar[3], lista_para_graficar[0], label='θ0 vs iteraciones')
plt.plot(lista_para_graficar[3], lista_para_graficar[1], label='θ1 vs iteraciones')
plt.xlabel('Numero de iteraciones')
plt.ylabel('Valores de Theta')
plt.title('θ0 y θ1 VS el número de iteraciones')
plt.legend()
plt.grid(True)
