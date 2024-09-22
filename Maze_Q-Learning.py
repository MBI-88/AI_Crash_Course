#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
np.random.seed(42)


# In[6]:


# Ambiente

location_state={
    'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11
}

action=[0,1,2,3,4,5,6,7,8,9,10,11]

alpha=0.9 # factor de descuent
gamma=0.75 # learning rate

# Sistema de premios

Rewards=np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0],# A
    [1,0,1,0,0,1,0,0,0,0,0,0],# B
    [0,1,0,0,0,0,1,0,0,0,0,0],# C
    [0,0,0,0,0,0,0,1,0,0,0,0],# D
    [0,0,0,0,0,0,0,0,1,0,0,0],# E
    [0,1,0,0,0,0,0,0,0,1,0,0],# F
    [0,0,1,0,0,1000,1,0,0,0,0,0],# G
    [0,0,0,1,0,0,1,0,0,0,0,1],# H
    [0,0,0,0,1,0,0,0,0,1,0,0],# I
    [0,0,0,0,0,1,0,0,1,0,1,0],# J
    [0,0,0,0,0,0,0,0,0,1,0,1],# K
    [0,0,0,0,0,0,0,1,0,0,1,0] # L
])

Q_value=np.array(np.zeros([12,12])) # funcion de valores Q

# Entrenamiento

for i in range(1000):
    current_state=np.random.randint(0,12) # Eleccion de un estado aleatorio
    playable_action=[] # Acciones elegidas
    for j in range(12):
        if Rewards[current_state,j] > 0:
            playable_action.append(j)
    next_state=np.random.choice(playable_action)
    TD=Rewards[current_state,next_state] + gamma * Q_value[next_state,np.argmax(Q_value[next_state,])]-Q_value[current_state,next_state] # Diferencia temporal
    Q_value[current_state,next_state]=Q_value[current_state,next_state] + alpha * TD # Ecuacion Bellman

print('Q_Values: ',Q_value.astype(int))


# In[15]:


# Inferencia

state_location={
    state : location for location, state in location_state.items()}

def prediction(starting_location,ending_location):
    route=[starting_location]
    next_location=starting_location
    while (next_location != ending_location):
        starting_state=location_state[starting_location]
        next_state=np.argmax(Q_value[starting_state,])
        next_location=state_location[next_state]
        route.append(next_location)
        starting_location=next_location
    return route

Ruta=prediction('E','G')
print('Direccion: ',Ruta)


# In[21]:


# Mejorando el modelo

def route(starting_location,ending_location):
    R_new=np.copy(Rewards)
    ending_state=location_state[ending_location]
    R_new[ending_state,ending_state]=1000
    Q_value_new=np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state=np.random.randint(0,12)
        playable_action=[]
        for j in range(12):
            if R_new[current_state,j] > 0:
                playable_action.append(j)
        next_state=np.random.choice(playable_action)
        TD=R_new[current_state,next_state]+ gamma * Q_value_new[next_state,np.argmax(Q_value_new[next_state,])]- Q_value_new[current_state,next_state]
        Q_value_new[current_state,next_state]=Q_value_new[current_state,next_state]+ alpha * TD
    route=[starting_location]
    next_location=starting_location
    while (next_location != ending_location):
        starting_state=location_state[starting_location]
        next_state=np.argmax(Q_value_new[starting_state,])
        next_location=state_location[next_state]
        route.append(next_location)
        starting_location=next_location
    return route

Location=route('E','G')
print(Location)


# In[24]:


# Mejoramiento 2 Creando una locacion intermedia para facilitar la ruta al modelo

def best_route(starting_location,intermdiary_location,ending_location):
    return route(starting_location,intermdiary_location) + route(intermdiary_location,ending_location)[1:]


Best_route=best_route('E','K','G')
print(Best_route)


# In[ ]:




