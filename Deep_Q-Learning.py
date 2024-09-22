#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import numpy as np 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(1)
tf.random.set_seed(1)


# In[2]:


# Red neuronal

class Brain(tf.keras.Model):
    def __init__(self,output_action,input_state,lr=0.001):
        self.lr=lr
        self.input_state=input_state
        self.output_action=output_action
        super(Brain,self).__init__()
        self._build()
       
    
    def _build(self):
      self.model=tf.keras.Sequential([
          tf.keras.layers.Input(shape=(self.input_state,)),
          tf.keras.layers.Dense(units=64,activation='sigmoid'),
          tf.keras.layers.Dropout(rate=0.1),
          tf.keras.layers.Dense(units=32,activation='sigmoid'),
          tf.keras.layers.Dropout(rate=0.1)
      ])
      self.model.add(tf.keras.layers.Dense(units=self.output_action,activation='softmax'))
      self.model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
          loss='mse'
      )
      return self.model
    


# In[3]:


# Ambiente

class Enviroment(object):
    def __init__(self,optimal_t=(18.0,24.0),initial_moth=0,initial_users=10,initial_rate_data=60):
        self.monthly_atmospheric_temperatures=[1.0,5.0,7.0,10.0,11.0,20.0,23.0,24.0,22.0,10.0,5.0,1.0]
        self.initial_moth=initial_moth
        self.atmospheric_temperature=self.monthly_atmospheric_temperatures[self.initial_moth]
        self.optimal_t=optimal_t
        self.min_temprature=-20
        self.max_temperature=80
        self.min_users=10
        self.max_users=100
        self.max_update_users=5
        self.min_rate_data=20
        self.max_rate_data=300
        self.max_update_data=10
        self.initial_number_users=initial_users
        self.current_number_users=initial_users
        self.initial_rate_data=initial_rate_data
        self.current_rate_data=initial_rate_data
        self.intrinsic_temperature=self.atmospheric_temperature+1.25*self.current_number_users+1.25*self.current_rate_data
        self.temperature_ai=self.intrinsic_temperature
        self.temperature_noai=(self.optimal_t[0]+self.optimal_t[1])/2.0
        self.total_energy_ai=0.0
        self.total_energy_noai=0.0
        self.reward=0.0
        self.game_over=0
        self.train=1
    
    def update_env(self,direction,energy_ai,month):
        energy_noai=0

        if (self.temperature_noai < self.optimal_t[0]):
            energy_noai=self.optimal_t[0]-self.temperature_noai
            self.temperature_noai=self.optimal_t[0]
        elif (self.temperature_noai > self.optimal_t[1]):
            energy_noai=self.temperature_noai-self.optimal_t[1]
            self.temperature_noai=self.optimal_t[1]
        
        self.reward = energy_noai-energy_ai
        self.reward = 1e-3*self.reward
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
        self.current_number_users += np.random.randint(-self.max_update_users,self.max_update_users)
        
        if (self.current_number_users > self.max_users):
            self.current_number_users = self.max_users
        elif (self.current_number_users < self.min_users):
            self.current_number_users=self.min_users
        
        self.current_rate_data += np.random.randint(-self.max_update_data,self.max_update_data)

        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        
        past_intrinsic_temperature=self.intrinsic_temperature
        self.intrinsic_temperature = self.atmospheric_temperature+1.25*self.current_number_users+1.25*self.current_rate_data
        delta_intrinsic_temperature=self.intrinsic_temperature - past_intrinsic_temperature

        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif (direction == 1):
            delta_temperature_ai = energy_ai

        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
        self.temperature_noai += delta_intrinsic_temperature

        if (self.temperature_ai < self.min_temprature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.optimal_t[0]-self.temperature_ai
                self.temperature_ai = self.optimal_t[0]
        elif (self.temperature_ai > self.max_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.temperature_ai - self.optimal_t[1]
                self.temperature_ai = self.optimal_t[1]

        self.total_energy_noai += energy_noai
        self.total_energy_ai += energy_ai
        
        scaled_temperature_ai = (self.temperature_ai - self.min_temprature)/(self.max_temperature - self.min_temprature)
        scaled_number_users = (self.current_number_users - self.min_users)/(self.max_users - self.min_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temperature_ai,scaled_number_users,scaled_rate_data])

        return next_state,self.reward,self.game_over
    
    def reset(self,new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_moth = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature +  1.25*self.current_number_users + 1.25*self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_t[0] + self.optimal_t[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
    
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temprature)/(self.max_temperature - self.min_temprature)
        scaled_number_users = (self.current_number_users - self.min_users)/(self.max_users - self.min_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai,scaled_number_users,scaled_rate_data])
        return current_state,self.reward,self.game_over
                                


# In[4]:


# Implementacion del Deep reinforcement learning algorithm

class DQN(object):
    def __init__(self,max_memory=100,discount=0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount
    
    def remember(self,trasition,game_over):
        self.memory.append([trasition,game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
        
    def get_batch(self,model,batch_size=10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_ouputs = model.model.output_shape[-1]
        inputs = np.zeros((min(len_memory,batch_size),num_inputs))
        targets = np.zeros((min(len_memory,batch_size),num_ouputs))

        for i, dx in enumerate(np.random.randint(0,len_memory,size=min(len_memory,batch_size))):
            current_state,action,reward,next_state = self.memory[dx][0] 
            game_over=self.memory[dx][1]
            inputs[i] = current_state
            targets[i] =  modelo.model.predict(current_state)[0]
            Q_sa = np.max(modelo.model.predict(next_state)[0])

            if game_over:
                targets[i,action] = reward
            else:
                targets[i,action] = reward + self.discount *Q_sa
            
        return inputs,targets
    


# In[5]:


# Entrenamineto del modelo

epsilon = .3
number_action = 5
input_in=3
direction_boundary = (number_action - 1)/ 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# LLamando al ambiente 
env=Enviroment(optimal_t=(18.0,24.0),initial_moth=0,initial_users=20,initial_rate_data=30)
# LLamando a la red 
modelo=Brain(output_action=number_action,lr=0.00001,input_state=input_in)
# LLamando al DQN
dqn=DQN(max_memory=max_memory)

train=True

env.train=train
if (env.train):
    for epochs in range(1,number_epochs):
        total_reward=0
        loss=0.
        new_month=np.random.randint(0,12)
        env.reset(new_month=new_month)
        game_over=False
        current_state,_,_=env.observe()
        timetep=0

        while ((not game_over) and timetep <= 5*30*24*60):
            if np.random.rand() <= epsilon:
                action=np.random.randint(0,number_action)
                if (action - direction_boundary  < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary)*temperature_step
            else:
                q_value=modelo.model.predict(current_state)
                action=np.argmax(q_value[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary)*temperature_step
            
            next_state,reward,game_over=env.update_env(direction,energy_ai,(new_month+ int(timetep/(30*24*60)))%12)
            total_reward += reward
            dqn.remember([current_state,action,reward,next_state],game_over)

            inputs,targets=dqn.get_batch(modelo,batch_size=batch_size)
            loss += modelo.model.train_on_batch(inputs,targets)
            timetep += 1
            current_state = next_state

        print('\n')
        print('Epochs: {:03d}/{:03d}'.format(epochs,number_epochs))
        print('Total Energy spent with an AI: {:.0f}'.format(env.total_energy_ai))
        print('Total Energy spent with no AI:  {:.0f}'.format(env.total_energy_noai))


# In[ ]:




