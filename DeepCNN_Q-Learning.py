#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pygame as pg
import tensorflow as tf 
import matplotlib.pyplot as plt 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)
np.random.seed(1)


# In[2]:


# Ambiente
class Enviroment():
    def __init__(self,waitTime):
        self.width = 880
        self.height =  880
        self.nRows = 10
        self.nColumns = 10
        self.initSnakelen = 2
        self.defReward = -0.03
        self.negReward = -1
        self.posReward = 2
        self.waitTime = waitTime

        if self.initSnakelen > self.nRows/2:
            self.initSnakelen = int(self.nRows/2)
        
        self.screen = pg.display.set_mode((self.width,self.height))
        self.snakePos = list()
        self.screenMap = np.zeros((self.nRows,self.nColumns))

        for i in range(self.initSnakelen):
            self.snakePos.append((int(self.nRows/2)+i,int(self.nColumns/2)))
            self.screenMap[int(self.nRows/2)+i][int(self.nColumns/2)] = 0.5
        
        self.applePos = self.placeAple()
        self.drawScreen()
        self.collected = False
        self.lastMove = 0
    
    def placeAple(self):
        posx = np.random.randint(0,self.nColumns)
        posy = np.random.randint(0,self.nRows)
        while self.screenMap[posx][posy] == 0.5:
            posx = np.random.randint(0,self.nColumns)
            posy = np.random.randint(0,self.nRows)
        self.screenMap[posx][posy] = 1
        return (posx,posy)
    
    def drawScreen(self):
        self.screen.fill((0,0,0))
        cellWidth = self.width/self.nColumns
        cellHeight = self.height/self.nRows

        for i in range(self.nRows):
            for j in range(self.nColumns):
                if self.screenMap[i][j] ==  0.5:
                    pg.draw.rect(self.screen,(255,255,255),(j*cellWidth + 1,i*cellHeight + 1,cellWidth - 2,cellHeight - 2))
                elif self.screenMap[i][j] == 1:
                    pg.draw.rect(self.screen,(255,0,0),(j*cellWidth + 1,i*cellHeight +  1,cellWidth - 2,cellHeight - 2))
        
        pg.display.flip()
    
    def moveSnake(self,nextPos,col):
        self.snakePos.insert(0,nextPos)

        if not col :
            self.snakePos.pop(len(self.snakePos)-1)

        self.screenMap = np.zeros((self.nRows,self.nColumns))

        for i in range(len(self.snakePos)):
            self.screenMap[self.snakePos[i][0]][self.snakePos[i][1]] = 0.5
        
        if col:
            self.applePos = self.placeAple()
            self.collected = True
        
        self.screenMap[self.applePos[0]][self.applePos[1]] = 1
    
    def step(self,action):
        gameOver = False
        reward = self.defReward
        self.collected = False

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
        snakeX = self.snakePos[0][1]
        snakeY = self.snakePos[0][0]

        if action == 1 and self.lastMove == 0:
            action = 0
        if action == 0 and self.lastMove == 1:
            action = 1
        if action == 3 and self.lastMove == 2:
            action = 2
        if  action == 2 and self.lastMove == 3:
            action = 3
        
        if action == 0:
            if snakeY > 0:
                if self.screenMap[snakeY - 1][snakeX]  == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY - 1][snakeX] ==  1:
                    reward = self.posReward
                    self.moveSnake((snakeY - 1,snakeX),True)
                elif self.screenMap[snakeY - 1][snakeX] == 0:
                    self.moveSnake((snakeY - 1,snakeX),False)
            else:
                gameOver = True
                reward = self.negReward
        
        elif action == 1:
            if snakeY < self.nRows - 1:
                if self.screenMap[snakeY + 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY + 1][snakeX] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY + 1,snakeX),True)
                elif self.screenMap[snakeY + 1][snakeX] == 0:
                    self.moveSnake((snakeY + 1,snakeX),False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 2:
            if snakeX < self.nColumns - 1:
                if self.screenMap[snakeY][snakeX + 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX + 1] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY,snakeX + 1),True)
                elif self.screenMap[snakeY][snakeX + 1] == 0:
                    self.moveSnake((snakeY,snakeX + 1),False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 3:
            if snakeX > 0:
                if self.screenMap[snakeY][snakeX - 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX - 1] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY,snakeX - 1),True)
                elif  self.screenMap[snakeY][snakeX - 1] == 0:
                    self.moveSnake((snakeY,snakeX - 1),False)
            else:
                gameOver = True
                reward = self.negReward
        
        self.drawScreen()
        self.lastMove = action
        pg.time.wait(self.waitTime)
        return self.screenMap,reward,gameOver

    def reset(self):
        self.screenMap = np.zeros((self.nRows,self.nColumns))
        self.snakePos = list()
        
        for i  in range(self.initSnakelen):
            self.snakePos.append((int(self.nRows/2)+i,int(self.nColumns/2)))
            self.screenMap[int(self.nRows/2)+i][int(self.nColumns/2)] = 0.5
        self.screenMap[self.applePos[0]][self.applePos[1]] = 1
        self.lastMove = 0


# In[3]:


# DNN
class Brain():
    def __init__(self,iS=(100,100,3),lr=0.0005):
        self.learning_rate = lr
        self.input_shape = iS
        self.numOutput = 4
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64,(2,2),activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256,activation='relu'),
            tf.keras.layers.Dense(units=self.numOutput)
        ])
        self.model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

    def loadModel(self,filepath):
        self.model = tf.keras.models.load_model(filepath)
        return self.model


# In[4]:


# Agente
class DQN(object):
    def __init__(self,max_memory=100,disount=0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = disount
    
    def remember(self,transition,game_over):
        self.memory.append([transition,game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
    
    def get_batch(self,model,batch_size=10):
        len_memory = len(self.memory)
        inputs = np.zeros((min(len_memory,batch_size),self.memory[0][0][0].shape[1],self.memory[0][0][0].shape[2],self.memory[0][0][0].shape[3]))
        num_ouput = model.output_shape[-1]
        targets = np.zeros((min(len_memory,batch_size),num_ouput))
        
        for i,dx in enumerate(np.random.randint(0,len_memory,size=min(len_memory,batch_size))):
            current_state,action,reward,next_state = self.memory[dx][0]
            game_over = self.memory[dx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sam = np.max(model.predict(next_state)[0])
            if game_over :
                targets[i,action] = reward
            else:
                targets[i,action] = reward + self.discount*Q_sam
            
        return inputs,targets
    


# In[5]:


# Training the AI
menSize = 60000
batchsize = 32
learningrate = 0.0001
gamma = 0.9
nLastStates = 4
epsilon = 1.
epsilon_decay = 0.0002
min_epsilon = 0.05

env = Enviroment(0)
brain = Brain((env.nRows,env.nColumns,nLastStates),learningrate)
model = brain.model
dqn = DQN(menSize,gamma)

def resetState():
    currentState = np.zeros((1,env.nRows,env.nColumns,nLastStates))

    for i in range(nLastStates):
        currentState[:,:,:,i] = env.screenMap
    
    return  currentState,currentState

epoch = 0
scores = list()
maxNCollected = 0
nCollected = 0.
totNCollected =  0

while True:
    env.reset()
    currentState,nextState = resetState()
    epoch += 1
    gameOver = False

    while not gameOver:
        if np.random.rand() < epsilon:
            action = np.random.randint(0,4)
        else:
            qvalue = model.predict(currentState)[0]
            action = np.argmax(qvalue)
        state,reward,gameOver = env.step(action)
        state = np.reshape(state,(1,env.nRows,env.nColumns,1))
        nextState = np.append(nextState,state,axis=3)
        nextState = np.delete(nextState,0,axis=3)
        dqn.remember([currentState,action,reward,nextState],gameOver)
        inputs,targats = dqn.get_batch(model,batchsize)
        model.train_on_batch(inputs,targats)
        if env.collected:
            nCollected += 1
        currentState = nextState
    if nCollected > maxNCollected and nCollected > 2:
        maxNCollected = nCollected
    totNCollected += nCollected
    nCollected = 0

    if epoch % 100 == 0 and epoch != 0 :
        scores.append(totNCollected/100)
        totNCollected = 0
        plt.plot(scores)
        plt.xlabel('Epoch / 100')
        plt.ylabel('Average Score')
        plt.close()
    
    if epsilon > min_epsilon:
        epsilon -= epsilon_decay
    
    print('Epoch: '+str(epoch)+ ' Current  Best: '+str(maxNCollected)+' Epsilon: {:.5f}'.format(epsilon))


# In[ ]:




