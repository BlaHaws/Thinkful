#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.stats import truncnorm
from mesa import Agent, Model
from mesa.time import RandomActivation
from sklearn import ensemble
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

warnings.filterwarnings("ignore", category=FutureWarning)


# In[41]:


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def generate_t_agents():
    agents = pd.DataFrame()
    ages = np.round(np.random.normal(36, 5, 10000), 0)
    males = np.zeros((9000,), dtype=int)
    females = np.ones((1000,), dtype=int)
    genders = np.concatenate((males, females), axis=None)
    reli = np.zeros((8000,), dtype=int)
    relo = np.ones((2000,), dtype=int)
    religions = np.concatenate((reli, relo), axis=None)
    np.random.shuffle(genders)
    np.random.shuffle(religions)
    X1 = get_truncated_normal(mean=.25, sd=.25, low=0, upp=1)
    X2 = get_truncated_normal(mean=.5, sd=.15, low=0, upp=1)
    X3 = get_truncated_normal(mean=.35, sd=.3, low=0, upp=1)
    X4 = get_truncated_normal(mean=.2, sd=.35, low=0, upp=1)
    X5 = get_truncated_normal(mean=.5, sd=.4, low=0, upp=1)
    X6 = get_truncated_normal(mean=.15, sd=.4, low=0, upp=1)
    agr_bhv = X1.rvs(10000)
    rel_fnt = X2.rvs(10000)
    rel_conv = X6.rvs(10000)
    hst_twd_for = X3.rvs(10000)
    lvl_rct_act = X4.rvs(10000)
    crt_agr_lvl = X5.rvs(10000)
    prob_threat = np.zeros((10000,), dtype=float)

    agents['ages'] = ages.astype(int)
    agents['gender'] = genders
    agents['religion'] = religions
    agents['agr_bhv'] = agr_bhv
    agents['rel_fnt'] = rel_fnt
    agents['rel_conv'] = rel_conv
    agents['hst_twd_for'] = hst_twd_for
    agents['lvl_rct_act'] = lvl_rct_act
    agents['crt_agr_lvl'] = crt_agr_lvl
    agents['prob_threat'] = prob_threat

    return agents

def generate_pred_agents():
    agents = pd.DataFrame()

    ages = np.round(np.random.normal(36, 5, 10000), 0)
    males = np.zeros((9000,), dtype=int)
    females = np.ones((1000,), dtype=int)
    genders = np.concatenate((males, females), axis=None)
    reli = np.zeros((8000,), dtype=int)
    relo = np.ones((2000,), dtype=int)
    religions = np.concatenate((reli, relo), axis=None)
    np.random.shuffle(genders)
    np.random.shuffle(religions)
    X1 = get_truncated_normal(mean=.25, sd=.25, low=0, upp=1)
    X2 = get_truncated_normal(mean=.5, sd=.15, low=0, upp=1)
    X3 = get_truncated_normal(mean=.35, sd=.3, low=0, upp=1)
    X4 = get_truncated_normal(mean=.2, sd=.35, low=0, upp=1)
    X5 = get_truncated_normal(mean=.5, sd=.4, low=0, upp=1)
    X6 = get_truncated_normal(mean=.15, sd=.4, low=0, upp=1)
    agr_bhv = X1.rvs(10000)
    rel_fnt = X2.rvs(10000)
    rel_conv = X6.rvs(10000)
    hst_twd_for = X3.rvs(10000)
    lvl_rct_act = X4.rvs(10000)
    crt_agr_lvl = X5.rvs(10000)
    prob_threat = X2.rvs(10000)

    agents['ages'] = ages.astype(int)
    agents['gender'] = genders
    agents['religion'] = religions
    agents['agr_bhv'] = agr_bhv
    agents['rel_fnt'] = rel_fnt
    agents['rel_conv'] = rel_conv
    agents['hst_twd_for'] = hst_twd_for
    agents['lvl_rct_act'] = lvl_rct_act
    agents['crt_agr_lvl'] = crt_agr_lvl
    agents['prob_threat'] = prob_threat

    return agents

def generate_mil_agents():
    agents = pd.DataFrame()
    
    return agents

def generate_civ_agents():
    agents = pd.DataFrame()
    
    ages = np.round(np.random.normal(36, 5, 10000), 0)
    males = np.zeros((9000,), dtype=int)
    females = np.ones((1000,), dtype=int)
    genders = np.concatenate((males, females), axis=None)
    reli = np.zeros((8000,), dtype=int)
    relo = np.ones((2000,), dtype=int)
    religions = np.concatenate((reli, relo), axis=None)
    np.random.shuffle(genders)
    np.random.shuffle(religions)
    X1 = get_truncated_normal(mean=.25, sd=.3, low=0, upp=1)
    agr_bhv = X1.rvs(10000)
    rel_fnt = X1.rvs(10000)
    rel_conv = X1.rvs(10000)
    hst_twd_for = X1.rvs(10000)
    lvl_rct_act = X1.rvs(10000)
    crt_agr_lvl = X1.rvs(10000)
    
    agents['ages'] = ages.astype(int)
    agents['gender'] = genders
    agents['religion'] = religions
    agents['agr_bhv'] = agr_bhv
    agents['rel_fnt'] = rel_fnt
    agents['rel_conv'] = rel_conv
    agents['hst_twd_for'] = hst_twd_for
    agents['lvl_rct_act'] = lvl_rct_act
    agents['crt_agr_lvl'] = crt_agr_lvl
    
    return agent


# In[42]:


def train_model(agents):
    rfg = ensemble.RandomForestRegressor()
    X = agents.drop(['prob_threat'], 1)
    Y = agents.prob_threat

    rfg.fit(X, Y)

    return rfg


# In[43]:


class TerroristAgent(Agent):
    
    def __init__(self, unique_id, model, agent, pred_model):
        super().__init__(unique_id, model)
        self.pred_model = pred_model
        self.age = int(agent.ages)
        self.gender = int(agent.gender)
        self.religion = int(agent.religion)
        self.agr_bhv = float(agent.agr_bhv)
        self.rel_fnt = float(agent.rel_fnt)
        self.rel_conv = float(agent.rel_conv)
        self.hst_twd_for = float(agent.hst_twd_for)
        self.lvl_rct_act = float(agent.lvl_rct_act)
        self.crt_agr_lvl = float(agent.crt_agr_lvl)
        self.prob_threat = 0
        
    def step(self):
        if((self.agr_bhv >= .75) or (self.rel_fnt >= .75) or (self.hst_twd_for >= .75) or (self.crt_agr_lvl >= .65)):
            self.crt_agr_lvl += .005
        if((self.agr_bhv <= .25) or (self.rel_fnt >= .25) or (self.hst_twd_for <= .25) or (self.crt_agr_lvl <= .25)):
            self.crt_agr_lvl -= .005

        self.prob_threat = float(self.pred_model.predict([[self.age, self.gender, self.religion, self.agr_bhv, self.rel_fnt,
                                                self.rel_conv, self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl]]))
        
        if(self.prob_threat >= .75):
            self.aggr_action()
        else:
            self.convert()
            
        self.move()
            
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def aggr_action(self):
        pass
    
    def convert(self):
        #print('%d: %.5f' % (self.unique_id, self.prob_threat))
        pass


# In[46]:


class MapModel(Model):
    
    def __init__(self):
        self.grid = MultiGrid(100, 100, True)
        self.t_agents = generate_t_agents()
        self.pred_agents = generate_pred_agents()
        self.pred_model = train_model(self.pred_agents)
        self.schedule = RandomActivation(self)
        for x in range(len(self.t_agents)):
            a = TerroristAgent(x, self, self.t_agents[x:x+1], self.pred_model)
            self.schedule.add(a)
            
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            
    def step(self):
        self.schedule.step()


# In[ ]:




