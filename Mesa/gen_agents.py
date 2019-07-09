import pandas as pd
import numpy as np
from scipy.stats import truncnorm

class GenAgents(object):

	def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
			return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

	def generate_ter_agents(self, agent_size):
		agents = pd.DataFrame()
		ages = np.round(np.random.normal(36, 5, agent_size), 0)
		males = np.zeros((int(agent_size * .9),), dtype=int)
		females = np.ones((int(agent_size - len(males)),), dtype=int)
		genders = np.concatenate((males, females), axis=None)
		reli = np.zeros((int(agent_size * .8),), dtype=int)
		relo = np.ones((int(agent_size - len(reli)),), dtype=int)
		religions = np.concatenate((reli, relo), axis=None)
		np.random.shuffle(genders)
		np.random.shuffle(religions)
		X1 = self.get_truncated_normal(mean=.25, sd=.25, low=0, upp=1)
		X2 = self.get_truncated_normal(mean=.5, sd=.15, low=0, upp=1)
		X3 = self.get_truncated_normal(mean=.35, sd=.3, low=0, upp=1)
		X4 = self.get_truncated_normal(mean=.2, sd=.35, low=0, upp=1)
		X5 = self.get_truncated_normal(mean=.5, sd=.4, low=0, upp=1)
		X6 = self.get_truncated_normal(mean=.15, sd=.4, low=0, upp=1)
		agr_bhv = X1.rvs(int(agent_size))
		rel_fnt = X2.rvs(int(agent_size))
		rel_conv = X6.rvs(int(agent_size))
		hst_twd_for = X3.rvs(int(agent_size))
		lvl_rct_act = X4.rvs(int(agent_size))
		crt_agr_lvl = X5.rvs(int(agent_size))
		prob_threat = np.zeros((int(agent_size),), dtype=float)

		agents['age'] = ages.astype(int)
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

	def generate_pred_agents(self, agent_size):
		agents = pd.DataFrame()

		ages = np.round(np.random.normal(36, 5, agent_size), 0)
		males = np.zeros((int(agent_size * .9),), dtype=int)
		females = np.ones((int(agent_size * .1),), dtype=int)
		genders = np.concatenate((males, females), axis=None)
		reli = np.zeros((int(agent_size * .8),), dtype=int)
		relo = np.ones((int(agent_size * .2),), dtype=int)
		religions = np.concatenate((reli, relo), axis=None)
		np.random.shuffle(genders)
		np.random.shuffle(religions)
		X1 = self.get_truncated_normal(mean=.25, sd=.25, low=0, upp=1)
		X2 = self.get_truncated_normal(mean=.5, sd=.15, low=0, upp=1)
		X3 = self.get_truncated_normal(mean=.35, sd=.3, low=0, upp=1)
		X4 = self.get_truncated_normal(mean=.2, sd=.35, low=0, upp=1)
		X5 = self.get_truncated_normal(mean=.5, sd=.4, low=0, upp=1)
		X6 = self.get_truncated_normal(mean=.15, sd=.4, low=0, upp=1)
		agr_bhv = X1.rvs(int(agent_size))
		rel_fnt = X2.rvs(int(agent_size))
		rel_conv = X6.rvs(int(agent_size))
		hst_twd_for = X3.rvs(int(agent_size))
		lvl_rct_act = X4.rvs(int(agent_size))
		crt_agr_lvl = X5.rvs(int(agent_size))
		prob_threat = X2.rvs(int(agent_size))

		agents['age'] = ages.astype(int)
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

	def generate_mil_agents(self, agent_size):
		agents = pd.DataFrame()
		
		agents['thing'] = np.zeros((int(agent_size),), dtype=int)
		
		return agents

	def generate_civ_agents(self, agent_size):
		agents = pd.DataFrame()
		
		ages = np.round(np.random.normal(36, 5, agent_size), 0)
		males = np.zeros((int(agent_size * .9),), dtype=int)
		females = np.ones((int(agent_size - len(males)),), dtype=int)
		genders = np.concatenate((males, females), axis=None)
		reli = np.zeros((int(agent_size * .8),), dtype=int)
		relo = np.ones((int(agent_size - len(reli)),), dtype=int)
		religions = np.concatenate((reli, relo), axis=None)
		np.random.shuffle(genders)
		np.random.shuffle(religions)
		X1 = self.get_truncated_normal(mean=.25, sd=.3, low=0, upp=1)
		X2 = self.get_truncated_normal(mean=.8, sd=.3, low=0, upp=1)
		agr_bhv = X1.rvs(int(agent_size))
		rel_fnt = X1.rvs(int(agent_size))
		rel_conv = X2.rvs(int(agent_size))
		hst_twd_for = X1.rvs(int(agent_size))
		lvl_rct_act = X1.rvs(int(agent_size))
		crt_agr_lvl = X1.rvs(int(agent_size))
		prob_threat = np.zeros((int(agent_size),), dtype=float)

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