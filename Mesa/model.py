import pandas as pd
import numpy as np
import warnings

from mesa import Model
from mesa.time import RandomActivation
from sklearn import ensemble
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from scipy.stats import truncnorm
from t_agent import TerroristAgent
from c_agent import CivilianAgent
from m_agent import MilitaryAgent

warnings.filterwarnings("ignore", category=FutureWarning)

class MapModel(Model):
	
	def __init__(self, density=.1, height=50, width=50, map_size="Large", troop_size=10000):
		
		self.height = height
		self.width = width
		
		self.density = density
		self.map_size = map_size
		self.schedule = RandomActivation(self)
		self.grid = MultiGrid(height, width, False)
		self.pred_agents = self.generate_pred_agents(10000)
		self.pred_model = self.train_model(self.pred_agents)
		del self.pred_agents
		
		self.metro_size = 50#000
		self.metro_civ = int(self.metro_size * (1 - self.density))
		self.metro_ter = int(self.metro_size * self.density)
		
		self.city_size = 10#00
		self.city_civ = int(self.city_size * (1 - self.density))
		self.city_ter = int(self.city_size * self.density)
		
		self.village = 5#0
		self.village_civ = int(self.village * (1 - self.density))
		self.village_ter = int(self.village * self.density)
		
		self.troop_size = troop_size
		
		if self.map_size == "Large":
			#self.height = 50
			#self.width = 50
			self.basecamp = int(self.troop_size * .8)
			self.outpost = int(self.troop_size * .2)
			self.metro_loc = {"X": 25, "Y": 25}
			self.city1_loc = {"X": 20, "Y": 20}
			self.city2_loc = {"X": 45, "Y": 25}
			self.village1_loc = {"X": 10, "Y": 45}
			self.village2_loc = {"X": 15, "Y": 30}
			self.village3_loc = {"X": 45, "Y": 45}
			self.basecamp_loc = {"X": 35, "Y": 10}
			self.outpost_loc = {"X": 30, "Y": 20}
			self.metro_t_agents = self.generate_t_agents(self.metro_ter)
			self.metro_c_agents = self.generate_civ_agents(self.metro_civ)
			self.city1_t_agents = self.generate_t_agents(self.city_ter)
			self.city1_c_agents = self.generate_civ_agents(self.city_civ)
			self.city2_t_agents = self.generate_t_agents(self.city_ter)
			self.city2_c_agents = self.generate_civ_agents(self.city_civ)
			self.village1_t_agents = self.generate_t_agents(self.village_ter)
			self.village1_c_agents = self.generate_civ_agents(self.village_civ)
			self.village2_t_agents = self.generate_t_agents(self.village_ter)
			self.village2_c_agents = self.generate_civ_agents(self.village_civ)
			self.village3_t_agents = self.generate_t_agents(self.village_ter)
			self.village3_c_agents = self.generate_civ_agents(self.village_civ)
			self.basecamp_agents = self.generate_mil_agents(self.basecamp)
			self.outpost_agents = self.generate_mil_agents(self.outpost)
			for x in range(len(self.metro_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.metro_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.metro_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.metro_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.city1_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city1_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city1_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.city1_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city2_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city2_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city2_loc["X"], self.city2_loc["Y"]))
			for x in range(len(self.city2_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.city2_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city2_loc["X"], self.city2_loc["Y"]))
			for x in range(len(self.village1_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.village1_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village1_loc["X"], self.village1_loc["Y"]))
			for x in range(len(self.village1_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.village1_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village1_loc["X"], self.village1_loc["Y"]))
			for x in range(len(self.village2_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.village2_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village2_loc["X"], self.village2_loc["Y"]))
			for x in range(len(self.village2_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.village2_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village2_loc["X"], self.village2_loc["Y"]))
			for x in range(len(self.village3_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.village3_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village3_loc["X"], self.village3_loc["Y"]))
			for x in range(len(self.village3_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.village3_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village3_loc["X"], self.village3_loc["Y"]))
			for x in range(len(self.basecamp_agents)):
				a = MilitaryAgent('m'+str(x), self, self.basecamp_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.basecamp_loc["X"], self.basecamp_loc["Y"]))
			for x in range(len(self.outpost_agents)):
				a = MilitaryAgent('m'+str(x), self, self.outpost_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.outpost_loc["X"], self.outpost_loc["Y"]))		
		elif self.map_size == "Medium":
			#self.height = 50
			#self.width = 50
			self.metro_loc = {"X": 25, "Y": 25}
			self.city1_loc = {"X": 20, "Y": 20}
			self.city2_loc = {"X": 45, "Y": 25}
			self.basecamp_loc = {"X": 45, "Y": 10}
			self.metro_t_agents = self.generate_t_agents(self.metro_ter)
			self.metro_c_agents = self.generate_civ_agents(self.metro_civ)
			self.city1_t_agents = self.generate_t_agents(self.city_ter)
			self.city1_c_agents = self.generate_civ_agents(self.city_civ)
			self.city2_t_agents = self.generate_t_agents(self.city_ter)
			self.city2_c_agents = self.generate_civ_agents(self.city_civ)
			self.basecamp_agents = self.generate_mil_agents(self.troop_size)
			for x in range(len(self.metro_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.metro_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.metro_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.metro_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.city1_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city1_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city1_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.city1_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city2_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city2_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city2_loc["X"], self.city2_loc["Y"]))
			for x in range(len(self.city2_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.city2_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city2_loc["X"], self.city2_loc["Y"]))
			for x in range(len(self.basecamp_agents)):
				a = MilitaryAgent('m'+str(x), self, self.basecamp_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.basecamp_loc["X"], self.basecamp_loc["Y"]))
		elif self.map_size == "Small":
			#self.height = 20
			#self.width = 20
			self.metro_loc = {"X": 10, "Y": 10}
			self.basecamp_loc = {"X": 15, "Y": 5}
			self.metro_t_agents = self.generate_t_agents(self.metro_ter)
			self.metro_c_agents = self.generate_civ_agents(self.metro_civ)
			self.basecamp_agents = self.generate_mil_agents(self.troop_size)
			for x in range(len(self.metro_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.metro_t_agents[x:x+1], self.pred_model)
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.metro_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.metro_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.basecamp_agents)):
				a = MilitaryAgent('m'+str(x), self, self.basecamp_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.basecamp_loc["X"], self.basecamp_loc["Y"]))
		
		self.running = True
		
		
	def step(self):
		self.schedule.step()


	def train_model(self, agents):
		rfg = ensemble.RandomForestRegressor()
		X = agents.drop(['prob_threat'], 1)
		Y = agents.prob_threat

		rfg.fit(X, Y)

		return rfg


	def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
		return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

	def generate_t_agents(self, agent_size):
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
		agr_bhv = X1.rvs(int(agent_size))
		rel_fnt = X1.rvs(int(agent_size))
		rel_conv = X1.rvs(int(agent_size))
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