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
from dqn_tf import DeepQNetwork
from hivemind_ter import HiveMindTer
from hivemind_mil import HiveMindMil
from gen_agents import GenAgents

warnings.filterwarnings("ignore", category=FutureWarning)

class MapModel(Model):
	
	def __init__(self, density=.1, height=50, width=50, map_size="Large", troop_size=10000,
				t_hive=HiveMindTer(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(1, 9, 1),
								n_actions=4, mem_size=4000, batch_size=1),
				m_hive=HiveMindMil(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(1, 9, 1),
								n_actions=3, mem_size=4000, batch_size=1)):

		self.height = height
		self.width = width
		self.density = density
		self.map_size = map_size
		self.gen_agents = GenAgents()
		
		self.schedule = RandomActivation(self)
		self.grid = MultiGrid(height, width, False)
		
		self.terror_score = 0
		self.civilian_score = 0
		
		self.pred_agents = self.gen_agents.generate_pred_agents(10000)
		self.pred_model = self.train_model(self.pred_agents)
		self.t_hive = t_hive
		self.m_hive = m_hive
		
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
			self.metro_t_agents = self.gen_agents.generate_ter_agents(self.metro_ter)
			self.metro_c_agents = self.gen_agents.generate_civ_agents(self.metro_civ)
			self.city1_t_agents = self.gen_agents.generate_ter_agents(self.city_ter)
			self.city1_c_agents = self.gen_agents.generate_civ_agents(self.city_civ)
			self.city2_t_agents = self.gen_agents.generate_ter_agents(self.city_ter)
			self.city2_c_agents = self.gen_agents.generate_civ_agents(self.city_civ)
			self.village1_t_agents = self.gen_agents.generate_ter_agents(self.village_ter)
			self.village1_c_agents = self.gen_agents.generate_civ_agents(self.village_civ)
			self.village2_t_agents = self.gen_agents.generate_ter_agents(self.village_ter)
			self.village2_c_agents = self.gen_agents.generate_civ_agents(self.village_civ)
			self.village3_t_agents = self.gen_agents.generate_ter_agents(self.village_ter)
			self.village3_c_agents = self.gen_agents.generate_civ_agents(self.village_civ)
			self.basecamp_agents = self.gen_agents.generate_mil_agents(self.basecamp)
			self.outpost_agents = self.gen_agents.generate_mil_agents(self.outpost)
			for x in range(len(self.metro_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.metro_t_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.metro_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.metro_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.city1_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city1_t_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city1_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.city1_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city2_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city2_t_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city2_loc["X"], self.city2_loc["Y"]))
			for x in range(len(self.city2_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.city2_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city2_loc["X"], self.city2_loc["Y"]))
			for x in range(len(self.village1_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.village1_t_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village1_loc["X"], self.village1_loc["Y"]))
			for x in range(len(self.village1_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.village1_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village1_loc["X"], self.village1_loc["Y"]))
			for x in range(len(self.village2_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.village2_t_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village2_loc["X"], self.village2_loc["Y"]))
			for x in range(len(self.village2_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.village2_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.village2_loc["X"], self.village2_loc["Y"]))
			for x in range(len(self.village3_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.village3_t_agents[x:x+1])
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
			del self.metro_c_agents
			del self.metro_t_agents
			del self.city1_c_agents
			del self.city1_t_agents
			del self.city2_c_agents
			del self.city2_t_agents
			del self.village1_c_agents
			del self.village1_t_agents
			del self.village2_c_agents
			del self.village2_t_agents
			del self.village3_c_agents
			del self.village3_t_agents
			del self.basecamp
			del self.outpost
			del self.basecamp_agents
			del self.outpost_agents
			
		elif self.map_size == "Medium":
			#self.height = 50
			#self.width = 50
			self.metro_loc = {"X": 25, "Y": 25}
			self.city1_loc = {"X": 20, "Y": 20}
			self.city2_loc = {"X": 45, "Y": 25}
			self.basecamp_loc = {"X": 45, "Y": 10}
			self.metro_t_agents = self.gen_agents.generate_ter_agents(self.metro_ter)
			self.metro_c_agents = self.gen_agents.generate_civ_agents(self.metro_civ)
			self.city1_t_agents = self.gen_agents.generate_ter_agents(self.city_ter)
			self.city1_c_agents = self.gen_agents.generate_civ_agents(self.city_civ)
			self.city2_t_agents = self.gen_agents.generate_ter_agents(self.city_ter)
			self.city2_c_agents = self.gen_agents.generate_civ_agents(self.city_civ)
			self.basecamp_agents = self.gen_agents.generate_mil_agents(self.troop_size)
			for x in range(len(self.metro_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.metro_t_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.metro_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.metro_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.metro_loc["X"], self.metro_loc["Y"]))
			for x in range(len(self.city1_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city1_t_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city1_c_agents)):
				a = CivilianAgent('c'+str(x), self, self.city1_c_agents[x:x+1])
				self.schedule.add(a)
				self.grid.place_agent(a, (self.city1_loc["X"], self.city1_loc["Y"]))
			for x in range(len(self.city2_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.city2_t_agents[x:x+1])
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
				
			del self.metro_c_agents
			del self.metro_t_agents
			del self.city1_c_agents
			del self.city1_t_agents
			del self.city2_c_agents
			del self.city2_t_agents
			del self.basecamp
			del self.basecamp_agents
			
		elif self.map_size == "Small":
			#self.height = 20
			#self.width = 20
			self.metro_loc = {"X": 25, "Y": 25}
			self.basecamp_loc = {"X": 45, "Y": 10}
			self.metro_t_agents = self.gen_agents.generate_ter_agents(self.metro_ter)
			self.metro_c_agents = self.gen_agents.generate_civ_agents(self.metro_civ)
			self.basecamp_agents = self.gen_agents.generate_mil_agents(self.troop_size)
			for x in range(len(self.metro_t_agents)):
				a = TerroristAgent('t'+str(x), self, self.metro_t_agents[x:x+1])
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
				
			del self.metro_c_agents
			del self.metro_t_agents
			del self.basecamp
			del self.basecamp_agents
		
		del self.pred_agents
		del self.metro_size
		del self.metro_civ
		del self.metro_ter
		del self.city_civ
		del self.city_ter
		del self.village_civ
		del self.village_ter
		
		self.set_terror_score()
		self.set_civil_score()

		self.running = True

	def step(self):
		self.schedule.step()
		
	def get_agent_count(self, type):
		count = 0
		for agent in self.schedule.agents:
			if agent.type == type:
				count += 1

		return count
		
	def get_agent_list(self, type):
		agents = []
		for agent in self.schedule.agents:
			if agent.type == type:
				agents.append(agent)
				
		return agents

	def set_terror_score(self):
		t_count = self.get_agent_count('Terrorist')
		c_count = self.get_agent_count('Civilian')
		m_count = self.get_agent_count('Military')
		
		if t_count >= c_count:
			self.terror_score = t_count - (c_count/2) - m_count
		else:
			self.terror_score = t_count - c_count - m_count
			
	def set_civil_score(self):
		t_count = self.get_agent_count('Terrorist')
		c_count = self.get_agent_count('Civilian')
		m_count = self.get_agent_count('Military')
		
		self.civil_score = c_count + (m_count / 2) - t_count
		
	def get_same_square_agents(self, x_pos, y_pos):
		agents = []
		for agent in self.schedule.agents:
			if agent.pos[0] == x_pos and agent.pos[1] == y_pos:
				agents.append(agent)

		return agents

	def train_model(self, agents):
		rfg = ensemble.RandomForestRegressor()
		X = agents.drop(['prob_threat'], 1)
		Y = agents.prob_threat

		rfg.fit(X, Y)

		return rfg