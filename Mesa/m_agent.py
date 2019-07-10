from mesa import Agent
import numpy as np

class MilitaryAgent(Agent):
	
	def __init__(self, unique_id, model, agent):
		super().__init__(unique_id, model)
		
		self.wounded = False
		self.wounded_count = 0
		self.state = [self.model.terror_score, self.model.civilian_score,
					self.model.get_agent_count('Terrorist'), self.model.get_agent_count('Civilian'),
					self.model.get_agent_count('Military')]
		self.type = "Military"
		
	def step(self):
		if not self.wounded:
			self.choose_action(self.model.m_hive.choose_action(np.expand_dims(np.array(self.state).reshape((1, 5, 1)), 1)))
		else:
			if self.wounded_count > 0:
				self.wounded_count -= 1
			else:
				self.wounded = False
		self.model.m_hive.learn()
		self.model.m_gamma = self.model.m_hive.gamma
		
	def choose_action(self, action):
		self.action = action
		if self.action == 0:
			state = np.array(self.state).reshape((1, 5, 1))
			c_score = self.model.civilian_score
			agents = self.model.get_agent_list('Terrorist')
			if len(agents) > 0:
				nearest = self.model.find_nearest_agent(self, agents)
				x, y = self.model.move_toward_nearest(self, nearest)
				self.model.grid.move_agent(self, (self.pos[0]+x, self.pos[1]+y))
			self.model.set_terror_score()
			self.model.set_civil_score()
			c_score_ = self.model.civilian_score
			state_ = np.array([self.model.terror_score,	self.model.civilian_score,
						self.model.get_agent_count('Terrorist'), self.model.get_agent_count('Civilian'),
						self.model.get_agent_count('Military')])
			self.state = state_
			state_ = state_.reshape((1, 5, 1))
			if c_score >= c_score_:
				reward = -1
			else:
				reward = 1
			self.model.m_hive.store_transition(state, action, reward, state_)
			'''
			Agent find nearest terrorist agent and moves toward.
			'''
		elif self.action == 1:
			state = np.array(self.state).reshape((1, 5, 1))
			c_score = self.model.civilian_score
			reward = 0
			agents = self.model.get_same_square_type_agents(self.pos[0], self.pos[1], 'Terrorist')
			if len(agents) > 0:
				selected_agent = np.random.choice(agents)
				rand = np.random.random()
				if rand <= 0.4:
					self.model.schedule.remove(selected_agent)
					reward += 2
			self.model.set_terror_score()
			self.model.set_civil_score()
			c_score_ = self.model.civilian_score
			state_ = np.array([self.model.terror_score,	self.model.civilian_score,
						self.model.get_agent_count('Terrorist'), self.model.get_agent_count('Civilian'),
						self.model.get_agent_count('Military')])
			self.state = state_
			state_ = state_.reshape((1, 5, 1))
			if c_score >= c_score_:
				reward = -1
			else:
				reward = 1
			self.model.m_hive.store_transition(state, action, reward, state_)
			'''
			Agent randomly chooses a terrorist agent within the same square
			40% success rate, reward is 2-3x larger
			'''
		elif self.action == 2:
			reward = 0
			state = np.array(self.state).reshape((1, 5, 1))
			c_score = self.model.civilian_score
			ter_neighbors = self.model.get_neighbor_type(self, 'Terrorist')
			civ_neighbors = self.model.get_neighbor_type(self, 'Civilian')
			if len(ter_neighbors) > 0:
				choice = np.random.choice(ter_neighbors)
				rand = np.random.random()
				if rand >= 0.6:
					choice.wounded = True
					choice.wounded_count = 3
				elif rand <= 0.4 and rand > 0.05:
					self.model.schedule.remove(choice)
				elif rand <= 0.05:
					if len(civ_neighbors) > 0:
						choice2 = np.random.choice(civ_neighbors)
						self.model.schedule.remove(choice2)
						reward -= 1
			self.model.set_terror_score()
			self.model.set_civil_score()
			c_score_ = self.model.civilian_score
			state_ = np.array([self.model.terror_score,	self.model.civilian_score,
						self.model.get_agent_count('Terrorist'), self.model.get_agent_count('Civilian'),
						self.model.get_agent_count('Military')])
			self.state = state_
			state_ = state_.reshape((1, 5, 1))
			if c_score >= c_score_:
				reward += -1
			else:
				reward += 1
			self.model.m_hive.store_transition(state, action, reward, state_)
			'''
			Agent randomly selects a terrorist agent within 1 square and attacks.
			60% to wound, 40% to kill.
			5% to kill civilian agent
			'''