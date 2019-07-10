from mesa import Agent
import numpy as np

class TerroristAgent(Agent):

	def __init__(self, unique_id, model, agent):
		super().__init__(unique_id, model)
		
		self.wounded = False
		self.wounded_count = 0
		self.age = int(agent.age)
		self.gender = int(agent.gender)
		self.religion = int(agent.religion)
		self.char_list = ['agr_bhv', 'rel_fnt', 'rel_conv', 'hst_twd_for', 'lvl_rct_act', 'crt_agr_lvl']
		self.agr_bhv = float(agent.agr_bhv)
		self.rel_fnt = float(agent.rel_fnt)
		self.rel_conv = float(agent.rel_conv)
		self.hst_twd_for = float(agent.hst_twd_for)
		self.lvl_rct_act = float(agent.lvl_rct_act)
		self.crt_agr_lvl = float(agent.crt_agr_lvl)
		self.prob_threat = 0
		self.type = 'Terrorist'
		self.state = [self.gender, self.religion, self.agr_bhv, self.rel_fnt, self.rel_conv,
						self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl, self.model.terror_score,
						self.model.civilian_score, self.model.get_agent_count('Terrorist'), 
						self.model.get_agent_count('Civilian'), self.model.get_agent_count('Military')]

	def step(self):
		self.grow()
		if not self.wounded:
			self.choose_action(self.model.t_hive.choose_action(np.array(self.state).reshape((1, 13, 1))))
		else:
			if self.wounded_count > 0:
				self.wounded_count -= 1
			else:
				self.wounded = False
		self.model.t_hive.learn()
		self.model.t_gamma = self.model.t_hive.gamma
            
	def grow(self):
		if((self.agr_bhv >= .75) or (self.rel_fnt >= .75) or (self.hst_twd_for >= .75) or (self.crt_agr_lvl >= .65)):
			self.crt_agr_lvl += .005
		if((self.agr_bhv <= .25) or (self.rel_fnt <= .25) or (self.hst_twd_for <= .25) or (self.crt_agr_lvl <= .25)):
			self.crt_agr_lvl -= .005
		if((self.agr_bhv >= .75) and ((self.rel_fnt > .75) or (self.hst_twd_for) >= .75)):
			self.crt_agr_lvl += .05
		if((self.agr_bhv <= .25) and ((self.rel_fnt < .25) or (self.hst_twd_for) <= .25)):
			self.crt_agr_lvl +- .05

		self.agr_bhv += 0.00001
		self.rel_fnt += 0.00001
		self.rel_conv += 0.00001
		self.hst_twd_for += 0.00001
		self.crt_agr_lvl += 0.00001
		
		if np.random.random() <= 0.05:
			choice = np.random.choice(self.char_list)
			attr_value = getattr(self, choice)
			setattr(self, choice, attr_value * np.random.random())

		self.prob_threat = float(self.model.pred_model.predict([[self.age, self.gender, self.religion, self.agr_bhv, self.rel_fnt,
                                                self.rel_conv, self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl]]))
	
	def choose_action(self, action):
		if action == 0:
			state = np.array(self.state).reshape((1, 13 ,1))
			t_score = self.model.terror_score
			agents = self.model.get_same_square_agents(self.pos[0], self.pos[1])
			deaths = np.array([1,2])#,3,4,5,6,7,8,9,10,25])
			choice = np.random.choice(deaths)
			if len(agents) > choice:
				killed_agents = np.random.choice(agents, choice, replace=False)
				for agent in killed_agents:
						self.model.schedule.remove(agent)
				self.model.schedule.remove(self)
			self.model.set_terror_score()
			self.model.set_civil_score()
			t_score_ = self.model.terror_score
			state_ = np.array([self.gender, self.religion, 0, 0, 0, 0, 0, 0, self.model.terror_score, self.model.civilian_score, self.model.get_agent_count('Terrorist'), 
						self.model.get_agent_count('Civilian'), self.model.get_agent_count('Military')])
			self.state = state_
			state_ = state_.reshape((1,13,1))
			if t_score >= t_score_:
				reward = -1
			else:
				reward = 1
			self.model.t_hive.store_transition(state, action, reward, state_)
			'''
			Remove this agent from the schedule
			Remove a random number of agents on this square from the schedule
			'''
		elif action == 1:
			state = np.array(self.state).reshape((1, 13, 1))
			t_score = self.model.terror_score
			agents = self.model.get_same_square_type_agents(self.pos[0], self.pos[1], 'Civilian')
			if len(agents) > 0:
				selected_agent = np.random.choice(agents)
				if selected_agent.rel_conv <= self.rel_conv:
					self.model.add_terrorist(selected_agent, self.pos[0], self.pos[1])
					self.model.schedule.remove(selected_agent)
			self.model.set_terror_score()
			self.model.set_civil_score()
			t_score_ = self.model.terror_score
			state_ = np.array([self.gender, self.religion, self.agr_bhv, self.rel_fnt, self.rel_conv,
						self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl, self.model.terror_score,
						self.model.civilian_score, self.model.get_agent_count('Terrorist'), 
						self.model.get_agent_count('Civilian'), self.model.get_agent_count('Military')])
			self.state = state_
			state_ = state_.reshape((1, 13, 1))
			if t_score >= t_score_:
				reward = -1
			else:
				reward = 1
			self.model.t_hive.store_transition(state, action, reward, state_)
			'''
			Find a random civilian agent on the same square.
			If civilian agent rel_conv < agent rel_conv, civilian agent becomes a terrorist agent
			'''
		elif action == 2:
			reward = 0
			state = np.array(self.state).reshape((1, 13, 1))
			t_score = self.model.terror_score
			mil_neighbors = self.model.get_neighbor_type(self, 'Military')
			civ_neighbors = self.model.get_neighbor_type(self, 'Civilian')
			if len(mil_neighbors) > 0:
				choice = np.random.choice(mil_neighbors)
				rand = np.random.random()
				if rand >= 0.7:
					choice.wounded = True
					choice.wounded_count = 3
				elif rand <= 0.2 and rand > 0.05:
					self.model.schedule.remove(choice)
					reward += 1
				elif rand <= 0.05:
					if len(civ_neighbors) > 0:
						choice2 = np.random.choice(civ_neighbors)
						self.model.schedule.remove(choice2)
			self.model.set_terror_score()
			self.model.set_civil_score()
			t_score_ = self.model.terror_score
			state_ = np.array([self.gender, self.religion, self.agr_bhv, self.rel_fnt, self.rel_conv,
						self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl, self.model.terror_score,
						self.model.civilian_score, self.model.get_agent_count('Terrorist'), 
						self.model.get_agent_count('Civilian'), self.model.get_agent_count('Military')])
			self.state = state_
			state_ = state_.reshape((1, 13, 1))
			if t_score >= t_score_:
				reward += -1
			else:
				reward += 1
			self.model.t_hive.store_transition(state, action, reward, state_)
			'''
			Find a military agent within 1 square of agent.
			30% chance to wound, 20% to kill. 
			5% chance to kill civilian
			'''
		elif action == 3:
			state = np.array(self.state).reshape((1, 13, 1))
			t_score = self.model.terror_score
			agents = self.model.get_agent_list('Military')
			nearest = self.model.find_nearest_agent(self, agents)
			x, y = self.model.move_toward_nearest(self, nearest)
			self.model.grid.move_agent(self, (self.pos[0]+x, self.pos[1]+y))
			self.model.set_terror_score()
			self.model.set_civil_score()
			t_score_ = self.model.terror_score
			state_ = np.array([self.gender, self.religion, self.agr_bhv, self.rel_fnt, self.rel_conv,
						self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl, self.model.terror_score,
						self.model.civilian_score, self.model.get_agent_count('Terrorist'), 
						self.model.get_agent_count('Civilian'), self.model.get_agent_count('Military')])
			self.state = state_
			state_ = state_.reshape((1, 13, 1))
			if t_score >= t_score_:
				reward = -1
			else:
				reward = 1
			self.model.t_hive.store_transition(state, action, reward, state_)
			'''
			Find the nearest military agent and move toward.
			'''