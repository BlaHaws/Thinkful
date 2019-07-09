from mesa import Agent
from dqn_tf import DeepQNetwork
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
						self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl, self.model.terror_score]

	def step(self):
		self.grow()
		if not self.wounded:
			self.choose_action(self.model.t_hive.choose_action(self.state))
			#self.t_hive.learn()
		else:
			if self.wounded_count > 0:
				self.wounded_count -= 1
			else:
				self.wounded = False
            
	def grow(self):
		if((self.agr_bhv >= .75) or (self.rel_fnt >= .75) or (self.hst_twd_for >= .75) or (self.crt_agr_lvl >= .65)):
			self.crt_agr_lvl += .005
		if((self.agr_bhv <= .25) or (self.rel_fnt <= .25) or (self.hst_twd_for <= .25) or (self.crt_agr_lvl <= .25)):
			self.crt_agr_lvl -= .005
		if((self.agr_bhv >= .75) and ((self.rel_fnt > .75) or (self.hst_twd_for) >= .75)):
			self.crt_agr_lvl += .05
		if((self.agr_bhv <= .25) and ((self.rel_fnt < .25) or (self.hst_twd_for) <= .25)):
			self.crt_agr_lvl +- .05
			
		if np.random.random() <= 0.05:
			choice = np.random.choice(self.char_list)
			attr_value = getattr(self, choice)
			setattr(self, choice, attr_value * np.random.random())

		self.prob_threat = float(self.model.pred_model.predict([[self.age, self.gender, self.religion, self.agr_bhv, self.rel_fnt,
                                                self.rel_conv, self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl]]))
	
	def choose_action(self, action):
		if action == 1:
			print("Suicide Bombing")
			state = np.array(self.state).reshape((1, 9 ,1))
			t_score = self.model.terror_score
			agents = self.model.get_same_square_agents(self.pos[0], self.pos[1])
			deaths = np.array([0,1,2,3,4,5,6,7,8,9,10,25])
			choice = np.random.choice(deaths)
			self.model.schedule.remove(self)
			killed_agents = np.random.choice(agents, choice)
			for agent in killed_agents:
				self.model.schedule.remove(agent)
			self.model.set_terror_score()
			self.model.set_civil_score()
			t_score_ = self.model.terror_score
			state_ = np.array([self.gender, self.religion, 0, 0, 0, 0, 0, 0, self.model.terror_score]).reshape((1,9,1))
			if t_score >= t_score_:
				reward = -1
			else:
				reward = 1
			self.model.t_hive.store_transition(state, action, reward, state_)
			'''
			Remove this agent from the schedule
			Remove a random number of agents on this square from the schedule
			'''
		elif action == 2:
			print('Convert')
			'''
			Find a random civilian agent on the same square.
			If civilian agent rel_conv < agent rel_conv, civilian agent becomes a terrorist agent
			'''
		elif action == 3:
			print('Attack Military')
			'''
			Find a military agent within 1 square of agent.
			30% chance to wound, 20% to kill. 
			5% chance to kill civilian
			'''
		elif action == 4:
			print('Move Toward Military')
			'''
			Find the nearest military agent and move toward.
			'''