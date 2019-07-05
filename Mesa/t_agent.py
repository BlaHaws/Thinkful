from mesa import Agent
from dqn_tf import DeepQNetwork

class TerroristAgent(Agent):

	def __init__(self, unique_id, model, agent, pred_model, hivemind):
		super().__init__(unique_id, model)
		
		self.hivemind = hivemind
		self.pred_model = pred_model
		self.age = int(agent.age)
		self.gender = int(agent.gender)
		self.religion = int(agent.religion)
		self.agr_bhv = float(agent.agr_bhv)
		self.rel_fnt = float(agent.rel_fnt)
		self.rel_conv = float(agent.rel_conv)
		self.hst_twd_for = float(agent.hst_twd_for)
		self.lvl_rct_act = float(agent.lvl_rct_act)
		self.crt_agr_lvl = float(agent.crt_agr_lvl)
		self.prob_threat = 0
		self.type = 'Terrorist'
		self.state = [self.gender, self.religion, self.agr_bhv, self.rel_fnt, self.rel_conv,
						self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl, self.model.score]

	def step(self):
		self.grow()
		self.choose_action(self.hivemind.choose_action(self.state))
            
	def grow(self):
		if((self.agr_bhv >= .75) or (self.rel_fnt >= .75) or (self.hst_twd_for >= .75) or (self.crt_agr_lvl >= .65)):
			self.crt_agr_lvl += .005
		if((self.agr_bhv <= .25) or (self.rel_fnt <= .25) or (self.hst_twd_for <= .25) or (self.crt_agr_lvl <= .25)):
			self.crt_agr_lvl -= .005
		if((self.agr_bhv >= .75) and ((self.rel_fnt > .75) or (self.hst_twd_for) >= .75)):
			self.crt_agr_lvl += .05
		if((self.agr_bhv <= .25) and ((self.rel_fnt < .25) or (self.hst_twd_for) <= .25)):
			self.crt_agr_lvl +- .05
            
		self.prob_threat = float(self.pred_model.predict([[self.age, self.gender, self.religion, self.agr_bhv, self.rel_fnt,
                                                self.rel_conv, self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl]]))

	def update_state(self):
		self.state = [self.gender, self.religion, self.agr_bhv, self.rel_fnt, self.rel_conv,
						self.hst_twd_for, self.lvl_rct_act, self.crt_agr_lvl, self.model.score]
	
	def choose_action(self, action):
		self.action = action
		if self.action == 1:
			print('We did a thing!')
			old_score = self.model.score
			self.model.score += 1
			new_score = self.model.score
			if old_score >= new_score:
				reward = 1
			else:
				reward = -1
			old_state = self.state
			self.update_state()
			new_state = self.state
			self.hivemind.store_transition(old_state, self.action, reward, new_state)
			agent.learn()
		elif self.action == 2:
			print('Choice 2')
		elif self.action == 3:
			print('Choice 3')
		elif self.action == 4:
			print('Choice 4')
		elif self.action == 5:
			print('Choice 5')

	def aggr_action(self):
		pass
    
	def convert(self):
		pass