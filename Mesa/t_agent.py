from mesa import Agent

class TerroristAgent(Agent):

	def __init__(self, unique_id, model, agent, pred_model):
		super().__init__(unique_id, model)

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

	def step(self):
		self.grow()
		self.choose_action()
            
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

	def choose_action(self):
		if(self.prob_threat >= .75):
			self.lvl_rct_act += .5
			self.aggr_action()
		else:
			self.convert()

	def aggr_action(self):
		pass
    
	def convert(self):
		pass