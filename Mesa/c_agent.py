from mesa import Agent

class CivilianAgent(Agent):

	def __init__(self, unique_id, model, agent):
		super().__init__(unique_id, model)

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
		self.type = 'Civilian'
        
	def step(self):
		pass