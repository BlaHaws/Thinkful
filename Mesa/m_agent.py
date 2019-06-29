from mesa import Agent

class MilitaryAgent(Agent):
	
	def __init__(self, unique_id, model, agent):
		super().__init__(unique_id, model)
		
		self.type = "Military"
		
	def step(self):
		pass	