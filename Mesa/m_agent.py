from mesa import Agent

class MilitaryAgent(Agent):
	
	def __init__(self, unique_id, model, agent):
		super().__init__(unique_id, model)
		
		self.wounded = False
		self.wounded_count = 0
		self.state = []
		self.type = "Military"
		
	def step(self):
		if not self.wounded:
			#self.choose_action(self.model.m_hive.choose_action(self.state))
			#self.model.m_hive.learn()
			pass
		else:
			if self.wounded_count > 0:
				self.wounded_count -= 1
			else:
				self.wounded = False
		
	def choose_action(self, action):
		self.action = action
		if self.action == 1:
			print("Move toward terrorist")
			'''
			Agent find nearest terrorist agent and moves toward.
			'''
		elif self.action == 2:
			print('Arrest terrorist')
			'''
			Agent randomly chooses a terrorist agent within the same square
			40% success rate, reward is 2-3x larger
			'''
		elif self.action == 3:
			print('Attack terrorist')
			'''
			Agent randomly selects a terrorist agent within 1 square and attacks.
			60% to wound, 40% to kill.
			5% to kill civilian agent
			'''