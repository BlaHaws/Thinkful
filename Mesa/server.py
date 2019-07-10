from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from model import MapModel

class CCountElement(TextElement):
	def __init__(self):
		pass
	
	def render(self, model):
		return "# of Civilians: " + str(model.get_agent_count('Civilian') + " Terror Score: " + str(model.terror_score))
		

class TCountElement(TextElement):
	def __init__(self):
		pass
	
	def render(self, model):
		return "# of Terrorists: " + str(model.get_agent_count('Terrorist') + " Civil Score: " + str(model.civilian_score))
		
class MCountElement(TextElement):
	def __init__(self):
		pass
	
	def render(self, model):
		return "# of Troops: " + str(model.get_agent_count('Military'))
		

def get_model_params():

	height = None
	width = None
	
	map_size = UserSettableParameter("choice", "Map size", value="Large", choices=["Small", "Medium", "Large"])
	if map_size.value == "Large":
		height = 50
		width = 50
	elif map_size.value == "Medium":
		height = 25
		width = 25
	elif map_size.value == "Small":
		height = 10
		width = 10
	density = UserSettableParameter("slider", "Terrorist density", 0.25, 0.00, 1.00, 0.25)
	troop_size = UserSettableParameter("number", "Troop size", 10000)
	
	model_params = {"height": height, "width": width, "density": density, "map_size": map_size, "troop_size": troop_size}
	#print(map_size.value)
	return model_params

def mapmodel_draw(agent):
	
	if agent is None:
		return
	portrayal = {"Shape": "circle", "r": 0.8, "Filled": "true", "Layer": 0}
	
	if agent.type == "Terrorist":
		portrayal["Color"] = ["#FF0000"]
		portrayal["stroke_color"] = "#000000"
	elif agent.type == "Civilian":
		portrayal["Color"] = ["#00ff00"]
		portrayal["stroke_color"] = "#000000"
	elif agent.type == "Military":
		portrayal["Color"] = ["#0000FF"]
		portrayal["stroke_color"] = "#000000"
	return portrayal

model_params = get_model_params()

c_count_element = CCountElement()
t_count_element = TCountElement()
m_count_element = MCountElement()

canvas_element = CanvasGrid(mapmodel_draw, model_params["height"], model_params["width"], 500, 500)
ter_gamma_chart = ChartModule([{"Label": "Terrorist Epsilon", "Color": "Red"}, {"Label": "Military Epsilon", "Color": "Blue"}])

server = ModularServer(MapModel,
						[canvas_element, c_count_element, t_count_element, m_count_element, ter_gamma_chart],
						"Terrorist Response", model_params)
						
server.launch()