from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.UserParam import UserSettableParameter

from model import MapModel

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
	portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}
	
	if agent.type == "Terrorist":
		portrayal["Color"] = ["#FF0000", "#FF9999"]
		portrayal["stroke_color"] = "#00FF00"
	elif agent.type == "Civilian":
		portrayal["Color"] = ["#00ff00", "#9999FF"]
		portrayal["stroke_color"] = "#000000"
	elif agent.type == "Military":
		portrayal["Color"] = ["#0000FF", "#9999FF"]
		portrayal["stroke_color"] = "#000000"
	return portrayal

model_params = get_model_params()
	
canvas_element = CanvasGrid(mapmodel_draw, model_params["height"], model_params["width"], 500, 500)

server = ModularServer(MapModel,
						[canvas_element],
						"Terrorist Response", model_params)
						
server.launch()