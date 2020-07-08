import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from params import *
from Agents import *

from utils import *
from WhaleModel import *
# VISUALIZATION

def agent_portrayal(agent):
    portrayal = {
                        "Shape": "arrowHead",
                        "Filled": "true",
                        "scale": 1,
                        "heading_x": int(agent.face[0]),
                        "heading_y": int(agent.face[1]),
                        "Color": "blue",
                        "Layer": 0
                }
    if type(agent) is Predator: 
        portrayal["Color"] = "red"
        portrayal["Layer"] = 1
    
    if not agent.alive: 
        portrayal["Color"] = "grey"
    return portrayal

grid = CanvasGrid(agent_portrayal, width, height, 800, 500)
server = ModularServer(WhaleModel,
                       [grid],
                       "Whale Model",
                       {"N": n_whales, "M": n_predators, "width":width, "height":height})
server.port = 8520 # The default
server.launch()