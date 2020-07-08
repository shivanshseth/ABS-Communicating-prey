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

class WhaleModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, M, width, height):
        self.num_whales = N
        self.num_preds = M
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
 
        # Create agents
        for i in range(self.num_whales):
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            pos = (x, y)
            face = self.random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            a = Whale(i, self, pos, face)
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))
            
        assign_vocab(self.schedule.agents)

        for i in range(self.num_whales, self.num_preds + self.num_whales):
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            pos = (x, y)
            face = self.random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            a = Predator(i, self, pos, face)
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))
        # a = Predator(1, self, (0, 0), (1, 0))
        # self.schedule.add(a)
        # self.grid.place_agent(a, (0, 0))
    def step(self):
        self.schedule.step()
