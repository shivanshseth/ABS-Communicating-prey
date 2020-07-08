import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

# PARAMETERS

range_eco = 20
width_eco = 10
range_signal = 10
range_pred = 30
whale_speed = 1
n_whales = 3
n_predators = 1
height = 10
width = 10
pred_speed = 1
go_come_cutoff = 0.3
MAX = 10000
alpha = 0.3

# UTILITY FUNCTIONS

def rms(word1, word2):
    ans = 0
    for i in  enumerate(word1):
        ans += (word2[i[0]] - i[1])**2
    return (ans/len(word1))**0.5 
        
def assign_vocab(agents):
    # print(type(agents))
    agentsnum = len(agents)
    #agents = list of whale agents , agents num len(list of whale agents)
    mu, sigma = 100, 70
    pitch =np.random.normal(mu,sigma,agentsnum)
    mu, sigma = 80,70
    loudness = np.random.normal(mu,sigma,agentsnum)
    mu, sigma = 30,20
    length = np.random.normal(mu,sigma,agentsnum)
    gowords = []
    otherwords = []

    for i in pitch:
        for j in loudness:
            for k in length :
                gowords.append((i,j,k))
                otherwords.append((i,j,k))

    mu, sigma = 80, 30
    pitch =np.random.normal(mu,sigma,agentsnum)
    mu, sigma = 60, 40
    loudness = np.random.normal(mu,sigma,agentsnum)
    mu, sigma = 15,10
    length = np.random.normal(mu,sigma,agentsnum)
    comewords = []
    for i in pitch:
        for j in loudness:
            for k in length :
                comewords.append((i,j,k))
                otherwords.append((i,j,k))

    idx = 0
    for i in agents:
        i.language_prob(gowords[idx],comewords[idx],otherwords)
        idx+=1

def pos_to_intensity(agent_pos, target_poss):
    pred_info = []
    for i in target_poss:
        intensity = ((i[0] - agent_pos[0])**2 + (i[1] - agent_pos[1])**2) ** 0.5 + 1/MAX
        intensity = 1/intensity
        h, v = i[0] - agent_pos[0], i[1] - agent_pos[1]
        direction = (0, 0)

        if abs(h) > abs(v):
            direction = np.array((1, 0)) if h>=0 else np.array((-1, 0))
        else:
            direction = np.array((0, 1)) if v>=0 else np.array((0, -1))

        pred_info.append((direction, intensity, i))
    return pred_info

# AGENTS

class Predator(Agent):
    def __init__(self, unique_id, model, pos, face):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.face = np.array(face) # N = (0, 1), S = (0, -1), E = (1, 0), W = (-1, 0)
        self.alive = True
        self.prey_info = None
        self.id = unique_id
 
    def move(self):
        try: 
            # print(self.prey_info, self.pos)
            self.face = self.prey_info[0]
        except:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.face = np.array(dirs[self.random.choice([0, 1, 2, 3])])
        new_position = tuple(self.pos + pred_speed * self.face)
        if new_position[0] < 0 or new_position[0] >= width or new_position[1] < 0 or new_position[1] >=height:
            # self.face = self.face * -1
            # new_position = tuple(self.pos + pred_speed * self.face)
            new_position = tuple(self.pos)
        self.model.grid.move_agent(self, new_position)
    
    def set_closest_prey(self, prey_coords):
        closest_prey = None
        for i in prey_coords:
            if not closest_prey:
                closest_prey = i
            if i[1] > closest_prey[1]:
                closest_prey = i
        self.prey_info = closest_prey

    def smell_prey(self):
        """ Search for whales in the all direction in its range (smell) """
        area_scanned = []
        x, y = self.pos
        prey_coords = []

        for r in range(-1 * (range_pred//2), range_pred//2 + 1): 
            for w in range (-1 * (range_pred//2), range_pred//2 + 1):
                coords = (x + w, y + r)
                if not (coords[0] < 0 or coords[0] >= width or coords[1] < 0 or  coords[1] >=height):
                    area_scanned.append(coords)

        for cell in area_scanned:
            try:
                cellmates = self.model.grid.get_cell_list_contents([cell])
                for i in cellmates: 
                    if type(i) is Whale and i.alive: prey_coords.append(cell)
            except: 
                pass

        if prey_coords:
            prey_coords = pos_to_intensity(self.pos, prey_coords)
            self.set_closest_prey(prey_coords) 
            return prey_coords
        return False
    
    def eat(self):
        """ Eat any(1) whale on the same cell"""
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for i in cellmates: 
            if type(i) is Whale: 
                i.alive = False
                self.prey_info = None
                break    

    def step(self):
        self.eat()
        self.smell_prey()
        self.move()
 
class Whale(Agent):
    """ A whale agent, which can use echolocation to detect objects and send signals to other whales. """
    def __init__(self, unique_id, model, pos, face):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.pos = np.array(pos)
        self.face = np.array(face) # N = (0, 1), S = (0, -1), E = (1, 0), W = (-1, 0)
        self.alive = True
        self.preds_near = []
        self.pred_info = None # (dir, intensity)
        self.go = {} #dictionary that corresponds with probability of words with word go
        self.come = {} #dictionary that corresponds with probability of words with come
 
    def move(self):
        try: 
            # print(self.pred_info, self.pos)
            self.face = self.pred_info[0] * -1
        except:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.face = np.array(dirs[self.random.choice([0, 1, 2, 3])])
        new_position = tuple(self.pos + whale_speed * self.face)
        if new_position[0] < 0 or new_position[0] >= width or new_position[1] < 0 or  new_position[1] >=height: 
            # self.face[0] = self.face[1]
            # self.face[1] = self.face[0]
            # new_position = tuple(self.pos + whale_speed * self.face)
            new_position = tuple(self.pos)
        self.model.grid.move_agent(self, new_position)
    
    def set_closest_pred(self, pred_coords):
        closest_pred = None
        for i in pred_coords:
            if not closest_pred:
                closest_pred = i
            if i[1] > closest_pred[1]:
                closest_pred = i
        self.pred_info = closest_pred
 
    def echolocation(self): 
        """ Send a signal in the direction it is currently facing. 
            Check for predators or food in that direction. 
            Returns the location of predator if found. """
        area_scanned = []
        x, y = self.pos
        dx, dy = self.face
        pred_coords = []
        for r in range(1, range_eco + 1): 
            for w in range (-1 * (width_eco//2), width_eco//2 + 1):
                if dx == 0: 
                    coords = (x + w, y + dy * r)
                else: 
                    coords = (x + dx * r, y + w)
                if not (coords[0] < 0 or coords[0] >= width or coords[1] < 0 or  coords[1] >=height):
                    area_scanned.append(coords)

        for cell in area_scanned:
            try:
                cellmates = self.model.grid.get_cell_list_contents([cell])
                for i in cellmates: 
                    if type(i) is Predator: pred_coords.append(cell)
            except: 
                pass

        if pred_coords:
            pred_coords = pos_to_intensity(self.pos, pred_coords)
            self.set_closest_pred(pred_coords) 
            return pred_coords
        return False
 
    def alert(self): 
        area_scanned = []
        x, y = self.pos
        goword = max(self.go)
        comeword = max(self.come)
        call = goword if (self.pred_info[1] > go_come_cutoff) else comeword

        for r in range(-1 * (range_signal//2), range_signal//2 + 1): 
            for w in range (-1 * (range_signal//2), range_signal//2 + 1):
                coords = (x + w, y + r)
                if not (coords[0] < 0 or coords[0] >= width or coords[1] < 0 or  coords[1] >=height):
                    area_scanned.append(coords)
        for cell in area_scanned:
            try:
                cellmates = self.model.grid.get_cell_list_contents([cell])
                for i in cellmates: 
                    if type(i) is Whale and i.alive: 
                        # print(type(i), cell)
                        i.recieve_signal(call, (x, y))
            except: 
                pass

    def recieve_signal(self, signal, origin):
        [call_info] = pos_to_intensity(self.pos, [origin])
        temp = self.face
        print(self.come[signal] < self.go[signal])
        if self.come[signal] < self.go[signal]:
            print("GO from ", origin, "to ", self.pos)
            # check in the direction of the signal
            self.face = call_info[0]
            pred = self.echolocation()
            if pred:
                self.preds_near += pred
                self.go[signal] += alpha*self.go[signal]
                if self.go[signal] > 1: self.go[signal] = 1 
                self.come[signal] -= alpha*self.come[signal]
                if self.come[signal] < 0: self.come[signal] = 0
                self.face *= -1
            else: 
                self.go[signal] -= alpha*self.come[signal] 
                if self.go[signal] < 0: self.go[signal] = 0
                self.come[signal] += alpha*self.come[signal]
                if self.come[signal] > 1: self.come[signal] = 1  
        else:
            print("COME from ", origin, "to ", self.pos)
            # check opp to the direction of the signal
            self.face = -1 * call_info[0]
            pred = self.echolocation()
            if pred:    
                self.preds_near += pred
                self.go[signal] -= alpha*self.come[signal] 
                if self.go[signal] < 0: self.go[signal] = 0
                self.come[signal] += alpha*self.come[signal]
                if self.come[signal] > 1: self.come[signal] = 1
            
            else:
                self.go[signal] += alpha*self.go[signal]
                if self.go[signal] > 1: self.go[signal] = 1 
                self.come[signal] -= alpha*self.come[signal]
                if self.come[signal] < 0: self.come[signal] = 0
                self.face *= -1
        self.set_closest_pred(self.preds_near)

    def language_prob(self,goword,comeword,otherwords):
        for word in otherwords:
            self.go[word] = MAX if rms(goword, word) == 0 or 1/rms(goword, word) > MAX else 1/rms(goword, word)
            self.come[word] = MAX if rms(comeword, word) == 0 or 1/rms(comeword, word) > MAX else 1/rms(comeword, word)
        
        gmax = self.go[max(self.go)]
        cmax = self.come[max(self.come)]
        
        for i in self.go:
            self.go[i] /= gmax
        
        for i in self.come:
            self.come[i] /= cmax

        self.go[goword] = 1.0
        self.go[comeword] = 0.0
        self.come[comeword] = 1.0
        self.come[goword] = 0.0

    def step(self):
        self.pred_info = None
        if self.alive:
            self.echolocation()
            if self.pred_info:
                self.alert()
            self.move()

# MODEL

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