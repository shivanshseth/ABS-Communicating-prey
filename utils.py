import numpy as np
from params import *

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
