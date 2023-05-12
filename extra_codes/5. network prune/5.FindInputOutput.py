import numpy as np
import pandas as pd

net = open('Omnipath_parsed_unique.txt')
net.readline()
network = net.readlines()
Sources = []
Targets = []

for i in network:
    line = i.split()
    Sources.append(line[0])
    Targets.append(line[1])
    
SourcesU = pd.unique(Sources)
TargetsU = pd.unique(Targets)
ST = np.setdiff1d(SourcesU,TargetsU)
TS = np.setdiff1d(TargetsU,SourcesU)

result = open('Input_Nodes.txt','w')
print('InputNodes',file=result)
for i in ST:
    print(i, file = result)
result.close()

result = open('Output_Nodes.txt','w')
print('OutputNodes',file=result)
for i in TS:
    print(i, file = result)
result.close()

net.close()