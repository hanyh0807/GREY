import numpy as np
import time as t

NodeList = open('Trametinib.txt')
# NodeList.readline()
OmnipathWithSign = open('GRNSN_Final.txt')
OmnipathWithSign.readline()
count = 1

result = open('Omnipath_parsed_network.txt','w')
print('Source\tTarget\tInteraction', file=result)

NList = NodeList.readlines()
OmniS = OmnipathWithSign.readlines()
for i in NList:
    start = t.time()
    for j in OmniS:
        mask = np.isin(i.strip(),j.split())
        if mask == True:
            print(j.strip(),file=result)
    print("Cycle #:{}, Running Time:{}".format(count,t.time()-start))
    count += 1

NodeList.close()
OmnipathWithSign.close()
result.close()
