import time as t

net = open('Omnipath_parsed_unique.txt')
net.readline()
network = net.readlines()

inN = open('Input_Nodes.txt')
inN.readline()
inputNodes = inN.readlines()
inputNodes = [i.strip('\n') for i in inputNodes]

outN = open('Output_Nodes.txt')
outN.readline()
outputNodes = outN.readlines()
outputNodes = [i.strip('\n') for i in outputNodes]

result = open('Parsed_Omnipath_symbol_Final.txt','w')
print('Source\tTarget',file=result)

count = 1

for i in network:
    start = t.time()
    line = i.split()
    if (line[0] not in inputNodes) and (line[1] not in outputNodes):
        print(i,file=result)
    print("File: Omni_Prune, Cycle #:{}, Running Time:{}".format(count,t.time()-start))
    count += 1

result.close()
net.close()
inN.close()
outN.close()

Coderesult = open('Parsed_Omnipath_symbol_Final.txt')
Coderesult.readline()
DelBlank = set(Coderesult.readlines())
resultFile = open('Parsed_Omnipath_symbol_Final_withoutBlank.txt','w')
print('Source\tTarget',file=resultFile)
resultFile.writelines(DelBlank)
resultFile.close()
Coderesult.close()