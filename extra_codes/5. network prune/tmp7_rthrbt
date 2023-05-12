import time as t

net = open('Parsed_Omnipath_symbol_Final_withoutBlank.txt')
net.readline()
network = net.readlines()

GRN = open('GRNSN_Final.txt')
GRN.readline()
GRNSN = GRN.readlines()

result = open('Trametinib_Omnipath_symbol.txt','w')
print('Source\tTarget\tSign\tNetwork',file=result)

count = 1

for i in network:
    start = t.time()
    nline = i.split()
    for j in GRNSN:
        gline = j.split()
        if (nline[0] == gline[0]) and (nline[1] == gline[1]):
            print(j,file=result)
            print("File: Omni_Sign, Cycle #:{}, Running Time:{}".format(count,t.time()-start))
            count += 1

result.close()
GRN.close()
net.close()

Coderesult = open('Trametinib_Omnipath_symbol.txt')
Coderesult.readline()
DelBlank = set(Coderesult.readlines())
resultFile = open('Trametinib_Omnipath_symbol_Final.txt','w')
print('Source\tTarget\tSign\tNetwork',file=resultFile)
resultFile.writelines(DelBlank)
resultFile.close()
Coderesult.close()