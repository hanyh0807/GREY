Network = open('Omnipath_parsed_network.txt')
Network.readline()
NetworkLinks = Network.readlines()

uniqueNetworkLinks = set(NetworkLinks)

result = open('Omnipath_parsed_unique.txt','w')
print('Source\tTarget\tInteraction', file=result)
result.writelines(uniqueNetworkLinks)

Network.close()
result.close()