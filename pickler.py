import numpy as np
import pickle

if __name__ == "__main__":
    coulomb_ptr = open('coulomb.txt','r')
    coulombs = []
    this_coulomb = []
    for line in coulomb_ptr:
        if line not in ['\n', '\r\n']:
            this_line = list(map(float, line.split()))
            this_coulomb.append(this_line)
        else:
            coulombs.append(this_coulomb)
            this_coulomb = []
    #print(coulombs)
    coulomb_ptr.close()
    with open('coulombs.pickle','wb') as pickle1:
        pickle.dump(coulombs, pickle1)
    
    properties_ptr = open('properties.txt','r')
    properties = []
    for line in properties_ptr:
        this_line = list(map(float, line.split()))
        properties.append(this_line)
    #print(properties)
    properties_ptr.close()
    with open('properties.pickle','wb') as pickle2:
        pickle.dump(properties, pickle2)
        

