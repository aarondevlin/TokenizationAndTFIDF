"""
PageRank Implementation
Aaron Devlin

INPUT: graph in text file form with 3 values repeated: node1 node2 edge=1
OUTPUT: PageRank vector that contains the probability of landing on each page
"""

import numpy, pandas

#break list into multiples of 3 items
def breaklist(theList):
    for i in range(0, len(theList), 3):
        yield theList[i:i+3]


""" Create a list that contains graph node information """
graph = open('./graph.txt', 'r')
text = graph.read()
graph.close()
tkns = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').split()
brokenlist = breaklist(tkns)
brokenlist = list(brokenlist)


""" Store out-degree edges for unique nodes """
outdegree = {} #a dictionary object
outdegreedup = {} #duplicates
for i in brokenlist:
    node = i[0]
    if node not in outdegree:
        outdegree[node] = 0
    if i[1] not in outdegree: # for the graph.txt files where only 1 is shown
        outdegree[i[1]] = 0
    if i[2] == "1": #is there an outdegree edge?

        if node + "->" + i[1] not in outdegreedup: #k=1 dictionary
            outdegree[node] += 1
            outdegreedup[node+"->"+i[1]] = 1


""" Visualize the Graph.txt File as a Matrix """
graphdataframe = pandas.DataFrame(columns=outdegree.keys(), index=outdegree.keys(), dtype="double").fillna(0)
for i in brokenlist:
    if i[2] == "1":
        graphdataframe.at[i[1],i[0]] = 1 / outdegree[i[0]]
print("---------------------------------------------------\n")
print("Graph Matrix:","\n\n", graphdataframe,"\n")
graphmatrix = graphdataframe.as_matrix() #create a numpy matrix


""" Create Original Rank Vector (rj) """
rj = []
for i in range(0, len(outdegree)):
    rj.append(1 / len(outdegree))
rjvec = numpy.matrix(rj).T
print("---------------------------------------------------\n")
print("Original Rank Vector (rj):","\n\n", rjvec, "\n")


""" PageRank Iterations """
exp = 0; #exponent for iterations
lastrvec = rjvec
currentrvec = numpy.empty(shape=(len(outdegree),1))  #(n,1) format for r vectors: n rows, one col
currentrvec.fill(-1) #start while loop with currentrvec values of -1
damp = .85  #damping factor to account for jumping to a new page that is unconnected to the user's current page.
jumpprob = (1- damp) / len(outdegree)  #probability that the user will jump

while(numpy.allclose(lastrvec, currentrvec, rtol=1e-06, atol=1e-06) == 0): #lastrvec and currentrvec are compared up to 6 decimal places
    lastrvec = rjvec
    exp += 1
    currentrvec = damp * graphmatrix * rjvec  +  jumpprob

    #print("---------------------------------------------------\n")  #uncomment these two lines to see each iteration of the process
    #print("currentrvec = .85 * matrix^", exp, " * rjvec", " + jump prob:\n\n", currentrvec, "\n", sep='')
    rjvec = currentrvec

print("---------------------------------------------------\n")
print("Converged Rank Vector (R):\n\n", rjvec, "\n")
print("---------------------------------------------------\n")
print("Number of Iterations Until Convergence:\n\n", exp, "\n")
print("---------------------------------------------------\n")
