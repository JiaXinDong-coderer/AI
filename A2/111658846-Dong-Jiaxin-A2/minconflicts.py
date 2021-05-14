import sys
import time
import random
import copy

answerTrue = []

class Graph:
    """we descirbing coloring map problems to graph
        in this class we have a value to keep track of number states that we explored
        a init function that can initialize colors, nodes, edges and nodeColor features
        to create this graph"""
    statenumber = 0
    def __init__(self, nodes,colors):
        """
        initialize all factors wee need to create this graph
        colors is a scalar that describe how many colors we can use
        nodes is a scalar that describe how many nodes we should have
        adj is a 2d list to describe relationships between nodes, a.k.a edges
        nodecolor is use for arc consistency, not neccessary needed

        :param nodes: number of nodes that should exist in this graph
        :param colors: how many colors that this graph can have
        """
        self.colors = colors
        self.nodes = nodes # list of nodes
        self.adj = [[] for i in range(nodes)] # adj list
        self.nodeColor = [[j for j in range(self.colors)] for i in range(self.nodes)]

    def addConstrains(self,node1,node2):
        """
        add an edge for two nodes, add in adj list with index of first node and second node
        (undirected graph)
        :param node1: first node of a edge
        :param node2: second node of a edge
        """
        self.adj[node1].append(node2)
        self.adj[node2].append(node1)

    def minConflict(self):
        """
        this is main function of min conflict algorithm
        it randomly generate a answer list and find minconflict node and colors
        and keep update answer list. if answer is correct, return it
        If function runs into a local optimum for too long, it will re-generate an random answer list
        to check again

        :return: answer list
        """
        for i in range(N):
            answer = [(x,random.randint(0,self.colors-1)) for x in range(self.nodes)]
            #       [(0, 0), (1, 1), (2, 2), (3, 3), (4, 2), (5, 0), (6, 1), (7, 1), (8, 0), (9, 3)]
            for j in range(N*90):
                endLong()
                self.statenumber+=1
                if self.isCorrect(answer):
                    return answer
                violateIndex = self.findVioletIndex(answer)
                minConfColor = self.findMinColor(violateIndex,answer)
                answer[violateIndex] = (answer[violateIndex][0],minConfColor)
        return False

    """
    below three functions are helper funtion of minconflict
    """
    def isCorrect(self,answer):
        for i in answer:
            for j in self.adj[i[0]]:
                if (j,i[1]) in answer:
                    return False
        return True

    def findVioletIndex(self,answer):
        lst = []
        for i in answer:
            for j in self.adj[i[0]]:
                if (j,i[1]) in answer:
                    lst.append(answer.index(i))
        return random.choice(lst)

    def findMinColor(self,index,answer):
        """
        this function find up a color/colors to change for the selected node
        it will try every color on this node to see which color can lead to less conflict
        and it will record all colors and then select a color randomly from the least conflict color/colors

        :param index: node to check
        :param answer: current answer list
        :return: what color to change
        """
        testAnswer = copy.deepcopy(answer)
        colorMark = {}
        for x in range(self.colors):
            testAnswer[index] = (testAnswer[index][0],x)
            for i in testAnswer:
                for j in self.adj[i[0]]:
                    if (j,i[1]) in answer:
                        if x not in colorMark:
                            colorMark[x] = 1
                        else:
                            colorMark[x] +=1
        colorMark = dict(sorted(colorMark.items(),key=lambda x:x[1]))
        minKeys = []
        lastValue = list(colorMark.values())[0]
        for k in colorMark.keys():
            if colorMark[k] > lastValue:
                break
            minKeys.append(k)

        if len(minKeys) == 1:
            return minKeys[0]
        return random.choice(minKeys)

def endLong():
    """
    this function stop program if run time is too long
    change the number if you want more/less time
    """
    currentTime = time.time()
    if currentTime-start >=65:
        print("One and half minute passed and still didn't get answer")
        print("Exiting now...")
        fout = open(output, "w")
        fout.write("No answer")
        fout.close()
        exit(-1)


# python minconflicts.py C100 C100out.txt
# python CSPGenerator.py 100 200 4 C100 0
if __name__ == '__main__':
    """
    this is main function that check argv and initialize map coloring problem and out put answer into txt
    """
    if len(sys.argv) != 3:
        print("Usage: python minconflicts.py <INPUT FILE> <OUTPUT FILE>")
        exit(-1)

    input = sys.argv[1]
    global output
    output = sys.argv[2]

    """#debug
    import os
    os.chdir("X:/JAR Work/CSE352/A2/")
    input = "N10-3"
    output = "minOut.txt"""

    # set timmer
    global start
    start = time.time()

    f = open(input,"r")
    firstLine = f.readline().replace("\n","").split()
    N,M,K = int(firstLine[0]),int(firstLine[1]),int(firstLine[2])
    problem = Graph(N,K)

    for i in range(M):
        edge = f.readline().split()
        problem.addConstrains(int(edge[0]),int(edge[1]))
    f.close()

    # to record answers
    direct = problem.minConflict()
    print(direct)

    fout = open(output, "w")
    if not direct:
        fout.write("No answer")
    else:
        for i in direct:  # check answers
            fout.write("{}\n".format(i[1]))
            # print(i[0],i[1])
    fout.close()
    end = time.time()
    #print("State Explored:",problem.statenumber)
    print("Run time is:",end-start)