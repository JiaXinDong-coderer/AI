import sys
import time

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

    def backtracking(self):
        """
        function that start backtracking algo.
        :return: result of search
        """
        return self.backtrackingRec([])

    def backtrackingRec(self,answerList):
        """
        use plain backtrack algorithm to search,
        if answer vector is correct answer, we output it, other wise we keep finding candidates
        and loop through all actions it can use

        if version is specified as 1, then prune the candidate actions before run into looping action step
        :param answerList: answer vector use to keep answers in each recursion
        :return:
        """
        endLong()
        self.statenumber+=1
        answer = answerList.copy()
        if len(answer) >= self.nodes:
            answerTrue.append(answer)
            return answer

        if version != "0":
            self.arcConsist()

        candidate = self.selectUnassign(answer)
        for i in range(self.colors):
            if self.checkConsist(candidate,i,answer):

                if version != "0":
                    self.nodeColor[candidate].remove(i)

                answer.append((candidate,i))
                result = self.backtrackingRec(answer)
                if result != False:
                    return result
                answer.remove((candidate,i))
                if version != "0":
                    self.nodeColor[candidate].append(i)
        return False

    """
    below three functions are helper function of backtrack algorithm
    """
    def selectUnassign(self,answer):
        aList  = [i for i in range(self.nodes)]
        for i in answer:
            if i[0] in aList:
                aList.remove(i[0])
        return aList[0]

    # check if neighbor color is same
    def checkConsist(self,candidate,color,answer):
        result = True
        neighbor = self.adj[candidate]
        for i in neighbor:
            for j in answer:
                if i == j[0]: # found neighbor in answer
                    if color == j[1]: # if color in answer
                        result = False
        return result

    def addall(self):
        lst = []
        for i in range(self.nodes):
            for j in self.adj[i]:
                instant = (i,j)
                if instant not in lst:
                    if (instant[1],instant[0]) not in lst:
                        lst.append(instant)
        return lst

    def arcConsist(self):
        """
        this is main function of arc3 algorithm
        it doesn't return but it might changed problem state
        """
        # initial arcs for all node
        que = self.addall()
        while len(que) != 0:
            tup = que.pop(0)
            if self.removeInconsistent(tup):
                for i in self.adj[tup[0]]:
                    que.append((i, tup[0]))

    def removeInconsistent(self,tup):
        """
        we check if any other tuple in list we remove them and back to main fucntion of arc3

        :param tup: a tuple that form as (node index, color index)
        :return:
        """
        removed = False
        for i in self.nodeColor[tup[0]]:
            satisfy = False
            for j in self.nodeColor[tup[1]]:
                if i!=j:
                    satisfy = True
            if not satisfy:
                self.nodeColor[tup[0]].remove(i)
                removed = True

        return removed

def endLong():
    """
    this function stop program if run time is too long
    change the number if you want more/less time
    """
    currentTime = time.time()
    if currentTime-start >= 65: # CHANGE THIS IF YOU WANT REASONABLE TIME TO BE DIFFERENT
        print("One and half minute passed and still didn't get answer")
        print("Exiting now...")
        fout = open(output, "w")
        fout.write("No answer")
        fout.close()
        exit(-1)


# python dfsb.py backtrack_hard backtrack_hard-output.txt 0
# python dfsb.py N400 N400out.txt 0
# python CSPGenerator.py 400 26666 4 N400 0
if __name__ == '__main__':
    """
    this is main function that check argv and initialize map coloring problem and out put answer into txt
    """
    if len(sys.argv) != 4:
        print("Usage: python dfsb.py <INPUT FILE> <OUTPUT FILE> <MODE FLAG>")
        exit(-1)

    input = sys.argv[1]
    global output
    output = sys.argv[2]
    global version
    version = sys.argv[3] # it's a string
    """#debug
    import os
    os.chdir("X:/JAR Work/CSE352/A2/")
    input = "backtrack_easy"
    global version
    version = "0"
    """

    # set timmer
    global start
    start = time.time()

    # get info from input
    f = open(input,"r")
    firstLine = f.readline().replace("\n","").split()
    N,M,K = int(firstLine[0]),int(firstLine[1]),int(firstLine[2])
    problem = Graph(N,K)

    # get edges
    for i in range(M):
        edge = f.readline().split()
        problem.addConstrains(int(edge[0]),int(edge[1]))
    f.close()

    # search and record answers
    direct = problem.backtracking()
    print(direct)

    # output answer to file
    fout = open(output, "w")
    if not direct:
        fout.write("No answer")
    else:
        for i in direct:  # check answers
            fout.write("{}\n".format(i[1]))
            # print(i[0],i[1])
    fout.close()

    end = time.time()
    #print("State Explored:", problem.statenumber)
    print("Run time is:",end-start)




