from sys import argv # use to examine input line
from sys import maxsize
from math import sqrt
import time

class Heuristics:
    """This function will compare each element in currentboard
            to correct answer. It returns the summation of mismatch
            distance. e.g. mismatch distance of 3 and 8 is 5.
            This heuristic function will always dominate the other one"""
    """@staticmethod
    def h1(currentBoard):
        
        #given:
        # current game board state, 2d list
        #return:
        # mismatch count of current state with answer
        count = 0
        for i in range(len(currentBoard)):
            for j in range(len(currentBoard)):
                if (currentBoard[i][j] == '') and (answer[i][j] == ''):
                    count += 0
                elif (currentBoard[i][j] == ''):
                    count += int(answer[i][j])
                elif (answer[i][j] == ''):
                    count += int(currentBoard[i][j])
                elif int(currentBoard[i][j]) != int(answer[i][j]):
                    count += abs(int(currentBoard[i][j])-int(answer[i][j]))
        return count"""

    """This function will count the total mis-ordering of current
            game state. Mis-ordering is x[k]+1 != x[k+1] e.g. 1234 has 0
            mis-order; 1235 has 1 mis-order; 1243 has 2 and 12435 has 3"""
    """@staticmethod
    def h2(currentBoard):
        
        #given:
        # current game board state, 2d list
        #return:
        # total mis-ordering count
        count = 0
        lastElement = 0
        for i in range(len(currentBoard)):
            for j in range(len(currentBoard)):
                if currentBoard[i][j]=='':
                    continue
                if lastElement == 0:
                    lastElement = int(currentBoard[i][j])
                else:
                    if lastElement+1 != int(currentBoard[i][j]):
                        count+=1
                    lastElement = int(currentBoard[i][j])
        return count"""

    @staticmethod
    def h1(currentBoard):
        """
        this is Manhattan distance heuristic function

        :param currentBoard: current game state
        :return: sum of Manhattan distance with answer of all elements
        """
        aLst = []
        for i in currentBoard:
            for j in i:
                aLst.append(j)
        ansLst = []
        for i in answer:
            for j in i:
                ansLst.append(j)

        sum = 0
        for num in range(1,int(N)**2):
            distance = abs(aLst.index(str(num)) - ansLst.index(str(num)))
            row = distance // int(N)
            col = int(distance % 3)
            sum += row
            sum += col
        return sum

    @staticmethod
    def h2(currentBoard):
        """
        this is sum of Euclidean distance of every single elements
        :param currentBoard: current game state
        :return: sum of Euclidean distance of all element
        """
        aLst = []
        for i in currentBoard:
            for j in i:
                aLst.append(j)
        ansLst = []
        for i in answer:
            for j in i:
                ansLst.append(j)

        sum = 0
        for num in range(1,int(N)**2):
            distance = abs(aLst.index(str(num)) - ansLst.index(str(num)))
            row = distance // int(N)
            col = int(distance % 3)
            euclidean = sqrt(row**2+col**2)
            sum += euclidean
        return sum

class TileProblem:
    def __init__(self,gameState,pastActions,cost,parentState,h):
        self.gameState = gameState
        self.pastActions = pastActions
        self.parent = parentState # to track backward
        self.h = h # h function
        if self.parent!=None:
            self.pathCost = self.parent.pathCost+1
        else:
            self.pathCost = 0
        self.cost = self.pathCost+cost

    def goalTest(self):
        """
        check if current state is same with answer
        :return: boolean if current state is answer
        """
        for i in range(len(self.gameState)):
            for j in range(len(self.gameState)):
                if self.gameState[i][j] != answer[i][j]:
                    return False
        return True

    # transition functions
    def upToBlank(self):
        """This function move the element right top of the blank
        to the blank place"""
        for i in range(len(self.gameState)):
            for j in range(len(self.gameState)):
                if self.gameState[i][j] == "":
                    if i == 0: # if this is first line, return nothing
                        return None
                    # make new tile problem obj and swap
                    instant = TileProblem([i.copy() for i in self.gameState],
                                          self.pastActions.copy(),
                                          self.cost,self,self.h)
                    instant.gameState[i][j] = instant.gameState[i-1][j]
                    instant.gameState[i-1][j] = ""
                    instant.pastActions.append("U")
                    instant.cost = instant.h(instant.gameState)+instant.pathCost
                    return instant

    def leftToBlank(self):
        """This function move the element right top of the blank
        to the blank place"""
        for i in range(len(self.gameState)):
            for j in range(len(self.gameState)):
                if self.gameState[i][j] == "":
                    if j == 0: # if this is left col, return nothing
                        return None
                    # make new tile problem obj and swap
                    instant = TileProblem([i.copy() for i in self.gameState],
                                          self.pastActions.copy(),
                                          self.cost,self,self.h)
                    instant.gameState[i][j] = instant.gameState[i][j-1]
                    instant.gameState[i][j-1] = ""
                    instant.pastActions.append("L")
                    instant.cost = instant.h(instant.gameState)+instant.pathCost
                    return instant
    def rightToBlank(self):
        """This function move the element right top of the blank
        to the blank place"""
        for i in range(len(self.gameState)):
            for j in range(len(self.gameState)):
                if self.gameState[i][j] == "":
                    if j == len(self.gameState)-1: # if this is right most col, return nothing
                        return None
                    # make new tile problem obj and swap
                    instant = TileProblem([i.copy() for i in self.gameState],
                                          self.pastActions.copy(),
                                          self.cost,self,self.h)
                    instant.gameState[i][j] = instant.gameState[i][j+1]
                    instant.gameState[i][j+1] = ""
                    instant.pastActions.append("R")
                    instant.cost = instant.h(instant.gameState)+instant.pathCost
                    return instant
    def downToBlank(self):
        """This function move the element right top of the blank
        to the blank place"""
        for i in range(len(self.gameState)):
            for j in range(len(self.gameState)):
                if self.gameState[i][j] == "":
                    if i == len(self.gameState)-1: # if this is down most row, return nothing
                        return None
                    # make new tile problem obj and swap
                    instant = TileProblem([i.copy() for i in self.gameState],
                                          self.pastActions.copy(),
                                          self.cost,self,self.h)
                    instant.gameState[i][j] = instant.gameState[i+1][j]
                    instant.gameState[i+1][j] = ""
                    instant.pastActions.append("D")
                    instant.cost = instant.h(instant.gameState)+instant.pathCost
                    return instant

# it is caller's job to make sure state list is not empty
# helper function of aStar
def cmpState(state1,state2):
    """
    compare two state element by element

    :param state1: frist state to compare
    :param state2: second state to compare
    :return: boolean value depend on if they are the same state
    """
    for i in range(len(state1.gameState)):
        for j in range(len(state1.gameState)):
            if state1.gameState[i][j] != state2.gameState[i][j]:
                return False
    return True

def findLeastH(stateList):
    """
    loop through the list and find the least cost state to return

    :param stateList: a list of states
    :return: state that has least cost
    """
    least = stateList[0]
    index = -1
    for i in range(len(stateList)):
        if stateList[i].cost <= least.cost:
            least = stateList[i]
            index = i
    least = stateList.pop(index)
    return least

def hasSame(state,openList):
    """
    see if there are two same state in open list

    :param state: a state to check in open list
    :param openList: a list of states which haven't search yet
    :return: if found same state, return index of the state for further use
        else return -1
    """
    count = 0
    for i in openList:
        if cmpState(state,i):
            return count
        count+=1
    return -1

def aStar(start):
    """
    pseudo code cite from:
    https://www.geeksforgeeks.org/a-search-algorithm/

    :param start: a start state of game
    :return: a list of actions that lead to answer
    """
    openList = []
    closeList = []
    openList.append(start)

    # compare current state to answer
    if start.goalTest():
        return start.pastActions  # if current state is goal, return process

    while len(openList)>0:
        processState = findLeastH(openList)
        # generate 4 candidate
        candidates = []
        candidates.append(processState.upToBlank())
        candidates.append(processState.downToBlank())
        candidates.append(processState.leftToBlank())
        candidates.append(processState.rightToBlank())

        for i in candidates:
            if i != None:
                if i.goalTest(): # if this candidate is goal, stop search and return
                    print("A* state explored:",len(closeList))
                    print(("A* heuristic depth:",i.cost))
                    return i.pastActions
                # if openlist has same postion and cost is lower than this candidate, ignore
                openIndex = hasSame(i,openList)

                if openIndex != -1: # has same position
                    if openList[openIndex].cost>i.cost: # has lower cost
                        # pop old state and add new state
                        openList.pop(openIndex)
                        openList.append(i)
                    else: # skip this candidate
                        continue

                closeIndex = hasSame(i, closeList)
                if closeIndex != -1: # has same in close list
                    if closeList[closeIndex].cost>i.cost:
                        continue
                    else:
                        openList.append(i)

                if closeIndex == -1 & openIndex == -1:
                    openList.append(i)
        closeList.append(processState)

global RBFSCloseList
RBFSCloseList = []

def RBFS(state,hLimite):
    """
    pseudo code cite from:
    https://pages.mtu.edu/~nilufer/classes/cs5811/2012-fall/lecture-slides/cs5811-ch03-search-b-informed-v2.pdf

    :param state: a start state of game
    :param hLimite: a thresh hold which start with infinity, i'm using
        sys.maxsize as infinity
    :return: the answer state of tile problem
    """
    RBFSCloseList.append("1")
    if state.goalTest():
        return state, None

    candidate = []
    up = state.upToBlank()
    if up != None:
        candidate.append((up.cost,0,up))
    left = state.leftToBlank()
    if left != None:
        candidate.append((left.cost,1,left))
    right = state.rightToBlank()
    if right != None:
        candidate.append((right.cost,2,right))
    down = state.downToBlank()
    if down != None:
        candidate.append((down.cost,3,down))
    if len(candidate)==0:
        return None, maxsize

    while len(candidate)!=0:

        candidate.sort(key=lambda x: x[0])
        best = candidate[0][2]
        if best.cost > hLimite:
            return None, best.cost
        alternative = candidate[1][0]


        result, best.cost = RBFS(best,min(hLimite,alternative))
        candidate[0] = (best.cost,candidate[0][1],best)
        if result != None:

            break
    return result,None



# python puzzleSolver.py <A> <N> <H> <INPUT FILE PATH> <OUTPUT FILE PATH>

# python puzzleSolver.py 1 3 1 puzzle3.txt puzzle1_output.txt
# python puzzleSolver.py 1 4 2 puzzle4.txt puzzle4_output.txt
# python puzzleSolver.py 2 3 1 puzzle3.txt puzzle4_output.txt
# python puzzleSolver.py 2 4 2 puzzle4.txt puzzle4_output.txt


if __name__ == '__main__':
    # examine input from stdin
    start = time.time()
    if len(argv)!=6:
        print("Usage of this script: python puzzleSolver.py "
              "<A> <N> <H> <INPUT FILE PATH> <OUTPUT FILE PATH>")
        exit(-1)
    global N
    A,N,H,inputPath,outputPath = argv[1],argv[2],argv[3],argv[4],argv[5]

    #generate answer
    global answer  # make it visible to every function
    answer = []
    for i in range(int(N)):
        answer.append([])
        for j in range(int(N)):
            answer[i].append(str(i*int(N)+j+1))
    answer[-1][-1] = '' #change last element to blank

    #generate gameboard from file
    gameboard = []
    f = open(inputPath,"r")
    for i in range(int(N)):
        gameboard.append(f.readline().replace("\n","").split(","))
    f.close()

    startState = TileProblem(gameboard,[],0,None,
                             h=Heuristics.h1 if H == "1" else Heuristics.h2)

    #debug only
    print("input game board is:")
    for i in gameboard:
        print(i)
    """print(startState.h)
    print("answer is:", answer)
    print("Shift up:",startState.upToBlank().gameState)
    print("up cost is:",startState.upToBlank().cost)
    print("up past action is:",startState.upToBlank().pastActions)
    print("Shift left:",startState.leftToBlank().gameState)
    print("left cost is:",startState.leftToBlank().cost)
    print("left past action is:", startState.leftToBlank().pastActions)
    print("Shift right:", startState.rightToBlank().gameState)
    print("right cost is:",startState.rightToBlank().cost)
    print("right past action is:", startState.rightToBlank().pastActions)
    print("Shift down:", startState.downToBlank().gameState)
    print("down cost is:",startState.downToBlank().cost)
    print("down past action is:", startState.downToBlank().pastActions)
    print("did it modify old one?",startState.gameState)"""

    step = []
    # choose algorithm
    if A == "1":

        step = aStar(startState)

    else:
        finalState = RBFS(startState,maxsize)
        step = finalState[0].pastActions
        print("RBFS state explored:", len(RBFSCloseList))
        print(("RBFS heuristic depth:", finalState[0].pathCost))

    end = time.time()
    print("RBFS time: {0:.20f}".format(end - start))

    print(step)
    ansString = ""
    ansString+=step[0]
    for i in range(1,len(step)):
        ansString+=","
        ansString+=step[i]
    print(ansString)

    fw = open(outputPath,"w")
    fw.write(ansString)
    fw.close()
