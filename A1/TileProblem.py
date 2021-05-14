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