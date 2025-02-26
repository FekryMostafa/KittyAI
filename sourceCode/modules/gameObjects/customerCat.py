from .animated import Animated
from ..FSMs.FSM import CustomerCatState
from ..managers.frameManager import FrameManager
import random


class CustomerCat(Animated):
    def __init__(self, position):
        customerImages = ["customer1.png", "customer2.png", "customer3.png"]
        whichCustomer = random.randint(0, 2)

        super().__init__(customerImages[whichCustomer], position)

        self._isServed = False
        self._hasLeft = False
        self._numberServedTotal = 0
        self._numberHasLeftTotal = 0

        self._nFrames = 3
        self._framesPerSecond = 2

        self._nFramesList = {
            "standing": 9
        }

        self._rowList = {
            "standing": 4
        }

        self._framesPerSecondList = {
            "standing": 4
        }

        self._state = CustomerCatState()
        self.transitionState(self._state.getState())


    def transitionState(self, state):
        self._nFrames = self._nFramesList[state]
        self._frame = 0
        self._row = self._rowList[state]
        self._framesPerSecond = self._framesPerSecondList[state]
        self._animationTimer = 0
        self._image = FrameManager.getInstance().getFrame(self._imageName, (self._row, self._frame))

    def getServed(self):
        self._isServed = True
        self._numberServedTotal += (random.randint(1,3))

    def getAnnoyed(self):
        self._hasLeft = True
        self._numberHasLeftTotal += (random.randint(1,3))

    def isServed(self):
        return self._isServed

    def hasLeft(self):
        return self._hasLeft

    def getNumberServed(self):
        return self._numberServedTotal

    def getNumberHasLeft(self):
        return self._numberHasLeftTotal

    def updateNumberServed(self):
        self._numberServedTotal += 1

    def updateNumberHasLeft(self):
        self._numberHasLeftTotal += 1