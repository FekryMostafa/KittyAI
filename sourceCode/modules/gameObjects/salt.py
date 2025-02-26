from ..FSMs.FSM import SaltState
from .animated import Animated
from ..managers.frameManager import FrameManager

import pygame


class Salt(Animated):
    def __init__(self, position):
        super().__init__("salt.png", position)

        self._nFrames = 4

        self._nFramesList = {
            "salting": 4
        }

        self._rowList = {
            "salting": 0
        }

        self._framesPerSecondList = {
            "salting": 1
        }

        self._state = SaltState()

    def transitionState(self, state):
        self._nFrames = self._nFramesList[state]
        self._frame = 0
        self._row = self._rowList[state]
        self._framesPerSecond = self._framesPerSecondList[state]
        self._animationTimer = 0
        self._image = FrameManager.getInstance().getFrame(self._imageName, (self._row, self._frame))
