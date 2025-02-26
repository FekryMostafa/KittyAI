from .mobile import Mobile
from ..FSMs.FSM import KarenCatState
from ..managers.frameManager import FrameManager

import pygame


class KarenCat(Mobile):
    def __init__(self, position):
        super().__init__("karencat.png", position)

        self._nFrames = 3
        self._vSpeed = 80
        self._framesPerSecond = 2

        self._nFramesList = {
            "moving": 3,
            "standing": 9,
        }

        self._rowList = {
            "moving": 1,
            "standing": 4,
        }

        self._framesPerSecondList = {
            "moving": 6,
            "standing": 4,
        }

        self._state = KarenCatState()


    # def handleEvent(self, event):
    #     if event.type == pygame.JOYAXISMOTION:
    #         if event.axis == 0:
    #             if abs(event.value) < 0.1 and abs(event.value) >= 0:
    #                 self._state.manageState("stopleft", self)
    #                 self._state.manageState("stopright", self)
    #             elif event.value < 0:
    #                 self._state.manageState("left", self)
    #                 self._state.manageState("stopright", self)
    #             elif event.value > 0:
    #                 self._state.manageState("right", self)
    #                 self._state.manageState("stopleft", self)

    #         if event.axis == 1:
    #             if abs(event.value) < 0.1 and abs(event.value) >= 0:
    #                 self._state.manageState("stopdown", self)
    #                 self._state.manageState("stopup", self)
    #             elif event.value < 0:
    #                 self._state.manageState("up", self)
    #                 self._state.manageState("stopdown", self)
    #             elif event.value > 0:
    #                 self._state.manageState("down", self)
    #                 self._state.manageState("stopup", self)

    def handleEvent(self, event): # Adjust for keyboard input
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                if event.key == pygame.K_LEFT:
                    if event.type == pygame.KEYDOWN:
                        self._state.manageState("left", self)
                        self._state.manageState("stopright", self)
                    elif event.type == pygame.KEYUP:
                        self._state.manageState("stopleft", self)
                        self._state.manageState("stopright", self)
                    
                if event.key == pygame.K_RIGHT:
                    if event.type == pygame.KEYDOWN:
                        self._state.manageState("right", self)
                        self._state.manageState("stopleft", self)
                    elif event.type == pygame.KEYUP:
                        self._state.manageState("stopright", self)
                        self._state.manageState("stopleft", self)

            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                if event.key == pygame.K_UP:
                    if event.type == pygame.KEYDOWN:
                        self._state.manageState("up", self)
                        self._state.manageState("stopdown", self)
                    elif event.type == pygame.KEYUP:
                        self._state.manageState("stopup", self)
                        self._state.manageState("stopdown", self)
                    
                if event.key == pygame.K_DOWN:
                    if event.type == pygame.KEYDOWN:
                        self._state.manageState("down", self)
                        self._state.manageState("stopup", self)
                    elif event.type == pygame.KEYUP:
                        self._state.manageState("stopdown", self)
                        self._state.manageState("stopup", self)


    

    def transitionState(self, state):
        self._nFrames = self._nFramesList[state]
        self._frame = 0
        self._row = self._rowList[state]
        self._framesPerSecond = self._framesPerSecondList[state]
        self._animationTimer = 0
        self._image = FrameManager.getInstance().getFrame(self._imageName, (self._row, self._frame))

    def updateMovement(self):
        self._state.manageState("stopright", self)
        self._state.manageState("stopleft", self)
        self._state.manageState("stopup", self)
        self._state.manageState("stopdown", self)