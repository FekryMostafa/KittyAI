from .basicManager import BasicManager
from .gameManager import GameManager
from ..FSMs.screenFSM import ScreenState
from ..UI.items import Text
from ..UI.displays import *
from ..gameObjects.vector2D import Vector2
from ..UI.screenInfo import SCREEN_SIZE
import pygame


class ScreenManager(BasicManager):

    def __init__(self):
        super().__init__()
        self._game = GameManager(SCREEN_SIZE)
        self._state = ScreenState()
        self._pausedText = Text(Vector2(0, 0), "Paused")

        size = self._pausedText.getSize()
        midPointX = SCREEN_SIZE.x // 2 - size[0] // 2
        midPointY = SCREEN_SIZE.y // 2 - size[1] // 2

        self._pausedText.setPosition(Vector2(midPointX, midPointY))

        hcMenu = HoverClickMenu("menuBG.jpg", fontName="default8")

        hcMenu.addOption("start", "Start Game",
                                   SCREEN_SIZE // 2 - Vector2(0, 50),
                                   center="both")
        hcMenu.addOption("exit", "Exit Game",
                                   SCREEN_SIZE // 2 + Vector2(0, 50),
                                   center="both")

        cursorMenu = CursorMenu("menuBG.jpg", fontName="default8")

        cursorMenu.addOption("start", "Click Here to Start Game",
                                         SCREEN_SIZE // 2 - Vector2(0, 50),
                                         center="both")
        cursorMenu.addOption("exit", "Click Here to Exit Game",
                                         SCREEN_SIZE // 2 + Vector2(0, 200),
                                         center="both")

        eventMenu = EventMenu("menuBG.jpg", fontName="default8")

        eventMenu.addOption("start", "Start Game",
                                        SCREEN_SIZE // 2 - Vector2(0, 50),
                                        lambda x: x.type == pygame.KEYDOWN and x.key == pygame.K_1,
                                        center="both")
        eventMenu.addOption("exit", "Exit Game",
                                        SCREEN_SIZE // 2 + Vector2(0, 50),
                                        lambda x: x.type == pygame.KEYDOWN and x.key == pygame.K_2,
                                        center="both")

        self._mainMenu = [eventMenu, hcMenu, cursorMenu]
        self._gameKaren = CursorMenu("gameKaren.jpg", fontName="default8")
        self._gameKaren.addOption("exit", "Hit Enter to Exit Game",
                             SCREEN_SIZE // 2 + Vector2(0, 205),
                             center="both")
        self._gameBaker = CursorMenu("gameBaker.jpg", fontName="default8")
        self._gameBaker.addOption("exit", "Hit Enter to Exit Game",
                             SCREEN_SIZE // 2 + Vector2(0, 205),
                             center="both")
        self._currentMenu = 1

    def draw(self, drawSurf):
        if self._state == "game":
            self._game.draw(drawSurf)

            if self._state.isPaused():
                self._pausedText.draw(drawSurf)

        elif self._state == "mainMenu":
            self._mainMenu[self._currentMenu].draw(drawSurf)
        elif self._state == "gameKaren":
            self._gameKaren.draw(drawSurf)
        elif self._state == "gameBaker":
            self._gameBaker.draw(drawSurf)

    def handleEvent(self, event):
        # Handle screen-changing events first
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            self._state.manageState("pause", self)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            self._state.manageState("mainMenu", self)
        else:
            if self._state == "game" and not self._state.isPaused():
                self._game.handleEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self._currentMenu = 0
                    elif event.key == pygame.K_2:
                        self._currentMenu = 1
                    elif event.key == pygame.K_3:
                        self._currentMenu = 2

            elif self._state == "mainMenu":
                choice = self._mainMenu[self._currentMenu].handleEvent(event)

                if choice == "start":
                    self._state.manageState("startGame", self)
                elif choice == "exit":
                    return "exit"

            elif self._state == "gameKaren":
                choice = self._gameKaren.handleEvent(event)

                if choice == "exit":
                    return "exit"

            elif self._state == "gameBaker":
                choice = self._gameBaker.handleEvent(event)

                if choice == "exit":
                    return "exit"

    def update(self, ticks):
        if self._state == "game" and not self._state.isPaused():
            status = self._game.update(ticks, SCREEN_SIZE)
            if status == "karenWin":
                self._state.manageState("gameKaren", self)
            elif status == "bakerWin":
                self._state.manageState("gameBaker", self)
        elif self._state == "mainMenu":
            self._mainMenu[self._currentMenu].update(ticks)
        elif self._state == "gameKaren":
            self._gameKaren.update(ticks)
        elif self._state == "gameBaker":
            self._gameBaker.update(ticks)


    def transitionState(self, state):
        if state == "game" and not self._state.isPaused():
            self._game.updateMovement()
