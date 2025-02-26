from .mobile import Mobile
from ..FSMs.FSM import BakerCatState
from ..managers.frameManager import FrameManager

import pygame

class BakerCat(Mobile):
   def __init__(self, position):
      super().__init__("bakercat.png", position)
      
      self._nFrames = 3
      self._vSpeed = 80
      self._framesPerSecond = 2
      
      self._nFramesList = {
         "moving": 3,
         "standing": 9,
      }
      
      self._rowList = {
         "moving" : 1,
         "standing" : 4,
      }
      
      self._framesPerSecondList = {
         "moving" : 6,
         "standing" : 4,
      }
      
      self._state = BakerCatState()

   # def handleEvent(self, event): # change the same thing for KarenCat
   #    # if event.type == pygame.JOYAXISMOTION
   #    if event.type == pygame.KEYDOWN: # or pygame.KEYUP (start doing/stop doing the thing) (2 different if statements)
   #       # key down is starting, key up is stopping
   #       # if event.key == pygame.K_w: or pyag
   #       if event.axis == 0:
   #          if abs(event.value) < 0.1 and abs(event.value) > 0:
   #             self._state.manageState("stopleft", self)
   #             self._state.manageState("stopright", self)
   #          elif event.value < 0:
   #             self._state.manageState("left", self)
   #             self._state.manageState("stopright", self)
   #          elif event.value > 0:
   #             self._state.manageState("right", self)
   #             self._state.manageState("stopleft", self)

   #       if event.axis == 1:
   #          if abs(event.value) < 0.1 and abs(event.value) > 0:
   #             self._state.manageState("stopdown", self)
   #             self._state.manageState("stopup", self)
   #          elif event.value < 0:
   #             self._state.manageState("up", self)
   #             self._state.manageState("stopdown", self)
   #          elif event.value > 0:
   #             self._state.manageState("down", self)
   #             self._state.manageState("stopup", self)

   def handleEvent(self, event):  # Adjust for keyboard input
      if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP: # key down is starting, key up is stopping
         if event.key == pygame.K_w: 
               if event.type == pygame.KEYDOWN:
                  self._state.manageState("up", self)  # Start moving up
               elif event.type == pygame.KEYUP:
                  self._state.manageState("stopup", self)  # Stop moving up

         if event.key == pygame.K_s:
               if event.type == pygame.KEYDOWN:
                  self._state.manageState("down", self)  # Start moving down
               elif event.type == pygame.KEYUP:
                  self._state.manageState("stopdown", self)  # Stop moving down

         if event.key == pygame.K_a:
               if event.type == pygame.KEYDOWN:
                  self._state.manageState("left", self)  # Start moving left
               elif event.type == pygame.KEYUP:
                  self._state.manageState("stopleft", self)  # Stop moving left

         if event.key == pygame.K_d:
               if event.type == pygame.KEYDOWN:
                  self._state.manageState("right", self)  # Start moving right
               elif event.type == pygame.KEYUP:
                  self._state.manageState("stopright", self)  # Stop moving right

   
  
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