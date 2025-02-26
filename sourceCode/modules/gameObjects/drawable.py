import pygame
from pygame import image
import os

from ..FSMs.FSM import BasicState
from ..managers.frameManager import FrameManager
from .vector2D import Vector2


class Drawable(object):
   _position: Vector2

   def __init__(self, imageName, position, offset=None, parallax=1):
      self._imageName = imageName

      # Let frame manager handle loading the image
      if self._imageName != "":  
         self._image = FrameManager.getInstance().getFrame(self._imageName, offset)

      self._position = Vector2(*position)
      self._state = BasicState()
      self._parallax = parallax

   def getPosition(self) -> object:
      return self._position

   def setPosition(self, newPosition):
      self._position = newPosition

   def getSize(self):
      return self._image.get_size()

   def setImage(self, surface):
      self._image = surface

   def getCollisionRect(self):
      newRect = self._position + self._image.get_rect()
      return newRect

   def draw(self, surface):
      blitImage = self._image

      if not self._state == "salting":
         if not self._state == "standing":
            if self._state and self._state.getFacing() == "right":
               blitImage = pygame.transform.flip(self._image, True, False)

      surface.blit(blitImage, (self._position[0], self._position[1]))
