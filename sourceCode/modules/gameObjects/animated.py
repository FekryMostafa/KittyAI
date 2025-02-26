from ..managers.frameManager import FrameManager
from .drawable import Drawable

class Animated(Drawable):
   
   def __init__(self, imageName, location):
      super().__init__(imageName, location, (0,0))
      
      self._frame = 0
      self._row = 0
      self._animationTimer = 0
      self._framesPerSecond = 10.0
      self._nFrames = 3
      
      self._animate = True
      

      
   def update(self, ticks):
      if self._animate:
         self._animationTimer += ticks
         
         if self._animationTimer > 1 / self._framesPerSecond:
            self._frame += 1
            self._frame %= self._nFrames
            self._animationTimer -= 1 / self._framesPerSecond
            self._image = FrameManager.getInstance().getFrame(self._imageName, (self._row, self._frame))
         