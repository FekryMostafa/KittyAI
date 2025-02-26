
from .animated import Animated
from .vector2D import Vector2

SCREEN_SIZE = Vector2(1530, 856)

class Mobile(Animated):
   def __init__(self, imageName, position):
      super().__init__(imageName, position)
      self._velocity = Vector2(0,0)
      
   
   def update(self, ticks):
      
      super().update(ticks)
      
      self._velocity = Vector2(0, 0)
      
      if self._state.getState() != "standing":
         currentFacing = self._state.getFacing()

         if self._state.movement["down"]:
            self._velocity[1] = self._vSpeed
         elif self._state.movement["up"]:
            self._velocity[1] = -self._vSpeed
         if self._state.movement["left"]:
            self._velocity[0] = -self._vSpeed
         elif self._state.movement["right"]:
            self._velocity[0] = self._vSpeed

         newPosition = self.getPosition() + self._velocity * ticks
         
         self.setPosition(newPosition)
      
      else:
         self._velocity = Vector2(0,0)
         
         

      