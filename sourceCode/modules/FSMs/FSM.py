
class BasicState(object):
   def __init__(self, facing="none"):
      self._facing = facing
      
   def getFacing(self):
      return self._facing

   def _setFacing(self, direction):
      self._facing = direction
      
class BakerCatState(object):
   def __init__(self, state="standing"):
      self._state = state
      self.movement = {
         "up" : False,
         "down" : False,
         "left" : False,
         "right" : False
      }
      
      self._lastFacing = "right"
      
   def getFacing(self):
      if self.movement["left"] == True:
         self._lastFacing = "left"
      elif self.movement["right"] == True:
         self._lastFacing = "right"
      
      return self._lastFacing

   def manageState(self, action, bakerCat):
      if action in self.movement.keys():
         if self.movement[action] == False:
            self.movement[action] = True
            if self._state == "standing":
               self._state = "moving"
               bakerCat.transitionState(self._state)
         
      elif action.startswith("stop") and action[4:] in self.movement.keys():
         direction = action[4:]
         if self.movement[direction] == True:            
            self.movement[direction] = False
            allStop = True
            for move in self.movement.keys():
               if self.movement[move] == True:
                  allStop = False
                  break
               
            if allStop:
               self._state = "standing"
               bakerCat.transitionState(self._state)

   def getState(self):
      return self._state

class KarenCatState(object):
   def __init__(self, state="standing"):
      self._state = state
      self.movement = {
         "up": False,
         "down": False,
         "left": False,
         "right": False
      }

      self._lastFacing = "right"

   def getFacing(self):
      if self.movement["left"] == True:
         self._lastFacing = "left"
      elif self.movement["right"] == True:
         self._lastFacing = "right"

      return self._lastFacing

   def manageState(self, action, karenCat):
      if action in self.movement.keys():
         if self.movement[action] == False:
            self.movement[action] = True
            if self._state == "standing":
               self._state = "moving"
               karenCat.transitionState(self._state)

      elif action.startswith("stop") and action[4:] in self.movement.keys():
         direction = action[4:]
         if self.movement[direction] == True:
            self.movement[direction] = False
            allStop = True
            for move in self.movement.keys():
               if self.movement[move] == True:
                  allStop = False
                  break

            if allStop:
               self._state = "standing"
               karenCat.transitionState(self._state)
         
   def getState(self):
      return self._state


class CustomerCatState(object):
   def __init__(self, state="standing"):
      self._state = state

   def manageState(self):

      if not self._state == "standing":
         self.transitionState(self._state)


   def getState(self):
      return self._state

   def __eq__(self, other):
      return self._state == other


class SaltState(object):
   def __init__(self, state="salting"):
      self._state = state

   def manageState(self):
      if not self._state == "salting":
         self.transitionState(self._state)


   def getState(self):
      return self._state

   def __eq__(self, other):
      return self._state == other
