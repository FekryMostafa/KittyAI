
from ..gameObjects.drawable import Drawable
from ..gameObjects.vector2D import Vector2
from .items import *
from .screenInfo import adjustMousePos
import pygame, os

class AbstractMenu(Drawable):
   """Abstract class for basic menus."""
   def __init__(self, background,
                fontName="default", color=(255,255,255)):
      super().__init__(background, (0,0), parallax=0)
         
      self._options = {}
      
      self._color = color      
      self._font = fontName
   
   def addOption(self, key, text, position, center=None):
      self._options[key] = Text(position, text, self._font, self._color)
      
      if center != None:

         x = position[0]
         y = position[1]
         
         size = self._options[key].getSize()
         
         if center in ["horizontal", "both"]:
            x -= size[0] // 2
         
         if center in ["veritcal", "both"]:
            y -= size[1] // 2
         
         position = Vector2(x, y)
      
         self._options[key].setPosition(position)
   
   def draw(self, surface):
      super().draw(surface)
      
      for item in self._options.values():
         item.draw(surface)
   
   def update(self, ticks):
      pass
      

class ClickMenu(AbstractMenu):
   """Menu which uses clicking/mouse events for selection."""
   def __init__(self, background, 
                fontName="default", color=(255,255,255)):
      super().__init__(background, fontName, color)
   
   def handleEvent(self, event):
      if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
         position = adjustMousePos(event.pos)
         
         collider = self._findCollision(position)
         
         if collider:
            return collider
   
   def _findCollision(self, position):      
      for key in self._options.keys():
         if self._options[key].getCollisionRect().collidepoint(*position):
      
            return key

class HoverClickMenu(ClickMenu):
   """Menu which uses clicking/mouse events for selection.
   Uses hover text instead of normal text."""
   
   def __init__(self, background, 
                fontName="default", color=(255,255,255),
                hoverColor=(255,0,0)):
      super().__init__(background, fontName, color)
      
      self._hoverColor = hoverColor
   
   
   def handleEvent(self, event):      
      for item in self._options.values():
         item.handleEvent(event)
      
      selection = super().handleEvent(event)
      
      if selection != None:
         self.clearHovers()
      
      return selection

   
   def addOption(self, key, text, position, center=None):
      super().addOption(key, text, position, center)
      
      t = self._options[key]
      
      self._options[key] = HoverText(t.getPosition(), text, self._font, self._color, self._hoverColor)
      
   
   def clearHovers(self):
      
      for item in self._options.values():
         item.clearHover()
         
            
class EventMenu(AbstractMenu):
   """Menu which uses event lambda functions for selection."""
   
   def __init__(self, background, fontName="default", color=(255,255,255)):
      super().__init__(background, fontName, color)
      
      self._eventMap = {}
   
   def addOption(self, key, text, position, eventLambda, center=None):
      super().addOption(key, text, position, center)
      
      self._eventMap[key] = eventLambda
   
   def draw(self, surface):      
      super().draw(surface) 
   
   def handleEvent(self, event):      
      for key in self._eventMap.keys():
         function = self._eventMap[key]
         if function(event):
            return key
         

class CursorMenu(AbstractMenu):
   """Menu which uses directional keys/a cursor for selection."""
   
   def __init__(self, background, cursor="arrow.png",
                fontName="default", color=(255,255,255)):
      super().__init__(background, fontName, color)
      
      self._cursor = Drawable(cursor, Vector2(0,0), parallax=0)
      self._current = None
      
      self._controls = {
         pygame.K_UP : "up",
         pygame.K_DOWN : "down",
         pygame.K_RIGHT : "right",
         pygame.K_LEFT : "left" 
      }
   
   def addOption(self, key, text, position, center=None):
      super().addOption(key, text, position, center)
      
      self._current = key
      self._moveCursor()
   
   def setCursor(self, key):
      self._current = key
      self._moveCursor()
      
   
   def draw(self, surface):      
      super().draw(surface)      
      self._cursor.draw(surface)
   
   def _moveCursor(self):
      self._cursor.setPosition(self._options[self._current].getPosition() - Vector2(self._cursor.getCollisionRect().width, 0))

   
   def _findNearestInDirection(self, direction):
      
      currentPosition = self._options[self._current].getPosition()
      
      nearest = None
      
      for key in self._options.keys():
         keyPosition = self._options[key].getPosition()
         
         if (direction == "up" and keyPosition[1] < currentPosition[1]) or \
            (direction == "down" and keyPosition[1] > currentPosition[1]) or \
            (direction == "left" and keyPosition[0] < currentPosition[0]) or \
            (direction == "right" and keyPosition[0] > currentPosition[0]):
            
            if nearest == None or (keyPosition - currentPosition).magnitude() < (self._options[nearest].getPosition() - currentPosition).magnitude():
               nearest = key
      
      return nearest
      
   
   def handleEvent(self, event):
      
      if event.type == pygame.KEYDOWN:
         if event.key in self._controls.keys():
            newCurr = self._findNearestInDirection(self._controls[event.key])
         
            if newCurr != None:
               self._current = newCurr
               self._moveCursor()
               
         elif event.key == pygame.K_RETURN:
            return self._current

         
   


        
   