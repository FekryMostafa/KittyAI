import pygame, os
from ..gameObjects.drawable import Drawable
from .screenInfo import adjustMousePos

class AbstractUIEntry(Drawable):
   """ Basic UI Entry Class
   Sets parallax to zero and contains information about fonts"""
   
   if not pygame.font.get_init():
      pygame.font.init()
   
   _FONT_FOLDER = os.path.join("sourceCode", "resources", "fonts")   
   _DEFAULT_FONT = "PressStart2P.ttf"
   _DEFAULT_SIZE = 16
   # /Users/len24/Desktop/RL-Final-Project/02-KittyKarenWars/resources/fonts/PressStart2P.ttf
   
   FONTS = {
      "default" : pygame.font.Font(os.path.join(_FONT_FOLDER, _DEFAULT_FONT), _DEFAULT_SIZE),
      "default8" : pygame.font.Font(os.path.join(_FONT_FOLDER, _DEFAULT_FONT), 8)
   }
   
   
   def __init__(self, position, parallax = 0):
      super().__init__("", position)
      
   
class Text(AbstractUIEntry):
   """A plain text UI entry."""
   
   def __init__(self, position, text, font="default", color=(255,255,255)):
      super().__init__(position)
      self._color = color
      
      self._image = AbstractUIEntry.FONTS[font].render(text, False, self._color)

class HoverText(Text):
   """Text which changes color when the mouse hovers over it."""
   def __init__(self, position, text, font, color, hoverColor):
      super().__init__(position, text, font, color)

      self._image = AbstractUIEntry.FONTS[font].render(text, False, color)
      self._passive = self._image
      self._hover = AbstractUIEntry.FONTS[font].render(text, False, hoverColor)
   
   def handleEvent(self, event):      
      if event.type == pygame.MOUSEMOTION:
         position = adjustMousePos(event.pos)
         if self.getCollisionRect().collidepoint(*position):
            self._image = self._hover
         else:
            self._image = self._passive
   
   def clearHover(self):
      self._image = self._passive


class AbstractItem(AbstractUIEntry):
   """Abstract class for countable UI items."""

   def __init__(self, position, initialValue, maxValue, minValue=0):
      super().__init__(position)
      self._value = initialValue
      self._maxValue = maxValue
      self._minValue = minValue

   def getValue(self):
      return self._value

   def change(self, value):
      self._value = max(self._minValue, min(self._maxValue, value))
      self._render()

   def increase(self, value=1):
      self.change(self._value + value)

   def decrease(self, value=1):
      self.change(self._value - value)

   def setMax(self, value):
      self._maxValue = value
      self._render()

   def setMin(self, value):
      self._minValue = value
      self._render()

   def update(self, seconds):
      pass


class BarItem(AbstractItem):
   def __init__(self, position, width, height, initialValue, maxValue=10,
                color=(255, 0, 0), outlineColor=(100, 100, 100), outlineWidth=2,
                backgroundColor=None):
      super().__init__(position, initialValue, maxValue)
      self._width = width
      self._height = height
      self._color = color
      self._outlineColor = outlineColor
      self._outlineWidth = outlineWidth
      self._backgroundColor = backgroundColor

      self._image = pygame.Surface((self._width + self._outlineWidth * 2,
                                    self._height + self._outlineWidth * 2),
                                   pygame.SRCALPHA, 32)
      self._render()

   def _render(self):
      valueRect = pygame.Rect(self._outlineWidth, self._outlineWidth,
                              int(self._width * (self._value / self._maxValue)),
                              self._height)
      fullRect = pygame.Rect(self._outlineWidth, self._outlineWidth,
                              self._width,
                              self._height)
      self._image.fill((0, 0, 0, 0))

      if self._backgroundColor:
         pygame.draw.rect(self._image, self._backgroundColor, fullRect)

      if valueRect.width >= 1:
         pygame.draw.rect(self._image, self._color, valueRect)

      if self._outlineWidth > 0:
         pygame.draw.rect(self._image, self._outlineColor, fullRect,
                           self._outlineWidth)


