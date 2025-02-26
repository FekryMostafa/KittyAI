import pygame
import random
import os
from .basicManager import BasicManager
from ..gameObjects.bakerCat import BakerCat
from ..gameObjects.karenCat import KarenCat
from ..gameObjects.customerCat import CustomerCat
from ..gameObjects.salt import Salt
from ..gameObjects.drawable import Drawable
from ..gameObjects.vector2D import Vector2
from ..UI.items import BarItem

key_states = {pygame.K_v: False, pygame.K_j: False, pygame.K_x: False, pygame.K_l: False, pygame.K_z: False, pygame.K_k: False, pygame.K_c: False}
#key_states = pygame.key.get_pressed()


class GameManager(BasicManager):
   
   WORLD_SIZE = Vector2(5000, 200)

   def get_game_state(self):
        game_state = {
            "baker_position": self._bakerCat.getPosition(),
            "karen_position": self._karenCat.getPosition(),
            "baker_bar_current": self._bakerBarCurrent,
            "karen_bar_current": self._karenBarCurrent,
            # Add other game state information as needed
        }

      #   # Include positions and states of other game objects
      #   for customer_id, customer in enumerate(self._customers):
      #       game_state[f"customer_{customer_id}_position"] = customer.getPosition()
      #       game_state[f"customer_{customer_id}_served"] = customer.isServed()

        return game_state
   

   def __init__(self, SCREEN_SIZE):
      # bar item related
      self._bakerBarCurrent = 0
      self._karenBarCurrent = 0
      self._bakerBar = BarItem(Vector2(410, 400), 100, 20, self._bakerBarCurrent, 100, color=(255, 91, 165))
      self._bakerIcon = Drawable("bakerIcon.png", Vector2(385, 397))
      self._karenBar = BarItem(Vector2(30, 400), 100, 20, self._karenBarCurrent, 100, color=(135,206,250))
      self._karenIcon = Drawable("karenIcon.png", Vector2(5, 397))

      # icon objects
      self._colorSaltIcon = Drawable("colorSaltIcon.png", Vector2(140, 397))
      self._bwSaltIcon = Drawable("bwSaltIcon.png", Vector2(140, 397))
      self._colorYelpIcon = Drawable("colorYelpIcon.png", Vector2(180, 397))
      self._bwYelpIcon = Drawable("bwYelpIcon.png", Vector2(180, 397))
      self._colorPusheenIcon = Drawable("colorPusheenIcon.png", Vector2(520, 397))
      self._bwPusheenIcon = Drawable("bwPusheenIcon.png", Vector2(520, 397))
      self._colorSampleIcon = Drawable("colorSampleIcon.png", Vector2(560, 397))
      self._bwSampleIcon = Drawable("bwSampleIcon.png", Vector2(560, 397))

      # furniture
      self._fridges = []
      self._fridgeXPreOffset = 68
      self._fridgeXOffset = 32

      self._counters = []
      self._counterXPreOffset = 48
      self._counterXOffset = 24

      self._table1 = Drawable("table.png", Vector2((SCREEN_SIZE.x // 2) + 200, (SCREEN_SIZE.y // 2) - 45))
      self._table2 = Drawable("table.png", Vector2((SCREEN_SIZE.x // 2) + 275, (SCREEN_SIZE.y // 2) - 95))
      self._table3 = Drawable("table.png", Vector2(SCREEN_SIZE.x - 385, SCREEN_SIZE.y - 388))

      self._tables = [self._table3, self._table2, self._table1]

      self._countertops = []
      self._countertopYPreOffset = 250
      self._countertopYOffset = 24

      self._ovens = []
      self._ovenPreOffset = 196
      self._ovenOffset = 38

      for f in range(1, 5):
         self._fridge = Drawable("fridge.png", Vector2(SCREEN_SIZE.x - (self._fridgeXPreOffset + self._fridgeXOffset), \
                                                 SCREEN_SIZE.y - 415))
         self._fridges.append(self._fridge)
         self._fridgeXPreOffset += self._fridgeXOffset

      for o in range(1, 4):
         self._oven = Drawable("oven.png", Vector2(SCREEN_SIZE.x - (self._ovenPreOffset + self._ovenOffset), \
                                             SCREEN_SIZE.y - 403))
         self._ovens.append(self._oven)
         self._ovenPreOffset += self._ovenOffset

      for c in range(1, 14):
         self._counter = Drawable("counter.png", Vector2(SCREEN_SIZE.x - (self._counterXPreOffset + self._counterXOffset), \
                                                   SCREEN_SIZE.y - 210))
         self._counters.append(self._counter)
         self._counterXPreOffset += self._counterXOffset

      for ct in range(1, 7):
         self._countertop = Drawable("countertop.png", Vector2(SCREEN_SIZE.x - 432, \
                                                         SCREEN_SIZE.y - (self._countertopYPreOffset + self._countertopYOffset)))
         self._countertops.append(self._countertop)
         self._countertopYPreOffset += self._countertopYOffset



      # customer spot creation and rect making
      self._counterRect = pygame.Rect((SCREEN_SIZE.x - 340, SCREEN_SIZE.y - 210), (336, 20))
      self._countertopRect = pygame.Rect((SCREEN_SIZE.x - 425, (SCREEN_SIZE.y // 2) - 190), (10, 130))

      self._worldBottom = pygame.Rect((0, (SCREEN_SIZE.y - 30)), (5000, 50))
      self._worldTop = pygame.Rect((0, (SCREEN_SIZE.y - 465)), (5000, 50))
      self._worldRight = pygame.Rect((SCREEN_SIZE.x - 40, 0), (50, 500))
      self._worldLeft = pygame.Rect((SCREEN_SIZE.x - 770, 0), (50, 500))

      self._spot1 = pygame.Rect((SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) - 150), (25, 25))
      self._spot2 = pygame.Rect((SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) - 50), (25, 25))
      self._spot3 = pygame.Rect((SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) + 50), (25, 25))
      self._spot4 = pygame.Rect((SCREEN_SIZE.x - 350, (SCREEN_SIZE.y // 2) + 50), (25, 25))
      self._spot5 = pygame.Rect((SCREEN_SIZE.x - 250, (SCREEN_SIZE.y // 2) + 50), (25, 25))
      self._spot6 = pygame.Rect((SCREEN_SIZE.x - 150, (SCREEN_SIZE.y // 2) + 50), (25, 25))

      # customer-leaving objects -- did most of set up for a visual pop up saying "yum!" if served but did not have time to complete
      self._spot1Yum = Drawable("yum.png", Vector2(SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) - 150))
      self._spot2Yum = Drawable("yum.png", Vector2(SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) - 50))
      self._spot3Yum = Drawable("yum.png", Vector2(SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) + 50))
      self._spot4Yum = Drawable("yum.png", Vector2(SCREEN_SIZE.x - 350, (SCREEN_SIZE.y // 2) + 50))
      self._spot5Yum = Drawable("yum.png", Vector2(SCREEN_SIZE.x - 250, (SCREEN_SIZE.y // 2) + 50))
      self._spot6Yum = Drawable("yum.png", Vector2(SCREEN_SIZE.x - 150, (SCREEN_SIZE.y // 2) + 50))

      self._yums = [self._spot1Yum, self._spot2Yum, self._spot3Yum, self._spot4Yum, self._spot5Yum, self._spot6Yum]

      self._spot1Ugh = pygame.Rect((SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) - 150), (25, 25))
      self._spot2Ugh = pygame.Rect((SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) - 50), (25, 25))
      self._spot3Ugh = pygame.Rect((SCREEN_SIZE.x - 475, (SCREEN_SIZE.y // 2) + 50), (25, 25))
      self._spot4Ugh = pygame.Rect((SCREEN_SIZE.x - 350, (SCREEN_SIZE.y // 2) + 50), (25, 25))
      self._spot5Ugh = pygame.Rect((SCREEN_SIZE.x - 250, (SCREEN_SIZE.y // 2) + 50), (25, 25))
      self._spot6Ugh = pygame.Rect((SCREEN_SIZE.x - 150, (SCREEN_SIZE.y // 2) + 50), (25, 25))

      self._spots = [self._spot1, self._spot2, self._spot3, self._spot4, self._spot5, self._spot6]

      self._spot1Taken = False
      self._spot2Taken = False
      self._spot3Taken = False
      self._spot4Taken = False
      self._spot5Taken = False
      self._spot6Taken = False

      self._spot1YumOn = False
      self._spot2YumOn = False
      self._spot3YumOn = False
      self._spot4YumOn = False
      self._spot5YumOn = False
      self._spot6YumOn = False

      self._yumOns = [self._spot1YumOn, self._spot2YumOn, self._spot3YumOn, self._spot4YumOn, self._spot5YumOn, \
                    self._spot6YumOn]

      self._yumSec1 = 2
      self._yumSec2 = 2
      self._yumSec3 = 2
      self._yumSec4 = 2
      self._yumSec5 = 2
      self._yumSec6 = 2

      self._yumTimers = [self._yumSec1, self._yumSec2, self._yumSec3, self._yumSec4, self._yumSec5, self._yumSec6]

      # was going to have "ugh"! picture appear after customer annoyed but also did not complete
      self._spot1UghOn = False
      self._spot2UghOn = False
      self._spot3UghOn = False
      self._spot4UghOn = False
      self._spot5UghOn = False
      self._spot6UghOn = False

      #customers
      self._customers = []
      self._customerTimer = random.randint(4,5)

      #ability related drawables
      self._pusheenWall = Drawable("pusheenWall.png", Vector2((SCREEN_SIZE.x // 2) - 40, (SCREEN_SIZE.y // 2) - 30))
      self._pusheenRect = pygame.Rect(((SCREEN_SIZE.x // 2) - 40, (SCREEN_SIZE.y // 2) - 30), (61, 64))

      self._samples = []
      for i in range (0, 50):
         temp = Drawable("sample.png", Vector2(random.randint(60, 690), random.randint(80, 377)))
         self._samples.append(temp)

      self._cake1 = Drawable("cake.png", Vector2((SCREEN_SIZE.x // 2) + 200, (SCREEN_SIZE.y // 2) - 50 - 15))
      self._cake2 = Drawable("cake.png", Vector2((SCREEN_SIZE.x // 2) + 275, (SCREEN_SIZE.y // 2) - 100 - 15))
      self._cake3 = Drawable("cake.png", Vector2(SCREEN_SIZE.x - 385, SCREEN_SIZE.y - 408))

      self._cakes = [self._cake1, self._cake2, self._cake3]

      self._salt1 = Salt(Vector2((SCREEN_SIZE.x // 2) + 205, (SCREEN_SIZE.y // 2) - 50 - 50))
      self._salt2 = Salt(Vector2((SCREEN_SIZE.x // 2) + 280, (SCREEN_SIZE.y // 2) - 100 - 50))
      self._salt3 = Salt(Vector2(SCREEN_SIZE.x - 380, SCREEN_SIZE.y - 438))

      self._yelp = Drawable("yelp.png", Vector2((SCREEN_SIZE.x // 2) - 320, (SCREEN_SIZE.y // 2) + 100))

      self._sign1 = Drawable("sign.png", Vector2((SCREEN_SIZE.x // 2) - 320, (SCREEN_SIZE.y // 2) - 200))
      self._sign2 = Drawable("sign.png", Vector2((SCREEN_SIZE.x // 2) - 160, (SCREEN_SIZE.y // 2) - 200))

      #ability booleans and timers
      self._firstSalt = True

      self._cakeIsSalted = False
      self._saltyCakeTimer = 8
      self._saltAbilityTimer = 25

      self._yelping = False
      self._yelpTimer = 8
      self._yelpAbilityTimer = 30

      self._sampling = False
      self._sampleTimer = 8
      self._sampleAbilityTimer = 30

      self._pusheenActivated = False
      self._pusheenTimer = 8
      self._pusheenAbilityTimer = 25

      #background
      self._background = Drawable("background.png", Vector2(0, 0))
      #funky block to make countertop at the end look normal
      self._counterSide = Drawable("counterside.png", Vector2(SCREEN_SIZE.x - 432, SCREEN_SIZE.y - 250))
      #player character creation
      self._bakerCat = BakerCat(Vector2(SCREEN_SIZE.x - 385, SCREEN_SIZE.y - 388))  # Starting position near table3 in kitchen
      self._karenCat = KarenCat(Vector2((SCREEN_SIZE.x // 2) - 100, (SCREEN_SIZE.y // 2) - 10))
   
   def draw(self, drawSurface):

      self._background.draw(drawSurface)
      self._counterSide.draw(drawSurface)
      self._table1.draw(drawSurface)
      self._table2.draw(drawSurface)
      self._table3.draw(drawSurface)
      self._cake1.draw(drawSurface)
      self._cake2.draw(drawSurface)
      self._cake3.draw(drawSurface)

      self._karenBar.draw(drawSurface)
      self._bakerBar.draw(drawSurface)
      self._karenIcon.draw(drawSurface)
      self._bakerIcon.draw(drawSurface)


      self._colorSaltIcon.draw(drawSurface)


      for fridge in self._fridges:
         fridge.draw(drawSurface)

      for oven in self._ovens:
         oven.draw(drawSurface)

      for counter in self._counters:
         counter.draw(drawSurface)

      for countertop in self._countertops:
         countertop.draw(drawSurface)


      if self._saltAbilityTimer > 0:
         self._bwSaltIcon.draw(drawSurface)

      if self._saltAbilityTimer <= 0:
         self._colorSaltIcon.draw(drawSurface)

      if self._pusheenAbilityTimer > 0:
         self._bwPusheenIcon.draw(drawSurface)

      if self._pusheenAbilityTimer <= 0:
         self._colorPusheenIcon.draw(drawSurface)

      if self._yelpAbilityTimer > 0:
         self._bwYelpIcon.draw(drawSurface)

      if self._yelpAbilityTimer <= 0:
         self._colorYelpIcon.draw(drawSurface)

      if self._sampleAbilityTimer > 0:
         self._bwSampleIcon.draw(drawSurface)

      if self._sampleAbilityTimer <= 0:
         self._colorSampleIcon.draw(drawSurface)

      if self._pusheenActivated:
         self._pusheenWall.draw(drawSurface)

      if self._yelping:
         self._yelp.draw(drawSurface)

      if self._sampling:
         self._sign1.draw(drawSurface)
         self._sign2.draw(drawSurface)
         for sample in self._samples:
            sample.draw(drawSurface)

      if self._cakeIsSalted:
         self._salt1.draw(drawSurface)
         self._salt2.draw(drawSurface)
         self._salt3.draw(drawSurface)


      for customer in self._customers:
         if not customer.isServed():
            customer.draw(drawSurface)

      for i in range(0, 6):
         if self._yumOns[i]:
            self._yums[i].draw(drawSurface)

      self._bakerCat.draw(drawSurface)
      self._karenCat.draw(drawSurface)
   
   
   def handleEvent(self, event):
      #cheat codes to test ending screens
      if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
         self._karenBarCurrent = 1000
      if event.type == pygame.KEYDOWN and event.key == pygame.K_2:
         self._bakerBarCurrent = 1000

      self._bakerCat.handleEvent(event)
      self._karenCat.handleEvent(event)
   
   
   def update(self, seconds, SCREEN_SIZE):
      #check for game win
      if self._bakerBarCurrent >= 100:
            return "bakerWin"
      if self._karenBarCurrent >= 100:
            return "karenWin"

      #basic updates
      self._bakerCat.update(seconds)
      self._karenCat.update(seconds)

      if self._cakeIsSalted:
         self._salt1.update(seconds)
         self._salt2.update(seconds)
         self._salt3.update(seconds)

      #timers decrease
      self._customerTimer -= seconds
      self._saltAbilityTimer -= seconds
      self._pusheenAbilityTimer -= seconds
      self._yelpAbilityTimer -= seconds
      self._sampleAbilityTimer -= seconds

      if len(self._samples) < 50:
         while len(self._samples) < 50:
            temp = Drawable("sample.png", Vector2(random.randint(60, 690), random.randint(80, 377)))
            self._samples.append(temp)


      #customer spawn
      if self._customerTimer <= 0 and len(self._customers) < 6:
         temp = random.randint(0, 5)
         if temp == 3 and not self._spot4Taken:
            self._customers.append(CustomerCat(Vector2(self._spot4.x, self._spot4.y)))
            self._spot4Taken = True
            self._customerTimer = random.randint(2, 5)
         if temp == 1 and not self._spot2Taken:
            self._customers.append(CustomerCat(Vector2(self._spot2.x, self._spot2.y)))
            self._spot2Taken = True
            self._customerTimer = random.randint(2, 5)
         if temp == 4 and not self._spot5Taken:
            self._customers.append(CustomerCat(Vector2(self._spot5.x, self._spot5.y)))
            self._spot5Taken = True
            self._customerTimer = random.randint(2, 5)
         if temp == 2 and not self._spot3Taken:
            self._customers.append(CustomerCat(Vector2(self._spot3.x, self._spot3.y)))
            self._spot3Taken = True
            self._customerTimer = random.randint(2, 5)
         if temp == 0 and not self._spot1Taken:
            self._customers.append(CustomerCat(Vector2(self._spot1.x, self._spot1.y)))
            self._spot1Taken = True
            self._customerTimer = random.randint(2, 5)
         if temp == 5 and not self._spot6Taken:
            self._customers.append(CustomerCat(Vector2(self._spot6.x, self._spot6.y)))
            self._spot6Taken = True
            self._customerTimer = random.randint(2, 5)

      for customer in self._customers:
         if not customer.isServed():
            customer.update(seconds)

      #would be for the yum! notification after customer served
      for timer in range(0, 6):
         if self._yumTimers[timer] <= 0:
            self._yumTimers[timer] = 2
            self._yumOns[timer] = False


      #ability related
      if self._cakeIsSalted:
         self._saltyCakeTimer -= seconds
         if self._saltyCakeTimer <= 0:
            self._cakeIsSalted = False
            self._firstSalt = True
            self._saltyCakeTimer = 5
      
      if key_states[pygame.K_v]:  
         if self._pusheenAbilityTimer <= 0:
            self._pusheenActivated = True

      if key_states[pygame.K_j]:  
         if self._yelpAbilityTimer <= 0:
            self._yelping = True

      if key_states[pygame.K_x]:  
         if self._sampleAbilityTimer <= 0:
            self._sampling = True


      if self._sampling:
         self._sampleTimer -= seconds
         if self._sampleTimer <= 0:
            self._sampling = False
            self._sampleTimer = 8
            self._sampleAbilityTimer = 30


      if self._yelping:
         self._yelpTimer -= seconds
         temp = (seconds * 2)
         self._bakerBarCurrent = max(0, self._bakerBarCurrent - temp)
         self._bakerBar = BarItem(Vector2(410, 400), 100, 20, self._bakerBarCurrent, 100, color=(255, 91, 165))
         if self._yelpTimer <= 0:
            self._yelping = False
            self._yelpTimer = 8
            self._yelpAbilityTimer = 30

      if self._pusheenActivated:
         self._pusheenTimer -= seconds
         if self._pusheenTimer <= 0:
            self._pusheenActivated = False
            self._pusheenTimer = 12
            self._pusheenAbilityTimer = 15


         # Detect collision
      if self._pusheenActivated:
         if self._bakerCat.getCollisionRect().colliderect(self._pusheenRect):
            if self._bakerCat.getPosition().x < self._pusheenRect.x:
               self._bakerCat._state.manageState("stopright", self._bakerCat)

            elif self._bakerCat.getPosition().y > self._pusheenRect.y:
               self._bakerCat._state.manageState("stopup", self._bakerCat)

            elif self._bakerCat.getPosition().y < self._pusheenRect.y:
               self._bakerCat._state.manageState("stopdown", self._bakerCat)

            elif self._bakerCat.getPosition().x > self._pusheenRect.x:
               self._bakerCat._state.manageState("stopleft", self._bakerCat)

         if self._karenCat.getCollisionRect().colliderect(self._pusheenRect):
            if self._karenCat.getPosition().x < self._pusheenRect.x:
               self._karenCat._state.manageState("stopright", self._karenCat)

            elif self._karenCat.getPosition().y > self._pusheenRect.y:
               self._karenCat._state.manageState("stopup", self._karenCat)

            elif self._karenCat.getPosition().y < self._pusheenRect.y:
               self._karenCat._state.manageState("stopdown", self._karenCat)

            elif self._karenCat.getPosition().x > self._pusheenRect.x:
               self._karenCat._state.manageState("stopleft", self._karenCat)


      for sample in self._samples:
         if self._sampling:
            if self._karenCat.getCollisionRect().colliderect(sample.getCollisionRect()):
               temp = random.randint(2, 3)
               self._karenBarCurrent = max(0, self._karenBarCurrent - temp)
               self._karenBar = BarItem(Vector2(30, 400), 100, 20, self._karenBarCurrent, 100, color=(135, 206, 250))
               self._samples.remove(sample)

      if self._yelping:
         if (self._bakerCat.getCollisionRect().colliderect(self._yelp.getCollisionRect()) 
             and key_states[pygame.K_c]):  
            self._yelping = False # turn off wifi
            self._yelpTimer = 8
            self._yelpAbilityTimer = 30

      for cake in self._cakes:
        
         if self._karenCat.getCollisionRect().colliderect(cake.getCollisionRect()) and key_states[pygame.K_k]:  
            if self._firstSalt:
               temp = random.randint(3,5)
               self._bakerBarCurrent = max(0, self._bakerBarCurrent - temp)
               self._bakerBar = BarItem(Vector2(410, 400), 100, 20, self._bakerBarCurrent, 100, color=(255, 91, 165))
               self._firstSalt = False
            if self._saltAbilityTimer <= 0:
               self._cakeIsSalted = True
               self._saltAbilityTimer = 15


      for customer in self._customers:
         if self._bakerCat.getCollisionRect().colliderect(customer.getCollisionRect()):
            if key_states[pygame.K_z]:
               #this for would-be yum! part
               if customer.getCollisionRect().colliderect(self._spot2):
                  self._spot2Taken = False
                  self._spot2YumOn = True
                  self._yumSec2 -= seconds
               if customer.getCollisionRect().colliderect(self._spot3):
                  self._spot3Taken = False
                  self._spot3YumOn = True
                  self._yumSec3 -= seconds
               if customer.getCollisionRect().colliderect(self._spot4):
                  self._spot4Taken = False
                  self._spot4YumOn = True
                  self._yumSec4 -= seconds
               if customer.getCollisionRect().colliderect(self._spot5):
                  self._spot5Taken = False
                  self._spot5YumOn = True
                  self._yumSec5 -= seconds
               if customer.getCollisionRect().colliderect(self._spot1):
                  self._spot1Taken = False
                  self._spot1YumOn = True
                  self._yumSec1 -= seconds
               if customer.getCollisionRect().colliderect(self._spot6):
                  self._spot6Taken = False
                  self._spot6YumOn = True
                  self._yumSec6 -= seconds

               customer.getServed()

               if self._cakeIsSalted:
                  self._karenBarCurrent += random.randint(2, 4)
                  self._karenBar = BarItem(Vector2(30, 400), 100, 20, self._karenBarCurrent, 100, color=(135, 206, 250))
               else:
                  self._bakerBarCurrent += random.randint(2, 4)
               self._bakerBar = BarItem(Vector2(410, 400), 100, 20, self._bakerBarCurrent, 100, color=(255, 91, 165))
               self._customers.remove(customer)

         elif self._karenCat.getCollisionRect().colliderect(customer.getCollisionRect()):
            if key_states[pygame.K_l]:
               if customer.getCollisionRect().colliderect(self._spot2):
                  self._spot2Taken = False
               if customer.getCollisionRect().colliderect(self._spot3):
                  self._spot3Taken = False
               if customer.getCollisionRect().colliderect(self._spot4):
                  self._spot4Taken = False
               if customer.getCollisionRect().colliderect(self._spot5):
                  self._spot5Taken = False
               if customer.getCollisionRect().colliderect(self._spot1):
                  self._spot1Taken = False
               if customer.getCollisionRect().colliderect(self._spot6):
                  self._spot6Taken = False
               customer.getAnnoyed()
               self._karenBarCurrent += random.randint(1,3)
               self._karenBar = BarItem(Vector2(30, 400), 100, 20, self._karenBarCurrent, 100, color=(135,206,250))
               self._customers.remove(customer)



      #furniture and world bounds collision; kept table as basic keyword for all of the temp furniture variable-- made it easy to copy and paste
      if self._bakerCat.getCollisionRect().colliderect(self._counterRect):
         if self._bakerCat.getPosition().x < self._counterRect.x:
            self._bakerCat._state.manageState("stopright", self._bakerCat)

         if self._bakerCat.getPosition().y < self._counterRect.y:
            self._bakerCat._state.manageState("stopdown", self._bakerCat)

         if self._bakerCat.getPosition().y < self._counterRect.y:
            self._bakerCat._state.manageState("stopup", self._bakerCat)

      if self._bakerCat.getCollisionRect().colliderect(self._countertopRect):
         if self._bakerCat.getPosition().x < self._countertopRect.x:
            self._bakerCat._state.manageState("stopright", self._bakerCat)

         if self._bakerCat.getPosition().x > self._countertopRect.x:
            self._bakerCat._state.manageState("stopleft", self._bakerCat)

         if self._bakerCat.getPosition().y > self._countertopRect.y:
            self._bakerCat._state.manageState("stopup", self._bakerCat)

      if self._karenCat.getCollisionRect().colliderect(self._counterRect):
         if self._karenCat.getPosition().x < self._counterRect.x:
            self._karenCat._state.manageState("stopright", self._karenCat)

         if self._karenCat.getPosition().y > self._counterRect.y:
            self._karenCat._state.manageState("stopup", self._karenCat)

         if self._karenCat.getPosition().y < self._counterRect.y:
            self._karenCat._state.manageState("stopdown", self._karenCat)

      if self._karenCat.getCollisionRect().colliderect(self._countertopRect):
         if self._karenCat.getPosition().x < self._countertopRect.x:
            self._karenCat._state.manageState("stopright", self._karenCat)

         if self._karenCat.getPosition().x > self._countertopRect.x:
            self._karenCat._state.manageState("stopleft", self._karenCat)

         if self._karenCat.getPosition().y > self._countertopRect.y:
            self._karenCat._state.manageState("stopup", self._karenCat)


      for table in self._cakes:
         if self._bakerCat.getCollisionRect().colliderect(table.getCollisionRect()):
            if self._bakerCat.getPosition().x < table.getPosition().x:
               self._bakerCat._state.manageState("stopright", self._bakerCat)

            if self._bakerCat.getPosition().x > table.getPosition().x:
               self._bakerCat._state.manageState("stopleft", self._bakerCat)

            if self._bakerCat.getPosition().y > table.getPosition().y:
               self._bakerCat._state.manageState("stopup", self._bakerCat)

            if self._bakerCat.getPosition().y < table.getPosition().y:
               self._bakerCat._state.manageState("stopdown", self._bakerCat)

         if self._karenCat.getCollisionRect().colliderect(table.getCollisionRect()):
            if self._karenCat.getPosition().x < table.getPosition().x:
               self._karenCat._state.manageState("stopright", self._karenCat)

            if self._karenCat.getPosition().x > table.getPosition().x:
               self._karenCat._state.manageState("stopleft", self._karenCat)

            if self._karenCat.getPosition().y > table.getPosition().y:
               self._karenCat._state.manageState("stopup", self._karenCat)

            if self._karenCat.getPosition().y < table.getPosition().y:
               self._karenCat._state.manageState("stopdown", self._karenCat)

      for table in self._ovens:
         if self._bakerCat.getCollisionRect().colliderect(table.getCollisionRect()):
            if self._bakerCat.getPosition().x < table.getPosition().x:
               self._bakerCat._state.manageState("stopright", self._bakerCat)

            if self._bakerCat.getPosition().x > table.getPosition().x:
               self._bakerCat._state.manageState("stopleft", self._bakerCat)

            if self._bakerCat.getPosition().y > table.getPosition().y:
               self._bakerCat._state.manageState("stopup", self._bakerCat)

            if self._bakerCat.getPosition().y < table.getPosition().y:
               self._bakerCat._state.manageState("stopdown", self._bakerCat)

         if self._karenCat.getCollisionRect().colliderect(table.getCollisionRect()):
            if self._karenCat.getPosition().x < table.getPosition().x:
               self._karenCat._state.manageState("stopright", self._karenCat)

            if self._karenCat.getPosition().x > table.getPosition().x:
               self._karenCat._state.manageState("stopleft", self._karenCat)

            if self._karenCat.getPosition().y > table.getPosition().y:
               self._karenCat._state.manageState("stopup", self._karenCat)

            if self._karenCat.getPosition().y < table.getPosition().y:
               self._karenCat._state.manageState("stopdown", self._karenCat)

      for table in self._fridges:
         if self._bakerCat.getCollisionRect().colliderect(table.getCollisionRect()):
            if self._bakerCat.getPosition().x < table.getPosition().x:
               self._bakerCat._state.manageState("stopright", self._bakerCat)

            if self._bakerCat.getPosition().x > table.getPosition().x:
               self._bakerCat._state.manageState("stopleft", self._bakerCat)

            if self._bakerCat.getPosition().y > table.getPosition().y:
               self._bakerCat._state.manageState("stopup", self._bakerCat)

            if self._bakerCat.getPosition().y < table.getPosition().y:
               self._bakerCat._state.manageState("stopdown", self._bakerCat)

         if self._karenCat.getCollisionRect().colliderect(table.getCollisionRect()):
            if self._karenCat.getPosition().x < table.getPosition().x:
               self._karenCat._state.manageState("stopright", self._karenCat)

            if self._karenCat.getPosition().x > table.getPosition().x:
               self._karenCat._state.manageState("stopleft", self._karenCat)

            if self._karenCat.getPosition().y > table.getPosition().y:
               self._karenCat._state.manageState("stopup", self._karenCat)

            if self._karenCat.getPosition().y < table.getPosition().y:
               self._karenCat._state.manageState("stopdown", self._karenCat)

      if self._bakerCat.getCollisionRect().colliderect(self._worldTop):
         self._bakerCat._state.manageState("stopup", self._bakerCat)

      if self._bakerCat.getCollisionRect().colliderect(self._worldBottom):
         self._bakerCat._state.manageState("stopdown", self._bakerCat)

      if self._bakerCat.getCollisionRect().colliderect(self._worldLeft):
         self._bakerCat._state.manageState("stopleft", self._bakerCat)

      if self._bakerCat.getCollisionRect().colliderect(self._worldRight):
         self._bakerCat._state.manageState("stopright", self._bakerCat)

      if self._karenCat.getCollisionRect().colliderect(self._worldTop):
         self._karenCat._state.manageState("stopup", self._karenCat)

      if self._karenCat.getCollisionRect().colliderect(self._worldBottom):
         self._karenCat._state.manageState("stopdown", self._karenCat)

      if self._karenCat.getCollisionRect().colliderect(self._worldLeft):
         self._karenCat._state.manageState("stopleft", self._karenCat)

      if self._karenCat.getCollisionRect().colliderect(self._worldRight):
         self._karenCat._state.manageState("stopright", self._karenCat)

      # print cat positions after other updates
      #print("Baker Cat Position:", self._bakerCat.getPosition())
      #print("Karen Cat Position:", self._karenCat.getPosition())
      #print()
      #print()
   
   def updateMovement(self):
      self._bakerCat.updateMovement()
      self._karenCat.updateMovement()
