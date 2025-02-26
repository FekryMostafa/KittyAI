"""
A Singleton Frame Manager class
Author: Liz Matthews, 9/20/2019

Provides on-demand loading of images for a pygame program. Will load entire sprite sheets if given an offset.

"""


from pygame import image, Surface, Rect, SRCALPHA
from os.path import join


class FrameManager(object):
   """A singleton factory class to create and store frames on demand."""
   
   # The singleton instance variable
   _INSTANCE = None
   
   @classmethod
   def getInstance(cls):
      """Used to obtain the singleton instance"""
      if cls._INSTANCE == None:
         cls._INSTANCE = cls._FM()
      
      return cls._INSTANCE

   # Do not directly instantiate this class!
   class _FM(object):
      """An internal FrameManager class to contain the actual code. Is a private class."""
      
      # Folder in which images are stored
      _IMAGE_FOLDER = join("resources", "images")
      
      # Static information about the frame sizes of particular image sheets.
      _FRAME_SIZES = {
         "fridge.png": (32, 64),
         "arrow.png": (8, 8),
         "counter.png": (24, 35),
         "countertop.png": (24, 24),
         "bakercat.png": (48, 48),
         "karencat.png": (48, 48),
         "customer1.png": (48, 48),
         "customer2.png": (48, 48),
         "customer3.png": (48, 48),
         "table.png": (50, 50),
         "cake": (50, 43),
         "oven.png": (38, 50),
         "pusheenWall.png": (64, 61),
         "placeHolder.png": (60, 60),
         "salt.png": (40, 40),
         "bakerIcon.png": (25, 25),
         "karenIcon.png": (25, 25),
         "colorSaltIcon.png": (30, 30),
         "bwSaltIcon.png": (30, 30),
         "colorPusheenIcon.png": (30, 30),
         "bwPusheenIcon.png": (30, 30),
         "yum.png": (40, 40),
         "yelp.png": (80, 80),
         "bwYelpIcon.png": (30, 30),
         "colorYelpIcon.png": (30, 30),
         "sample.png": (20, 20),
         "colorSampleIcon.png": (30, 30),
         "bwSampleIcon.png": (30, 30),
         "sign.png": (90, 71)
      }
      
      # A default frame size
      _DEFAULT_FRAME = (32, 32)
      
      # A list of images that require to be loaded with transparency
      _TRANSPARENCY = ["fridge.png", "bakercat.png", "karencat.png", "customer1.png", "customer2.png", "customer3.png", "table.png", \
                       "cake.png", "oven.png", "pusheenWall.png", "placeHolder.png", "salt.png", "bakerIcon.png", "karenIcon.png", \
                       "colorSaltIcon.png", "bwSaltIcon.png", "colorPusheenIcon.png", "bwPusheenIcon.png", "yum.png", "yelp.png", \
                       "colorYelpIcon.png", "bwYelpIcon.png", "sample.png", "colorSampleIcon.png", "bwSampleIcon.png", "sign.png"]
      
      # A list of images that require to be loaded with a color key
      _COLOR_KEY = ["arrow.png"]
      
      
      
      def __init__(self):
         # Stores the surfaces indexed based on file name
         # The values in _surfaces can be a single Surface
         #  or a two dimentional grid of surfaces if it is an image sheet
         self._surfaces = {}
      
      
      def __getitem__(self, key):
         return self._surfaces[key]
   
      def __setitem__(self, key, item):
         self._surfaces[key] = item
      
      
      def getFrame(self, fileName, offset=None):
         # If this frame has not already been loaded, load the image from memory
         if fileName not in self._surfaces.keys():
            self._loadImage(fileName, offset != None)
         
         # If this is an image sheet, return the correctly offset sub surface
         if offset != None:
            return self[fileName][offset[0]][offset[1]]
         
         # Otherwise, return the sheet created
         return self[fileName]
      
      def _loadImage(self, fileName, sheet=False):
         # Load the full image
         fullImage = image.load(join("sourceCode", FrameManager._FM._IMAGE_FOLDER, fileName))
         
         # Look up some information about the image to be loaded
         transparent = fileName in FrameManager._FM._TRANSPARENCY
         colorKey = fileName in FrameManager._FM._COLOR_KEY
         
         # Detect if a transparency is needed
         if transparent:
            fullImage = fullImage.convert_alpha()
         else:
            fullImage = fullImage.convert()
         
         # If the image to be loaded is an image sheet, split it up based on the frame size
         if sheet:
               
            self[fileName] = []
            
            # Try to get the sprite size, use the default size if it is not stored
            spriteSize = FrameManager._FM._FRAME_SIZES.get(fileName, FrameManager._FM._DEFAULT_FRAME)
            
            # See how big the sprite sheet is
            sheetDimensions = fullImage.get_size()
            
            # Iterate over the entire sheet, increment by the sprite size
            for y in range(0, sheetDimensions[1], spriteSize[1]):
               self[fileName].append([])
               for x in range(0, sheetDimensions[0], spriteSize[0]):
                  
                  # If we need transparency
                  if transparent:
                     frame = Surface(spriteSize, SRCALPHA, 32)
                  else:
                     frame = Surface(spriteSize)
                  
                  frame.blit(fullImage, (0,0), Rect((x,y), spriteSize))
                  
                  # If we need to set the color key
                  if colorKey:
                     frame.set_colorkey(frame.get_at((0,0)))
                  
                  # Add the frame to the end of the current row
                  self[fileName][-1].append(frame)
         else:
            # Not a sprite sheet, full image is what we wish to store
            self[fileName] = fullImage
               
            # If we need to set the color key
            if colorKey:
               self[fileName].set_colorkey(self[fileName].get_at((0,0)))
               
            
         