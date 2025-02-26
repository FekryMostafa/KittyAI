from ..gameObjects.vector2D import Vector2


SCREEN_SIZE = Vector2(765, 428)
SCALE = 2
UPSCALED = SCREEN_SIZE * SCALE


def adjustMousePos(mousePos):
   return Vector2(*mousePos) // SCALE