import numpy as np
import os
import cv2
import pygame
import socket
import threading
import time
import sys
from modules.managers.gameManager import key_states
from modules.managers.screenManager import ScreenManager
from modules.managers.gameManager import GameManager
from modules.UI.screenInfo import SCREEN_SIZE, UPSCALED

# Constants for command processing
COMMAND_DELAY = 0.2
ABILITY_DELAY = 2.0
#-58
# Dictionary to map commands to corresponding pygame key constants
COMMAND_MAPPING = {
    "LEFT_DOWN": pygame.K_LEFT,
    "RIGHT_DOWN": pygame.K_RIGHT,
    "UP_DOWN": pygame.K_UP,
    "DOWN_DOWN": pygame.K_DOWN,
    "A_DOWN": pygame.K_a,
    "D_DOWN": pygame.K_d,
    "W_DOWN": pygame.K_w,
    "S_DOWN": pygame.K_s
}

ABILITY_MAPPING = {
    "K_DOWN": pygame.K_k,
    "J_DOWN": pygame.K_j,
    "L_DOWN": pygame.K_l,
    "Z_DOWN": pygame.K_z,
    "X_DOWN": pygame.K_x,
    "V_DOWN": pygame.K_v,
    "C_DOWN": pygame.K_c
}

# Global variables
is_paused = False


def process_command(command):
    """
    Process the given command by posting corresponding pygame events.
    """
    #print("PROCESSING COMMAND")
    pygame_key = COMMAND_MAPPING.get(command)
    if pygame_key is not None:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame_key))
        time.sleep(COMMAND_DELAY)
        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame_key))


def process_ability(command):
#     ""
#     Process special ability commands by posting corresponding pygame events and updating key states.
#     """
    #print("PROCESSING ABILITY")
    pygame_key = ABILITY_MAPPING.get(command)
    if pygame_key is not None:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame_key))
        key_states[pygame_key] = True
        time.sleep(ABILITY_DELAY)
        key_states[pygame_key] = False
        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame_key))

def get_observation():
      # Get the 2D array of mapped color values
     mapped_colors = pygame.surfarray.array2d(pygame.display.get_surface())

     # Convert mapped colors to RGB
     rgb_array = pygame.surfarray.make_surface(mapped_colors)
     pixel_data = pygame.surfarray.array3d(rgb_array)
     pixel_data = np.transpose(pixel_data, (1, 0, 2))  # Reorder the axes for OpenCV

     # Convert to grayscale
     grayscale = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2GRAY)

     # Resize the image
     resized = cv2.resize(grayscale, (84, 84))

     # Normalize the pixel values
     normalized = resized / 255.0

     return normalized.reshape(84, 84, 1)

def listen_for_commands(screen_manager, port):
    global is_paused
    retries = 5
    delay = 1

    while retries > 0:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Set socket options for reuse
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                s.listen()
                print(f"Listening on port {port}")
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    while True:
                        data = conn.recv(81920)
                        if not data:
                            break
                        command = data.decode()
                        # Unpause the game to process the command
                        is_paused = False
                        time.sleep(0.1)
                        
                        # Process the command
                        command += "_DOWN"
                        if command in COMMAND_MAPPING or command in ABILITY_MAPPING:
                            if command in COMMAND_MAPPING:
                                process_command(command)
                            elif command in ABILITY_MAPPING:
                                process_ability(command)

                            # Get game state
                            try:
                                karen_cat_position = screen_manager._game._karenCat.getPosition()
                                karen_cat_score = screen_manager._game._karenBarCurrent
                                customers = screen_manager._game._customers

                                # Format game state
                                delimiter = ","
                                ability = "False"
                                customer_positions = [(customer.getPosition().x, customer.getPosition().y) 
                                                    for customer in customers]
                                customer_positions_str = delimiter.join([f"{pos[0]}_{pos[1]}" 
                                                                      for pos in customer_positions])

                                # Create state string
                                full_string = (f"{karen_cat_position[0]}_{karen_cat_position[1]}"
                                             f"{delimiter}{karen_cat_score}"
                                             f"{delimiter}{screen_manager._game._saltAbilityTimer}"
                                             f"{delimiter}{screen_manager._game._yelpAbilityTimer}"
                                             f"{delimiter}{ability}"
                                             f"{delimiter}{customer_positions_str}")

                                # Send state
                                conn.sendall(full_string.encode())
                            except Exception as e:
                                print(f"Error getting game state: {e}")
                                conn.sendall(b"0_0,0,0,0,False")
                                
                        is_paused = True
                break  # Exit after successful connection and processing
        except OSError as e:
            print(f"Socket error: {e}")
            retries -= 1
            if retries > 0:
                print(f"Retrying in {delay} seconds... ({retries} attempts left)")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("Failed to establish socket connection")
                break

def main():
    global is_paused
    
    # Get port number from command line arguments
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    
    pygame.init()
    pygame.display.set_caption("Kitty Karen Wars")
    screen = pygame.display.set_mode(list(UPSCALED))
    draw_surface = pygame.Surface(list(SCREEN_SIZE))

    screen_manager = ScreenManager()
    game_clock = pygame.time.Clock()
    game_manager = GameManager(SCREEN_SIZE)
    screen_manager._state.manageState("startGame", screen_manager)
    
    # Start command listener thread with port
    threading.Thread(target=listen_for_commands, 
                    args=(screen_manager, port), 
                    daemon=True).start()

    running = True
    while running:
        if not is_paused:
            screen_manager.draw(draw_surface)
            pygame.transform.scale(draw_surface, list(UPSCALED), screen)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                result = screen_manager.handleEvent(event)
                if result == "exit":
                    running = False
                    break

            game_clock.tick(60)
            seconds = min(0.5, game_clock.get_time() / 1000)
            screen_manager.update(seconds)
            #if screen_manager._state == "gameKaren" or "gameBaker":
            #    screen_manager._state.manageState("startGame",screen_manager)

        else:
            time.sleep(0.1)

if __name__ == "__main__":
    main()