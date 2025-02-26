import datetime
import os
import socket
import random
import subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.models import clone_model
from keras.layers import Dense, Input, concatenate, Multiply, Add, Subtract, Lambda
from keras.models import Model
from keras import backend as K
import time
from sourceCode.modules.UI.screenInfo import SCREEN_SIZE
import metrics
import math
import glob

COMMANDSKAREN = ["UP", "DOWN", "RIGHT", "LEFT","L", "J", "K"]
COMMANDSBAKER = ["W", "S", "A", "D", "Z", "X", "V", "C"]

MAX_CUSTOMERS = 5
MAX_CAKES = 3
CUSTOMER_INTERACT_DISTANCE = 10
CAKE_INTERACT_DISTANCE = 5

# Enhanced hyperparameters
MEMORY_SIZE = 100000
BATCH_SIZE = 128
LEARNING_RATE = 0.00025
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 100000
TARGET_UPDATE_FREQ = 1000
MAX_STEPS = 100
WARMUP_STEPS = 100

def create_model(state_size, action_size):
    """Create a more sophisticated network with attention mechanism"""
    # Input layer
    input_layer = Input(shape=(state_size,))
    
    # Spatial features branch
    spatial = Dense(256, activation='relu')(input_layer)
    spatial = Dense(256, activation='relu')(spatial)
    
    # Game state features branch
    game_state = Dense(128, activation='relu')(input_layer)
    game_state = Dense(128, activation='relu')(game_state)
    
    # Attention mechanism
    attention = Dense(256, activation='tanh')(input_layer)
    attention = Dense(1, activation='sigmoid')(attention)
    
    # Combine branches with attention
    combined = concatenate([spatial, game_state])
    combined = Dense(256, activation='relu')(combined)
    combined = Multiply()([combined, attention])
    
    # Action advantage and state value streams (Dueling DQN)
    advantage_stream = Dense(128, activation='relu')(combined)
    advantage = Dense(action_size)(advantage_stream)
    
    value_stream = Dense(128, activation='relu')(combined)
    value = Dense(1)(value_stream)
    
    # Combine streams
    outputs = Add()([value, Subtract()([advantage, 
                                       Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)])])
    
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
                 loss=tf.keras.losses.Huber())
    return model

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment  # Beta annealing
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # New experiences get max priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch with prioritized experience replay"""
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)]
        else:
            probs = self.priorities
            
        # Calculate sampling probabilities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize
        
        # Increase beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Unpack samples
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error + 1e-5  # Small constant to ensure non-zero priority
            self.max_priority = max(self.max_priority, self.priorities[idx])
            
    def __len__(self):
        return len(self.buffer)

class EnhancedDQNAgent:
    def __init__(self, state_size, character="karen"):
        self.state_size = state_size
        self.character = character
        self.action_size = len(COMMANDSKAREN)
        
        # Create main and target networks
        self.model = create_model(state_size, self.action_size)
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
        # Initialize replay buffer with prioritized experience replay
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=0.6, beta=0.4)
        
        # Initialize tracking variables
        self.epsilon = EPSILON_START
        self.steps = 0
        self.last_action = ""
        self.previous_closest_customer_dist = float('inf')
        self.previous_closest_cake_dist = float('inf')
        self.last_reward = 0
        self.cumulative_reward = 0
        
        # Initialize action tracking
        self.action_counts = {action: 0 for action in COMMANDSKAREN}
        self.successful_actions = {action: 0 for action in COMMANDSKAREN}
        
        # Initialize opponent modeling
        self.opponent_action_history = []
        self.opponent_position_history = []
        
    def update_epsilon(self):
        """Adaptive epsilon decay based on performance"""
        if self.cumulative_reward > 0:
            # Decay faster if performing well
            self.epsilon = max(EPSILON_END, 
                             self.epsilon * 0.995)
        else:
            # Decay slower if not performing well
            self.epsilon = max(EPSILON_END, 
                             self.epsilon * 0.999)
        
    def act(self, state, available_actions_indexes):
        """Choose action with Double DQN and strategic exploration"""
        self.steps += 1
        self.update_epsilon()
        
        # Strategic exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            # Smart exploration - bias towards underutilized actions
            action_probs = []
            for idx in available_actions_indexes:
                action = COMMANDSKAREN[idx]
                # Calculate success rate for this action
                success_rate = self.successful_actions[action] / max(1, self.action_counts[action])
                # Higher probability for actions with higher success rate
                action_probs.append(0.1 + success_rate)
            
            # Normalize probabilities
            action_probs = np.array(action_probs) / sum(action_probs)
            action_idx = np.random.choice(len(available_actions_indexes), p=action_probs)
            action = available_actions_indexes[action_idx]
            q_value = 0.0
        else:
            # Exploitation with Double DQN
            state = np.array(state).reshape(1, -1)
            q_values = self.model.predict(state, verbose=0)
            
            # Filter for available actions
            valid_q_values = [q_values[0][i] for i in available_actions_indexes]
            max_q_idx = np.argmax(valid_q_values)
            action = available_actions_indexes[max_q_idx]
            q_value = valid_q_values[max_q_idx]
            
        return action, q_value
        
    def train(self):
        """Train with Prioritized Experience Replay and Double DQN"""
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE)
        
        # Convert to numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        weights = np.array(weights)
        
        # Ensure states have correct shape
        if len(states.shape) == 3:
            states = states.reshape(states.shape[0], -1)
        if len(next_states.shape) == 3:
            next_states = next_states.reshape(next_states.shape[0], -1)
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Double DQN: use main network to select actions, target network to evaluate them
        next_q_main = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        
        # Calculate TD errors for prioritized replay
        td_errors = np.zeros(BATCH_SIZE)
        
        for i in range(BATCH_SIZE):
            if dones[i]:
                target_q = rewards[i]
            else:
                # Double DQN update
                best_action = np.argmax(next_q_main[i])
                target_q = rewards[i] + GAMMA * next_q_target[i][best_action]
            
            # Calculate TD error
            td_errors[i] = abs(target_q - current_q[i][actions[i]])
            
            # Update Q-value
            current_q[i][actions[i]] = target_q
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Train with importance sampling weights
        self.model.fit(states, current_q, sample_weight=weights, batch_size=BATCH_SIZE, verbose=0)
        
        # Soft update target network
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Soft update target network"""
        tau = 0.01  # Soft update parameter
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
            
        self.target_model.set_weights(target_weights)
        
    def get_reward(self, agent_location, old_location, customers, agent_action, score_delta, ability_active, opponent_score=0, yelping=False):
        """Generic reward function to be overridden by specific agent classes"""
        # Basic reward based on score change
        reward = score_delta * 10
        
        # Parse locations
        agent_x, agent_y = map(float, agent_location.split('_'))
        old_x, old_y = map(float, old_location.split('_'))
        agent_pos = np.array([agent_x, agent_y])
        old_pos = np.array([old_x, old_y])
        
        # Reward for movement
        movement_dist = np.linalg.norm(agent_pos - old_pos)
        if movement_dist > 0:
            reward += 1  # Small reward for movement
            
        # Win condition
        if score_delta >= 100:
            reward += 1000
            
        return reward

def parse_state(state_string_list, character="karen"):
    """Enhanced state representation with spatial awareness and game context"""
    try:
        # Parse basic positions
        karen_pos = np.array([float(x) for x in state_string_list[0].split('_')])
        karen_score = float(state_string_list[1])
        salt_timer = float(state_string_list[2])
        yelp_timer = float(state_string_list[3])
        ability_active = 1.0 if state_string_list[4] == 'True' else 0.0
        
        # Parse customer positions
        customer_positions = []
        for i in range(5, len(state_string_list), 2):
            if i+1 < len(state_string_list) and state_string_list[i] and state_string_list[i+1]:
                try:
                    x = float(state_string_list[i])
                    y = float(state_string_list[i+1])
                    customer_positions.append(x)
                    customer_positions.append(y)
                except ValueError:
                    continue  # Skip invalid values
        
        # Pad customer positions to ensure consistent dimensions
        while len(customer_positions) < MAX_CUSTOMERS * 2:
            customer_positions.append(0.0)
        
        # Truncate if too many customers (shouldn't happen but just in case)
        customer_positions = customer_positions[:MAX_CUSTOMERS * 2]
        
        # Calculate distances to nearest customer and cake
        customer_distances = []
        for i in range(0, len(customer_positions), 2):
            if i+1 < len(customer_positions) and (customer_positions[i] != 0 or customer_positions[i+1] != 0):
                cust_pos = np.array([customer_positions[i], customer_positions[i+1]])
                dist = np.linalg.norm(karen_pos - cust_pos)
                customer_distances.append(dist)
        
        nearest_customer_dist = min(customer_distances) if customer_distances else 1000.0
        
        # Cake positions (hardcoded based on game setup)
        cake_positions = [
            np.array([(SCREEN_SIZE.x // 2) + 200, (SCREEN_SIZE.y // 2) - 50 - 15]),
            np.array([(SCREEN_SIZE.x // 2) + 275, (SCREEN_SIZE.y // 2) - 100 - 15]),
            np.array([SCREEN_SIZE.x - 385, SCREEN_SIZE.y - 408])
        ]
        
        cake_distances = [np.linalg.norm(karen_pos - cake_pos) for cake_pos in cake_positions]
        nearest_cake_dist = min(cake_distances) if cake_distances else 1000.0
        
        # Add strategic features
        strategic_features = [
            karen_score / 100.0,  # Normalized score
            nearest_customer_dist / 500.0,  # Normalized distance to nearest customer
            nearest_cake_dist / 500.0,  # Normalized distance to nearest cake
            salt_timer / 25.0,  # Normalized salt ability timer
            yelp_timer / 30.0,  # Normalized yelp ability timer
            ability_active
        ]
        
        # Normalize positions
        normalized_karen_pos = karen_pos / np.array([SCREEN_SIZE.x, SCREEN_SIZE.y])
        normalized_customer_positions = np.array(customer_positions).reshape(-1) / np.array([SCREEN_SIZE.x, SCREEN_SIZE.y, SCREEN_SIZE.x, SCREEN_SIZE.y, SCREEN_SIZE.x, SCREEN_SIZE.y, SCREEN_SIZE.x, SCREEN_SIZE.y, SCREEN_SIZE.x, SCREEN_SIZE.y])
        
        # Combine all features into a flat array
        state = np.concatenate([
            normalized_karen_pos,
            np.array(strategic_features),
            normalized_customer_positions
        ])
        
        return state
    except Exception as e:
        print(f"Error parsing state: {e}")
        # Return a default state with the correct dimensions
        return np.zeros(2 + 6 + MAX_CUSTOMERS * 2)  # karen_pos(2) + strategic_features(6) + customer_positions(MAX_CUSTOMERS*2)

def connect_to_server(s, retries=5, base_port=12345):
    """Attempt to connect to server with retries and port increment"""
    for i in range(retries):
        try:
            port = base_port + i
            print(f"Attempting to connect on port {port}...")
            
            # Reset socket for each attempt to avoid "Invalid argument" errors
            s.close()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Add a short delay before connecting
            time.sleep(1)
            
            s.connect(('localhost', port))
            print(f"Connected on port {port}")
            
            # Set a timeout to avoid hanging indefinitely but make it longer
            s.settimeout(30.0)  # Increased from 10.0 to 30.0
            return True, s
        except ConnectionRefusedError:
            if i < retries - 1:
                print(f"Connection failed on port {port}, trying next port...")
                time.sleep(1)
            else:
                print("Failed to connect after all retries")
                return False, s
        except socket.error as e:
            print(f"Socket error on port {port}: {e}")
            if i < retries - 1:
                print(f"Retrying connection in 1 second...")
                time.sleep(1)
            else:
                print("Failed to connect after all retries")
                return False, s
    return False, s

def send_command(command, s):
    """Send command with error handling"""
    try:
        s.sendall(command.encode())
        return True
    except socket.error as e:
        print(f"Error sending command: {e}")
        return False

def find_latest_checkpoint():
    """Find the latest model checkpoint in the checkpoints directory.
    
    Returns:
        str or None: Path to the latest checkpoint file, or None if no checkpoints found
    """
    # Check if checkpoints directory exists
    if not os.path.exists('checkpoints'):
        print("No checkpoints directory found.")
        return None
    
    # Get all timestamped directories in the checkpoints folder
    checkpoint_dirs = []
    try:
        # List all items in the checkpoints directory
        for item in os.listdir('checkpoints'):
            full_path = os.path.join('checkpoints', item)
            if os.path.isdir(full_path):
                checkpoint_dirs.append(full_path)
    except Exception as e:
        print(f"Error listing checkpoint directories: {e}")
    
    if not checkpoint_dirs:
        print("No timestamped checkpoint directories found.")
        return None
    
    # Sort directories by modification time (most recent first)
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Get the most recent directory
    latest_dir = checkpoint_dirs[0]
    
    try:
        all_files = []
        for root, dirs, files in os.walk(latest_dir):
            for file in files:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
                print(f"  - {full_path}")
        
        if not all_files:
            print(f"  No files found in {latest_dir}")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Look for model
    model_extensions = ['.h5', '.keras', '.model', '.weights', '.tf', '.ckpt', '.pt', '.pth', '.save']
    model_files = []
    
    # Search in all checkpoint directories
    for dir_path in checkpoint_dirs:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    full_path = os.path.join(root, file)
                    model_files.append(full_path)
    
    if not model_files:
        # Last resort: try to find any file that might be a model
        print("No model files with standard extensions found. Looking for any potential model files...")
        for dir_path in checkpoint_dirs:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    # Check if file is large enough to be a model (> 1MB)
                    full_path = os.path.join(root, file)
                    try:
                        if os.path.getsize(full_path) > 1000000:  # > 1MB
                            model_files.append(full_path)
                            print(f"Found potential model file (large file): {full_path}")
                    except Exception:
                        pass
    
    if not model_files:
        print("No model files found in any checkpoint directory.")
        return None
    
    # Find the best model file (prefer 'best_model' if it exists)
    best_model = None
    for file_path in model_files:
        if 'best_model' in os.path.basename(file_path).lower():
            best_model = file_path
            break
    
    # If no best_model, use the most recent model file
    if not best_model:
        best_model = max(model_files, key=os.path.getmtime)
    
    print(f"Selected checkpoint: {best_model} (Last modified: {datetime.datetime.fromtimestamp(os.path.getmtime(best_model))})")
    return best_model

def generate_random_baker_action():
    """Generate a random action for the baker"""
    return random.choice(COMMANDSBAKER)

def main():
    state_size = 2 + 6 + MAX_CUSTOMERS * 2  # karen_pos(2) + strategic_features(6) + customer_positions(MAX_CUSTOMERS*2)
    
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        print("Creating checkpoints directory...")
        os.makedirs('checkpoints')
    
    # Ask user if they want to train from beginning or from a checkpoint
    train_from_checkpoint = False
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        while True:
            user_choice = input("Do you want to train from the beginning or continue from the latest checkpoint? (begin/checkpoint): ").strip().lower()
            if user_choice in ['begin', 'b', 'beginning', 'start']:
                train_from_checkpoint = False
                break
            elif user_choice in ['checkpoint', 'c', 'continue', 'latest']:
                train_from_checkpoint = True
                break
            else:
                print("Invalid choice. Please enter 'begin' or 'checkpoint'.")
    else:
        print("No checkpoints found. Starting training from the beginning.")
    
    # Ask user to choose baker control mode
    baker_control_mode = ""
    while baker_control_mode not in ["1", "2"]:
        baker_control_mode = input("Choose baker control mode:\n1. Human-controlled\n2. Random actions\nEnter 1 or 2: ").strip()
    
    baker_mode = "human" if baker_control_mode == "1" else "random"
    print(f"Selected baker mode: {baker_mode}")
    
    # Initialize agent
    agent = EnhancedDQNAgent(state_size)
    
    # Load from checkpoint if requested and available
    if train_from_checkpoint and latest_checkpoint:
        print(f"Loading model from {latest_checkpoint}")
        try:
            agent.model.load_weights(latest_checkpoint)
            agent.target_model.load_weights(latest_checkpoint)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from the beginning instead.")
    
    episodes = 5
    
    # Create a new timestamped directory for this training run
    timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir = Path('checkpoints') / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created checkpoint directory for this run: {save_dir}")
    
    logger = metrics.MetricLogger(save_dir)
    
    best_score = float('-inf')
    
    # Kill any existing Python processes that might be using the ports
    try:
        if os.name == 'posix':  # Unix/Linux/MacOS
            os.system("pkill -f 'python.*main.py'")
        elif os.name == 'nt':  # Windows
            os.system("taskkill /f /im python.exe /fi \"WINDOWTITLE eq *main.py*\"")
        time.sleep(2)  # Wait for processes to be killed
    except Exception as e:
        print(f"Error killing existing processes: {e}")
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        # Start the game process with a unique port
        current_dir = os.path.dirname(os.path.realpath(__file__))
        main_py_path = os.path.join(current_dir, "sourceCode/main.py")
        port = 12345 + episode % 10  # Use different port for each episode, but cycle through 10 ports
        
        try:
            if os.name == 'posix':  # Unix/Linux/MacOS
                proc = subprocess.Popen(['python3', main_py_path, str(port)])
            else:  # Windows
                proc = subprocess.Popen(['python', main_py_path, str(port)])
            
            proc_pid = proc.pid
            print(f"Started game process with PID {proc_pid} on port {port}")
            
            # Wait for game to initialize
            time.sleep(3)
            
            # Create a new socket for each episode
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection_success, s = connect_to_server(s, retries=5, base_port=port)
            
            if not connection_success:
                print(f"Failed to connect for episode {episode + 1}, skipping...")
                try:
                    proc.terminate()
                    time.sleep(1)
                    if proc.poll() is None:  # If process hasn't terminated
                        proc.kill()
                except Exception as e:
                    print(f"Error terminating process: {e}")
                continue
            
            # Use the socket in a with statement to ensure proper cleanup
            try:
                total_reward = 0
                steps = 0
                episode_score = 0
                
                while steps < MAX_STEPS:
                    # Baker's turn
                    if baker_mode == "human":
                        print("Baker's turn (human-controlled).")
                        baker_action = input("Enter Baker command (W, S, A, D, Z, X, V, C): ").strip().upper()
                        if not baker_action:
                            print("No command entered, skipping Baker's turn.")
                    else:  # Random baker
                        baker_action = generate_random_baker_action()
                        print(f"Baker's turn (random action): {baker_action}")
                    
                    if baker_action:
                        if not send_command(baker_action, s):
                            print("Failed to send Baker's command, breaking episode loop.")
                            break
                    
                    # Get state after baker's action
                    try:
                        s.settimeout(5.0)  # Set timeout for receiving data
                        data = s.recv(81920)
                        if not data:
                            print("No data received from server, breaking episode loop")
                            break
                        game_data = data.decode().split(",")
                        current_state = parse_state(game_data)
                        old_location = game_data[0]
                        old_score = int(float(game_data[1]))
                    except socket.timeout:
                        print("Socket timeout while receiving data, breaking episode loop")
                        break
                    except Exception as e:
                        print(f"Error receiving game state: {e}")
                        break
                    
                    # Karen's turn (agent actions)
                    available_actions = list(range(len(COMMANDSKAREN)))
                    action_idx, q_value = agent.act(current_state, available_actions)
                    karen_action = COMMANDSKAREN[action_idx]
                    
                    # Execute Karen's action
                    if not send_command(karen_action, s):
                        print("Failed to send Karen's command, breaking episode loop")
                        break
                    
                    # Get new state after Karen's action
                    try:
                        s.settimeout(5.0)  # Set timeout for receiving data
                        data = s.recv(81920)
                        if not data:
                            print("No data received from server, breaking episode loop")
                            break
                        new_game_data = data.decode().split(",")
                        new_state = parse_state(new_game_data)
                        new_score = int(float(new_game_data[1]))
                        score_delta = new_score - old_score
                    except socket.timeout:
                        print("Socket timeout while receiving data, breaking episode loop")
                        break
                    except Exception as e:
                        print(f"Error receiving game state: {e}")
                        break
                    
                    # Calculate reward
                    reward = agent.get_reward(
                        new_game_data[0],  # Karen location
                        old_location,
                        new_game_data[5:],  # Customer locations
                        karen_action,
                        score_delta,
                        bool(new_game_data[4] == 'True'),  # Ability status
                        -score_delta if score_delta < 0 else 0  # Baker score change
                    )
                    
                    # Check if episode is done
                    done = steps >= MAX_STEPS - 1 or new_score >= 100
                    
                    # Store experience in memory
                    agent.memory.add(current_state, action_idx, reward, new_state, done)
                    
                    # Train the agent
                    if len(agent.memory) >= BATCH_SIZE:
                        agent.train()
                    
                    total_reward += reward
                    episode_score = max(episode_score, new_score)
                    steps += 1
                    
                    # Log step metrics
                    logger.log_step(reward, new_score, q_value)
                    
                    if steps % 10 == 0:
                        print(f"Step {steps}/{MAX_STEPS} | Score: {new_score} | Epsilon: {agent.epsilon:.3f}")
                    
                    if done:
                        break
                    
                    time.sleep(0.1)  # Control action rate
            except Exception as e:
                print(f"Error during episode: {e}")
            finally:
                # Close the socket
                try:
                    s.close()
                except:
                    pass
            
            # End of episode
            logger.log_episode()
            
            # Save best model
            if episode_score > best_score:
                best_score = episode_score
                model_path = save_dir / 'best_model.h5'
                agent.model.save(model_path)
                print(f"New best model saved with score {episode_score} at {model_path}")
            
            # Save checkpoint every episode
            checkpoint_path = save_dir / f'checkpoint_episode_{episode+1}.h5'
            agent.model.save(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
            # Log episode metrics
            print(f"Episode {episode + 1} finished:")
            print(f"Total steps: {steps}")
            print(f"Final score: {episode_score}")
            print(f"Total reward: {total_reward}")
            print(f"Epsilon: {agent.epsilon}")
            
            logger.record(
                episode=episode,
                epsilon=agent.epsilon,
                step=steps
            )
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
        finally:
            # Clean up game process
            try:
                if 'proc' in locals() and proc.poll() is None:  # If process is still running
                    if os.name == 'posix':  # Unix/Linux/MacOS
                        os.kill(proc_pid, 9)
                    else:  # Windows
                        proc.kill()
                    print(f"Terminated game process with PID {proc_pid}")
            except Exception as e:
                print(f"Error terminating process: {e}")
            
            time.sleep(1)  # Wait before starting next episode
    
    # Save final model
    final_model_path = save_dir / 'final_model.h5'
    agent.model.save(final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    print(f"Training completed. Models saved in {save_dir}")
    print(f"Best score achieved: {best_score}")

class KarenAgent(EnhancedDQNAgent):
    def __init__(self, state_size):
        super().__init__(state_size, character="karen")
        self.name = "karen"
        self.last_score = 0
        self.consecutive_no_reward_steps = 0
        self.exploration_bonus = 0.0
        self.visited_positions = set()
        
    def get_reward(self, karen_location, old_location, customers, karen_action, score_delta, ability_active, baker_score):
        """Enhanced reward function for KarenCat with strategic incentives"""
        reward = 0
        
        # Parse locations
        karen_x, karen_y = map(float, karen_location.split('_'))
        old_x, old_y = map(float, old_location.split('_'))
        karen_pos = np.array([karen_x, karen_y])
        old_pos = np.array([old_x, old_y])
        
        # Track visited positions (discretized to 10x10 grid cells)
        pos_key = (int(karen_x / 10), int(karen_y / 10))
        is_new_position = pos_key not in self.visited_positions
        if is_new_position:
            self.visited_positions.add(pos_key)
            self.exploration_bonus += 0.5  # Bonus for exploring new areas
            if len(self.visited_positions) > 100:  # Limit memory
                self.visited_positions.clear()
                
        # Customer interaction rewards
        customer_locations = []
        for c in customers:
            if c and '_' in str(c):
                customer_locations.append(c)
                
        if customer_locations:
            closest_customer_dist = float('inf')
            for customer in customer_locations:
                try:
                    cust_x, cust_y = map(float, customer.split('_'))
                    dist = np.linalg.norm(karen_pos - np.array([cust_x, cust_y]))
                    closest_customer_dist = min(closest_customer_dist, dist)
                except (ValueError, TypeError):
                    continue
            
            # Reward for moving towards customers
            if closest_customer_dist < self.previous_closest_customer_dist:
                reward += 2 * (self.previous_closest_customer_dist - closest_customer_dist) / 100
            
            # Extra reward for being very close to customers
            if closest_customer_dist < CUSTOMER_INTERACT_DISTANCE:
                reward += 5
                if karen_action == "L":  # Ready to annoy
                    reward += 10
            
            self.previous_closest_customer_dist = closest_customer_dist
        
        # Cake interaction rewards
        cake_positions = [
            np.array([(SCREEN_SIZE.x // 2) + 200, (SCREEN_SIZE.y // 2) - 50 - 15]),
            np.array([(SCREEN_SIZE.x // 2) + 275, (SCREEN_SIZE.y // 2) - 100 - 15]),
            np.array([SCREEN_SIZE.x - 385, SCREEN_SIZE.y - 408])
        ]
        
        closest_cake_dist = min([np.linalg.norm(karen_pos - cake_pos) for cake_pos in cake_positions])
        
        # Reward for moving towards cakes
        if closest_cake_dist < self.previous_closest_cake_dist:
            reward += 2 * (self.previous_closest_cake_dist - closest_cake_dist) / 100
        
        # Extra reward for being close to cakes
        if closest_cake_dist < CAKE_INTERACT_DISTANCE:
            reward += 5
            if karen_action == "K":  # Ready to salt cake
                reward += 10
        
        self.previous_closest_cake_dist = closest_cake_dist
        
        # Action-specific rewards
        if score_delta > 0:  # Successful action
            reward += score_delta * 20  # Base reward for scoring
            self.consecutive_no_reward_steps = 0
            
            if karen_action == "L":  # Successful annoy
                reward += 50
                self.successful_actions["L"] += 1
            elif karen_action == "K":  # Successful cake salting
                reward += 75
                self.successful_actions["K"] += 1
                # Extra reward for reducing baker's score
                reward += 25
            elif karen_action == "J":  # Successful Yelp review
                reward += 60
                self.successful_actions["J"] += 1
        else:
            # Small penalty for not making progress
            self.consecutive_no_reward_steps += 1
            if self.consecutive_no_reward_steps > 10:
                reward -= 0.5  # Small penalty to encourage finding rewards
        
        # Update action counts
        if karen_action in self.action_counts:
            self.action_counts[karen_action] += 1
        
        # Strategic rewards
        if baker_score < 0:  # Successfully reduced baker's score
            reward += abs(baker_score) * 15
        
        # Exploration rewards
        movement_dist = np.linalg.norm(karen_pos - old_pos)
        if movement_dist > 0:
            reward += 1  # Small reward for movement
            
            # Bonus for exploring new areas
            if is_new_position:
                reward += self.exploration_bonus
                self.exploration_bonus = max(0.5, self.exploration_bonus * 0.95)  # Decay exploration bonus
        
        # Game progress rewards
        if score_delta >= 100:  # Win condition
            reward += 1000
        
        # Store last score for next comparison
        self.last_score = score_delta
        
        return reward

class OpponentModel:
    def __init__(self):
        self.opponent_actions = []
        self.opponent_positions = []
        self.action_frequencies = {}
        self.position_heatmap = np.zeros((SCREEN_SIZE.x // 10, SCREEN_SIZE.y // 10))
        
    def update(self, opponent_action, opponent_position):
        """Update opponent model with new observation"""
        self.opponent_actions.append(opponent_action)
        self.opponent_positions.append(opponent_position)
        
        # Update action frequencies
        if opponent_action in self.action_frequencies:
            self.action_frequencies[opponent_action] += 1
        else:
            self.action_frequencies[opponent_action] = 1
            
        # Update position heatmap
        x, y = map(float, opponent_position.split('_'))
        x_idx = min(int(x) // 10, self.position_heatmap.shape[0] - 1)
        y_idx = min(int(y) // 10, self.position_heatmap.shape[1] - 1)
        self.position_heatmap[x_idx, y_idx] += 1
        
    def predict_next_action(self):
        """Predict opponent's next action based on history"""
        if not self.opponent_actions:
            return None
            
        # Simple frequency-based prediction
        most_common_action = max(self.action_frequencies.items(), key=lambda x: x[1])[0]
        return most_common_action
        
    def get_opponent_hotspots(self):
        """Get areas where opponent spends most time"""
        if np.sum(self.position_heatmap) == 0:
            return []
            
        # Find top 3 hotspots
        flat_indices = np.argsort(self.position_heatmap.flatten())[-3:]
        hotspots = []
        
        for idx in flat_indices:
            x_idx, y_idx = np.unravel_index(idx, self.position_heatmap.shape)
            hotspots.append((x_idx * 10, y_idx * 10))
            
        return hotspots

def get_strategic_opponent_action(agent, opponent_model):
    """Generate a strategic action for the opponent based on agent's state and opponent model"""
    # Get opponent hotspots
    hotspots = opponent_model.get_opponent_hotspots()
    
    # If we have hotspots, try to counter them
    if hotspots and agent.character == "karen":
        # Baker countering Karen
        # If Karen is near customers, use Pusheen Wall (V)
        if agent.previous_closest_customer_dist < CUSTOMER_INTERACT_DISTANCE * 2:
            return "V"
        # If Karen is near cakes, use free samples (X) to distract
        elif agent.previous_closest_cake_dist < CAKE_INTERACT_DISTANCE * 2:
            return "X"
        else:
            # Default to serving customers
            return "Z"
    elif hotspots and agent.character == "baker":
        # Karen countering Baker
        # If Baker is near customers, use salt (K) to ruin cakes
        if agent.previous_closest_customer_dist < CUSTOMER_INTERACT_DISTANCE * 2:
            return "K"
        # Otherwise annoy customers
        else:
            return "L"
    else:
        # If no hotspots or model data, use random movement
        if agent.character == "karen":
            return random.choice(["W", "S", "A", "D"])
        else:
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

def train_with_curriculum(agent, episodes, difficulty_levels=3, baker_mode="human"):
    """Train agent with curriculum learning - gradually increasing difficulty"""
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = metrics.MetricLogger(save_dir)
    
    best_score = float('-inf')
    
    # Define curriculum stages
    for difficulty in range(1, difficulty_levels + 1):
        print(f"\n=== Starting Curriculum Stage {difficulty}/{difficulty_levels} ===")
        
        episodes_per_stage = episodes // difficulty_levels
        for episode in range(episodes_per_stage):
            print(f"\nEpisode {episode + 1}/{episodes_per_stage} (Difficulty {difficulty})")
            
            # Start game process
            current_dir = os.path.dirname(os.path.realpath(__file__))
            main_py_path = os.path.join(current_dir, "sourceCode/main.py")
            port = 12345 + (episode % 10)  # Use different port for each episode, but cycle through 10 ports
            
            try:
                if os.name == 'posix':  # Unix/Linux/MacOS
                    proc = subprocess.Popen(['python3', main_py_path, str(port)])
                else:  # Windows
                    proc = subprocess.Popen(['python', main_py_path, str(port)])
                
                proc_pid = proc.pid
                print(f"Started game process with PID {proc_pid} on port {port}")
                
                # Wait for game to initialize
                time.sleep(3)
                
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    connection_success, s = connect_to_server(s, retries=5, base_port=port)
                    if not connection_success:
                        print(f"Failed to connect for episode {episode + 1}, skipping...")
                        try:
                            proc.terminate()
                            time.sleep(1)
                            if proc.poll() is None:  # If process hasn't terminated
                                proc.kill()
                        except Exception as e:
                            print(f"Error terminating process: {e}")
                        continue
                    
                    total_reward = 0
                    steps = 0
                    episode_score = 0
                    opponent_model = OpponentModel()
                    
                    # Adjust max steps based on difficulty
                    adjusted_max_steps = MAX_STEPS * difficulty
                    
                    try:
                        while steps < adjusted_max_steps:
                            # Baker's turn
                            if baker_mode == "human":
                                print("Baker's turn (human-controlled).")
                                baker_action = input("Enter Baker command (W, S, A, D, Z, X, V, C): ").strip().upper()
                                if not baker_action:
                                    print("No command entered, skipping Baker's turn.")
                            else:  # Random baker
                                baker_action = generate_random_baker_action()
                                print(f"Baker's turn (random action): {baker_action}")
                            
                            if baker_action:
                                if not send_command(baker_action, s):
                                    print("Failed to send Baker's command, breaking episode loop.")
                                    break
                            
                            # Get state after Baker's action
                            try:
                                s.settimeout(5.0)  # Set timeout for receiving data
                                data = s.recv(81920)
                                if not data:
                                    print("No data received from server, breaking episode loop")
                                    break
                                game_data = data.decode().split(",")
                                current_state = parse_state(game_data)
                                old_location = game_data[0]
                                old_score = int(float(game_data[1]))
                            except socket.timeout:
                                print("Socket timeout while receiving data, breaking episode loop")
                                break
                            except Exception as e:
                                print(f"Error receiving game state: {e}")
                                break
                            
                            # Track Baker's move
                            opponent_position = game_data[0]
                            opponent_model.update(baker_action, opponent_position)
                            
                            # Karen's turn (agent actions)
                            available_actions = list(range(len(COMMANDSKAREN)))
                            action_idx, q_value = agent.act(current_state, available_actions)
                            karen_action = COMMANDSKAREN[action_idx]
                            
                            # Execute Karen's action
                            if not send_command(karen_action, s):
                                print("Failed to send Karen's command, breaking episode loop")
                                break
                            
                            # Get new state after Karen's action
                            try:
                                s.settimeout(5.0)  # Set timeout for receiving data
                                data = s.recv(81920)
                                if not data:
                                    print("No data received from server, breaking episode loop")
                                    break
                                new_game_data = data.decode().split(",")
                                new_state = parse_state(new_game_data)
                                new_score = int(float(new_game_data[1]))
                                score_delta = new_score - old_score
                            except socket.timeout:
                                print("Socket timeout while receiving data, breaking episode loop")
                                break
                            except Exception as e:
                                print(f"Error receiving game state: {e}")
                                break
                            
                            # Calculate reward
                            reward = agent.get_reward(
                                new_game_data[0],  # Karen location
                                old_location,
                                new_game_data[5:],  # Customer locations
                                karen_action,
                                score_delta,
                                bool(new_game_data[4] == 'True'),  # Ability status
                                -score_delta if score_delta < 0 else 0  # Baker score change
                            )
                            
                            # Check if episode is done
                            done = steps >= adjusted_max_steps - 1 or new_score >= 100
                            
                            # Store experience in memory
                            agent.memory.add(current_state, action_idx, reward, new_state, done)
                            
                            # Train the agent
                            if len(agent.memory) >= BATCH_SIZE:
                                agent.train()
                            
                            total_reward += reward
                            episode_score = max(episode_score, new_score)
                            steps += 1
                            
                            # Log step metrics
                            logger.log_step(reward, new_score, q_value)
                            
                            if steps % 10 == 0:
                                print(f"Step {steps}/{adjusted_max_steps} | Score: {new_score} | Epsilon: {agent.epsilon:.3f}")
                            
                            if done:
                                break
                            
                            time.sleep(0.1)  # Control action rate
                    except Exception as e:
                        print(f"Error during episode: {e}")
                    
                    # End of episode
                    logger.log_episode()
                    
                    # Save best model
                    if episode_score > best_score:
                        best_score = episode_score
                        agent.model.save(save_dir / f'best_model_karen_diff{difficulty}.h5')
                        print(f"New best model saved with score {episode_score}")
                    
                    # Save checkpoint every 5 episodes
                    if episode % 5 == 0:
                        agent.model.save(save_dir / f'checkpoint_karen_diff{difficulty}_ep{episode}.h5')
                    
                    # Log episode metrics
                    print(f"Episode {episode + 1} finished:")
                    print(f"Total steps: {steps}")
                    print(f"Final score: {episode_score}")
                    print(f"Total reward: {total_reward}")
                    print(f"Epsilon: {agent.epsilon}")
                    
                    logger.record(
                        episode=episode,
                        epsilon=agent.epsilon,
                        step=steps
                    )
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
            finally:
                # Clean up game process
                try:
                    if 'proc' in locals() and proc.poll() is None:  # If process is still running
                        if os.name == 'posix':  # Unix/Linux/MacOS
                            os.kill(proc_pid, 9)
                        else:  # Windows
                            proc.kill()
                        print(f"Terminated game process with PID {proc_pid}")
                except Exception as e:
                    print(f"Error terminating process: {e}")
                
                time.sleep(1)  # Wait before starting next episode
        
        # Save final model for this difficulty level
        agent.model.save(save_dir / f'final_model_karen_diff{difficulty}.h5')
        print(f"Completed difficulty level {difficulty} training")
    
    # Save final model after all curriculum stages
    agent.model.save(save_dir / f'final_model_karen.h5')
    print(f"Curriculum training completed. Final model saved.")
    
    return agent

if __name__ == "__main__":
    main()
