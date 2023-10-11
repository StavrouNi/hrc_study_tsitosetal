import numpy as np
import matplotlib.pyplot as plt


# Load data for multiple participants # for the first 3 plots WINS REWARDS TRAVELLED DISTANCE we can have more than 1 filepath 
filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_ChristosPilot_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_NICKEXPERT_LfD_TL_1/data/test_data.csv'
    # Add more file paths as needed
]

steps_filepaths = [    
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_ChristosPilot_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_NICKEXPERT_LfD_TL_1/data/rl_test_data.csv'
]

# Sum the timesteps and check if they match with rows of rl_data matrix
for test_filepath, rl_filepath in zip(filepaths, steps_filepaths):
    test_data = np.loadtxt(test_filepath, delimiter=',', skiprows=1)
    rl_data = np.loadtxt(rl_filepath, delimiter=',', skiprows=1)

####################################WINSS#########################################

all_participant_wins = []
all_participant_rewards = []    

# Calculate win counts for each participant
for filepath in filepaths:
    test_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    win_counts = []
    for i in range(0, len(test_data), 10):
        block = test_data[i:i+10, 0]
        win_count = np.sum(block != -150) #-150 is a loss
        win_counts.append(win_count)
    all_participant_wins.append(win_counts)
all_participant_wins = np.array(all_participant_wins)

# Compute mean and variance across participants for each block
mean_wins = np.mean(all_participant_wins, axis=0)
variance_wins = np.var(all_participant_wins, axis=0)
std_deviation = np.sqrt(variance_wins)

# Plotting
block_numbers = np.arange(1, len(mean_wins) + 1)
plt.plot(block_numbers, mean_wins, label='Mean Wins', color='blue')
plt.fill_between(block_numbers, mean_wins - std_deviation, mean_wins + std_deviation, color='blue', alpha=0.2, label='1 Std. Dev.')
plt.xlabel('Block Number')
plt.ylabel('Number of Wins')
plt.title('Mean and Variance of Wins per Block across Participants')
plt.legend()
plt.show()

####################################REWARDS#########################################


for filepath in filepaths:
    test_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    rewards_per_block = []
    for i in range(0, len(test_data), 10):
        block = test_data[i:i+10, 0]
        block_rewards = 150 + block  # Subtract 10 from each game's value
        #print(block_rewards)
        block_reward = np.mean(block_rewards)  # Sum up the rewards in the block
        rewards_per_block.append(block_reward)
    all_participant_rewards.append(rewards_per_block)
all_participant_rewards = np.array(all_participant_rewards)

# Compute mean and variance across participants for each block
mean_rewards = np.mean(all_participant_rewards, axis=0)
variance_rewards = np.var(all_participant_rewards, axis=0)
std_deviation = np.sqrt(variance_rewards)

# Plotting
block_numbers = np.arange(1, len(mean_rewards) + 1)
plt.plot(block_numbers, mean_rewards, label='Mean Rewards', color='green')
plt.fill_between(block_numbers, mean_rewards - std_deviation, mean_rewards + std_deviation, color='green', alpha=0.2, label='1 Std. Dev.')
plt.xlabel('Block Number')
plt.ylabel('Reward')
#plt.ylim(0, 140)  # Set y-axis limits to [0, 140]
plt.title('Mean and Variance of Rewards per Block across Participants')
plt.legend()
plt.show()


####################################NORMALIZED DISTANCE#########################################
# First, find out the maximum duration across all data
max_duration = 0
for filepath in filepaths:
    test_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    current_max = np.max(test_data[:, 1])  # Assuming the 2nd column has durations
    max_duration = max(max_duration, current_max)

all_participant_normalized_distances = []

# Calculate normalized travelled distance for each participant
for filepath in filepaths:
    test_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    normalized_distances_per_block = []
    for i in range(0, len(test_data), 10):
        block_durations = test_data[i:i+10, 1]
        block_distances = test_data[i:i+10, 2]
        
        # Normalize the durations with the maximum duration
        normalized_durations = block_durations / max_duration
        
        # Compute normalized travelled distances for each game in the block
        normalized_distances = block_distances * normalized_durations
        
        # Take the mean of normalized distances for this block
        mean_normalized_distance = np.mean(normalized_distances)
        normalized_distances_per_block.append(mean_normalized_distance)
    
    all_participant_normalized_distances.append(normalized_distances_per_block)

all_participant_normalized_distances = np.array(all_participant_normalized_distances)

# Compute mean and variance across participants for each block
mean_distances = np.mean(all_participant_normalized_distances, axis=0)
variance_distances = np.var(all_participant_normalized_distances, axis=0)
std_deviation_distances = np.sqrt(variance_distances)

# Plotting
plt.plot(block_numbers, mean_distances, label='Mean Normalized Distance', color='purple')
plt.fill_between(block_numbers, mean_distances - std_deviation_distances, mean_distances + std_deviation_distances, color='purple', alpha=0.2, label='1 Std. Dev.')
plt.xlabel('Block Number')
plt.ylabel('Normalized Travelled Distance')
plt.title('Mean and Variance of Normalized Travelled Distance per Block across Participants')
plt.legend()
plt.show()

##############################################trajectory##################################

def get_game_data(game_number, test_data, rl_data):
    # Ensure the game number is valid
    if game_number < 0 or game_number >= len(test_data):
        raise ValueError("Invalid game number.")
    
    # Find the starting index for the desired game in rl_data
    if game_number == 0:
        start_index = 0
    else:
        start_index = int(np.sum(test_data[:game_number, -1])) + game_number
    
    # Determine how many rows to extract
    num_rows = int(test_data[game_number, -1])
    
    # Extract and return the relevant rows from rl_data
    game_data = rl_data[start_index:start_index+num_rows, :]

    return game_data


########################################DECLARE GAME FOR TRAJECTORY######################
game2_data = get_game_data(1, test_data, rl_data)
# Extracting x and y coordinates
x_coordinates = game2_data[:, 2]
y_coordinates = game2_data[:, 3]

# Plotting

plt.figure(figsize=(10, 10))
plt.plot(x_coordinates, y_coordinates, 'o-', label="Trajectory")
plt.scatter(x_coordinates[0], y_coordinates[0], color='red', s=100, label="Start Point", zorder=5)  # Start point with a distinct color and size
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Trajectory for Game 2")
plt.legend()
plt.grid(True)
plt.show()

##########################################HEATMAP##########################
def get_batch_data(batch_number, test_data, rl_data, games_per_batch=10):
    # Calculate the starting and ending game numbers for the batch
    start_game = batch_number * games_per_batch
    end_game = start_game + games_per_batch

    x_coords = []
    y_coords = []

    # Fetch the data for each game in the batch
    for game_num in range(start_game, end_game):
        game_data = get_game_data(game_num, test_data, rl_data)
        print(f"Game number {game_num} has {len(game_data)} rows.")
        x_coords.extend(game_data[:, 2])
        y_coords.extend(game_data[:, 3])
    
    print(f"Total games fetched: {end_game - start_game}")
    print(f"Total rows fetched: {len(x_coords)}")

    return x_coords, y_coords


def plot_heatmap(x_coords, y_coords):
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Position Heatmap for Block")
    plt.colorbar(label="Frequency")
    plt.show()

# ################################DECLARE THE BATCH FOR THE HEATMAP##########################
x_coords, y_coords = get_batch_data(1, test_data, rl_data)
plot_heatmap(x_coords, y_coords)
'''

################################################HUMAN ACTIONS##########################
def get_game_human_actions(game_number, test_data, rl_data):
    """Get human actions for a specific game."""
    game_data = get_game_data(game_number, test_data, rl_data)
    return game_data[:, 0]  # Assuming first column has human actions

def plot_human_actions_histogram(game_number, test_data, rl_data):
    """Plot histogram for human actions of a specific game."""
    actions = get_game_human_actions(game_number, test_data, rl_data)
    plt.figure(figsize=(10, 6))
    plt.hist(actions, bins=np.arange(actions.min(), actions.max() + 2) - 0.5, edgecolor="k")
    plt.xlabel("Human Actions")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Human Actions for Game {game_number}")
    plt.xticks(np.unique(actions))
    plt.grid(True)
    plt.show()

# Plot human actions for a specific game
plot_human_actions_histogram(59, test_data, rl_data)  # For game 59
'''