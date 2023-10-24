
import numpy as np
import pickle

def ExpertoDemo(filename, save_path):
    data = np.load(filename, allow_pickle=True, encoding='latin1')
    np.set_printoptions(threshold=np.inf)

    last_column = data[:, -1]
    true_count = np.sum(last_column)
    total_games = np.sum(data[:, -1])
    print("Total number of games:", total_games)

    indices = np.where(last_column)[0]
    splits = np.array_split(indices, len(indices) // 10)
    episodes = []

    num_extracted_games_per_episode = [0, 2, 3, 3, 2] # define the number of winning game per block

    for split in splits:
        start_idx = split[0] - 1 if split[0] != 0 else 0
        end_idx = split[-1] + 1
        split_data = data[start_idx:end_idx]
        episodes.append(split_data)
        print("Total games in episode {}: {}".format(len(episodes), len(split_data)))  # Added this line

    extracted_games = []


    for episode_idx, episode in enumerate(episodes): #with this updated method we collect the largest winning games of the 2,3,3,2 of each episode 
        column_3_values = episode[:, 2]
        winning_games_indices = np.where((column_3_values == 10) & (episode[:, -1] == True))[0]

        # Collect all winning games in the episode
        winning_games = []
        for game_idx in winning_games_indices:
            game_start_idx = game_idx - 1
            while game_start_idx >= 0 and not episode[game_start_idx, -1]:
                game_start_idx -= 1
            game_start_idx = max(0, game_start_idx)  # Reset to 0 if it becomes negative
            winning_games.append(episode[game_start_idx+1:game_idx + 1])  # Exclude the last line (when looking backwards)
        print("Number of WINNING winning games total:", len(winning_games))  # Print the length of demo_games

        # Shuffle the winning games and extract the specified number of games # ##FOR RANDOM
        np.random.shuffle(winning_games)
        num_extracted_games = min(num_extracted_games_per_episode[episode_idx], len(winning_games))
        extracted_games.extend(winning_games[:num_extracted_games])


    print(extracted_games)
    for game in extracted_games:
        print("Game shape:", game.shape)

    #if episodes:
        #print("Second episode:", episodes[1])
    #if extracted_games:
        #print("First game of extracted winning games:", extracted_games[0])
    print("Number of extracted winning games total:", len(extracted_games))

    # Save the demogames to an array and to a specific path
    demo_games=np.vstack(extracted_games)
    #np.save(save_path, demo_games, fix_imports=True)
        # Sort the extracted games by their length in ascending order
    extracted_games.sort(key=lambda x: x.shape[0])

    # Keep the shortest winning games if there are more than needed
    for episode_idx, num_to_keep in enumerate(num_extracted_games_per_episode):
        if num_to_keep < len(extracted_games):
            extracted_games = extracted_games[:num_to_keep]
    with open(save_path, 'wb') as f:
        pickle.dump(demo_games, f, protocol=2)
    print("Shortest games:")
    for i, game in enumerate(extracted_games[:5]):
        print("Game {} shape:".format(i+1), game.shape)
        print(game)
        print("-" * 30)

    print("shape", demo_games.shape)
    #print(type(demo_games))
   # print("Number of extracted winning games total:", len(demo_games))  # Print the length of demo_games
    #print("Total games overall: {}".format(len(data)))
    return demo_games
save_path='/home/ttsitos/catkin_ws/src/hrc_study_tsitosetal/buffers/demo_buffer/DELETE.npy'

filename = '/home/ttsitos/catkin_ws/src/hrc_study_tsitosetal/buffers/expert_buffer/buffer_data_3.npy'
ExpertoDemo(filename, save_path)
