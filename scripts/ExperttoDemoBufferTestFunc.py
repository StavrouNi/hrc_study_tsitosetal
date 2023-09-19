
import numpy as np
import pickle

def ExpertoDemo(filename, save_path):
    data = np.load(filename, allow_pickle=True, encoding='latin1')
    np.set_printoptions(threshold=np.inf)

    last_column = data[:, -1]
    true_count = np.sum(last_column)

    indices = np.where(last_column)[0]
    splits = np.array_split(indices, len(indices) // 10)
    episodes = []

    num_extracted_games_per_episode = [0, 2, 0, 4, 4] # define the number of winning game per block

    for split in splits:
        start_idx = split[0] - 1 if split[0] != 0 else 0
        end_idx = split[-1] + 1
        split_data = data[start_idx:end_idx]
        episodes.append(split_data)

    extracted_games = []

    #for episode_idx, episode in enumerate(episodes):
    #    column_3_values = episode[:, 2]
    #    winning_games_indices = np.where((column_3_values == 10) & (episode[:, -1] == True))[0]
    #    num_extracted_games = min(num_extracted_games_per_episode[episode_idx], len(winning_games_indices))
    #    for game_idx in winning_games_indices[:num_extracted_games]:
    #        game_start_idx = game_idx - 1
    #        while game_start_idx >= 0 and not episode[game_start_idx, -1]:
    #            game_start_idx -= 1
    #        game_start_idx = max(0, game_start_idx)  # Reset to 0 if it becomes negative
    #        extracted_games.append(episode[game_start_idx+1:game_idx + 1])

    #    print("Number of extracted winning games episode:", episode_idx, num_extracted_games)

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

        # Sort winning games by length in descending order and extract the largest ones
        winning_games = sorted(winning_games, key=len, reverse=True)
        num_extracted_games = min(num_extracted_games_per_episode[episode_idx], len(winning_games))
        extracted_games.extend(winning_games[:num_extracted_games])

    for game in extracted_games:
        print("Game shape:", game.shape)

    if episodes:
        print("Second episode:", episodes[1])
    if extracted_games:
        print("First game of extracted winning games:", extracted_games[0])

    print("Number of extracted winning games total:", len(extracted_games))

    # Save the demogames to an array and to a specific path
    demo_games=np.vstack(extracted_games)
    #np.save(save_path, demo_games, fix_imports=True)
    with open(save_path, 'wb') as f:
        pickle.dump(demo_games, f, protocol=2)
    print("shape", demo_games.shape)
    print(type(demo_games))

    return demo_games
save_path='/home/ttsitos/catkin_ws/src/hrc_study_tsitosetal/buffers/demo_buffer/demo_data.npy'

filename = '/home/ttsitos/catkin_ws/src/hrc_study_tsitosetal/buffers/expert_buffer/buffer_data_1.npy'
ExpertoDemo(filename, save_path)
