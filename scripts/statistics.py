import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

# Load data for multiple participants # for the first 3 plots WINS REWARDS TRAVELLED DISTANCE we can have more than 1 filepath 
TL_test_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/test_data.csv'
]
TL_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/data.csv'
]
No_TL_test_filepaths = [

    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/test_data.csv'
]
No_TL_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/data.csv'
]

TL_steps_test_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/rl_test_data.csv'
]
TL_steps_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/rl_data.csv'
]
No_TL_steps_test_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/rl_test_data.csv'
]
No_TL_steps_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/rl_data.csv'
]
TL_test_data=[]
TL_train_data=[]
NO_TL_test_data=[]
NO_TL_train_data=[]
TL_steps_test_data=[]
TL_steps_train_data=[]
NO_TL_steps_test_data=[]
NO_TL_steps_train_data=[]
for TL_test_data_files, TL_train_data_files, NO_TL_test_data_files, NO_TL_train_data_files in zip(TL_test_filepaths,TL_train_filepaths, No_TL_test_filepaths,No_TL_train_filepaths):
    TL_test_data.append(np.loadtxt(TL_test_data_files, delimiter=',', skiprows=1))
    TL_train_data.append(np.loadtxt(TL_train_data_files, delimiter=',', skiprows=1))
    NO_TL_test_data.append(np.loadtxt(NO_TL_test_data_files, delimiter=',', skiprows=1))
    NO_TL_train_data.append(np.loadtxt(NO_TL_train_data_files, delimiter=',', skiprows=1))

for TL_steps_test_data_files, TL_steps_train_data_files, NO_TL_steps_test_data_files, NO_TL_steps_train_data_files in zip(TL_steps_test_filepaths,TL_steps_train_filepaths, No_TL_steps_test_filepaths,No_TL_steps_train_filepaths):
    TL_steps_test_data.append(np.loadtxt(TL_steps_test_data_files, delimiter=',', skiprows=1))
    TL_steps_train_data.append(np.loadtxt(TL_steps_train_data_files, delimiter=',', skiprows=1))
    NO_TL_steps_test_data.append(np.loadtxt(NO_TL_steps_test_data_files, delimiter=',', skiprows=1))
    NO_TL_steps_train_data.append(np.loadtxt(NO_TL_steps_train_data_files, delimiter=',', skiprows=1))

TL_train_data = [participant_data[:, 1:] for participant_data in TL_train_data]
NO_TL_train_data = [participant_data[:, 1:] for participant_data in NO_TL_train_data]
#expert_train_data= [participant_data[:, 1:] for participant_data in expert_train_data]


def calculate_means(data_group):
    means = []
    # Check if data_group is a list of arrays
    for participant_data in data_group:
        if isinstance(participant_data, np.ndarray) and participant_data.ndim == 2:
            # Take the first 10 rows and the column for rewards
            means.append(np.mean(participant_data[:10, 0]+150))  # Change 0 to the column index for rewards if necessary
        else:
            raise ValueError("Each item in data_group must be a 2-dimensional NumPy array")
    return means

TL_test_means = calculate_means(TL_test_data)
NO_TL_test_means = calculate_means(NO_TL_test_data)

def perform_t_test(group1_means, group2_means):
    # Perform the t-test on the means
    t_stat, p_value = stats.ttest_ind(group1_means, group2_means, equal_var=False)  # Assuming equal_var=False for Welch's t-test
    return t_stat, p_value


t_stat, p_value = perform_t_test(TL_test_means, NO_TL_test_means)


def plot_means(group1_means, group2_means):
    plt.figure(figsize=(10, 6))

    plt.hist(group1_means, alpha=0.5, label='TL Group Means')
    plt.hist(group2_means, alpha=0.5, label='NO TL Group Means')

    plt.title('Distribution of Means')
    plt.xlabel('Mean Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    _, p_value_group1 = stats.shapiro(group1_means)
    _, p_value_group2 = stats.shapiro(group2_means)
    
    return p_value_group1, p_value_group2

p1, p2 = plot_means(TL_test_means, NO_TL_test_means)

u_stat, p_value_u_test = stats.mannwhitneyu(TL_test_means, NO_TL_test_means, alternative='two-sided')

print(f"TL_shapiro_normality: {p1}, NO_TL_shapiro_normality: {p2}")
print(f"T-statistic: {t_stat}, P-value: {p_value}")
print(f"Mann-Whitney U : {u_stat}, Mann-Whitney p-Value {p_value_u_test}")



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

# ... (Previous code for loading data and calculating means for rewards)

# Define a function to calculate mean wins
def calculate_mean_wins(data_group):
    means = []
    for participant_data in data_group:
        if isinstance(participant_data, np.ndarray) and participant_data.ndim == 2:
            # Calculate wins by counting non-negative rewards (excluding -150)
            wins = np.sum(participant_data[:10, 0] != -150)  # Change 0 to the column index for rewards if necessary
            means.append(wins)
        else:
            raise ValueError("Each item in data_group must be a 2-dimensional NumPy array")
    return means

# Calculate mean wins for TL and NO TL groups
TL_test_mean_wins = calculate_mean_wins(TL_test_data)
NO_TL_test_mean_wins = calculate_mean_wins(NO_TL_test_data)

# Perform a statistical test (e.g., Mann-Whitney U test) to compare mean wins
u_stat_wins, p_value_u_test_wins = stats.mannwhitneyu(TL_test_mean_wins, NO_TL_test_mean_wins, alternative='two-sided')

# Plot the distribution of mean wins
def plot_mean_wins(group1_means, group2_means):
    plt.figure(figsize=(10, 6))

    plt.hist(group1_means,bins=80, alpha=0.5, label='TL Group Mean Wins')
    plt.hist(group2_means,bins=80, alpha=0.5, label='NO TL Group Mean Wins')

    plt.title('Distribution of Mean Wins')
    plt.xlabel('Mean Wins')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

plot_mean_wins(TL_test_mean_wins, NO_TL_test_mean_wins)

print(f"Mann-Whitney U (Wins) : {u_stat_wins}, Mann-Whitney p-Value (Wins) {p_value_u_test_wins}")
# Define a function to perform a t-test and Shapiro-Wilk normality test
def perform_wins_tests(group1_means, group2_means):
    # Perform the t-test on the means
    t_stat_wins, p_value_wins = stats.ttest_ind(group1_means, group2_means, equal_var=False)  # Assuming equal_var=False for Welch's t-test
    
    # Perform Shapiro-Wilk normality test
    _, p_value_group1_wins = stats.shapiro(group1_means)
    _, p_value_group2_wins = stats.shapiro(group2_means)
    
    return t_stat_wins, p_value_wins, p_value_group1_wins, p_value_group2_wins

# Perform t-test and Shapiro-Wilk test for mean wins
t_stat_wins, p_value_wins, p_value_group1_wins, p_value_group2_wins = perform_wins_tests(TL_test_mean_wins, NO_TL_test_mean_wins)

# Plot the distribution of mean wins
def plot_mean_wins(group1_means, group2_means, num_bins):
    plt.figure(figsize=(10, 6))

    plt.hist(group1_means, num_bins, alpha=0.5, label='TL Group Mean Wins')
    plt.hist(group2_means, num_bins, alpha=0.5, label='NO TL Group Mean Wins')

    plt.title('Distribution of Mean Wins')
    plt.xlabel('Mean Wins')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

plot_mean_wins(TL_test_mean_wins, NO_TL_test_mean_wins, num_bins=80)

# Print t-test and Shapiro-Wilk test results
print(f"T-test (Wins) - T-statistic: {t_stat_wins}, P-value: {p_value_wins}")
print(f"Shapiro-Wilk Test (TL Group Wins) - P-value: {p_value_group1_wins}")
print(f"Shapiro-Wilk Test (NO TL Group Wins) - P-value: {p_value_group2_wins}")