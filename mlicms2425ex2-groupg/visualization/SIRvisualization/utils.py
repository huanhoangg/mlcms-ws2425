import os
import pandas as pd
import plotly.graph_objects as go


def file_df_to_count_df(df,
                        ID_SUSCEPTIBLE=1,
                        ID_INFECTED=0,
                        ID_REMOVED=2):  # Added ID_REMOVED for recovered state tracking
    """
    Converts the file DataFrame to a group count DataFrame that can be plotted.
    Parameters:
        df: DataFrame containing simulation data
        ID_SUSCEPTIBLE: ID representing susceptible state in Vadere (default: 1)
        ID_INFECTED: ID representing infected state in Vadere (default: 0)
        ID_REMOVED: ID representing recovered state in Vadere (default: 2)
    Returns:
        DataFrame with columns for time and counts of each state (S/I/R)
    """
    pedestrian_ids = df['pedestrianId'].unique()
    sim_times = df['simTime'].unique()
    # Added 'group-r' column to track recovered individuals
    group_counts = pd.DataFrame(columns=['simTime', 'group-s', 'group-i', 'group-r'])
    group_counts['simTime'] = sim_times
    group_counts['group-s'] = 0
    group_counts['group-i'] = 0
    group_counts['group-r'] = 0  # Initialize recovered count

    for pid in pedestrian_ids:
        # Get all state changes for this pedestrian
        simtime_group = df[df['pedestrianId'] == pid][['simTime', 'groupId-PID5']].values
        current_state = ID_SUSCEPTIBLE  # Everyone starts as susceptible
        group_counts.loc[group_counts['simTime'] >= 0, 'group-s'] += 1
        
        for (st, g) in simtime_group:
            if g != current_state:  # State change detected
                if g == ID_INFECTED and current_state == ID_SUSCEPTIBLE:
                    # S → I transition
                    current_state = g
                    group_counts.loc[group_counts['simTime'] > st, 'group-s'] -= 1
                    group_counts.loc[group_counts['simTime'] > st, 'group-i'] += 1
                elif g == ID_REMOVED and current_state == ID_INFECTED:
                    # I → R transition (new addition)
                    # Update counts when someone recovers
                    current_state = g
                    group_counts.loc[group_counts['simTime'] > st, 'group-i'] -= 1
                    group_counts.loc[group_counts['simTime'] > st, 'group-r'] += 1
    return group_counts

def create_folder_data_scatter(folder):
    """
    Create scatter plots from folder data showing S/I/R populations over time.
    Parameters:
        folder: Path to the folder containing simulation output
    Returns:
        tuple: (list of scatter plots, group counts DataFrame)
    """
    file_path = os.path.join(folder, "SIRinformation.csv")
    if not os.path.exists(file_path):
        return None, None
    data = pd.read_csv(file_path, delimiter=" ")

    # Define state IDs as used in Vadere
    ID_SUSCEPTIBLE = 1
    ID_INFECTED = 0
    ID_REMOVED = 2  # Added ID for recovered state

    # Get population counts over time
    group_counts = file_df_to_count_df(data, ID_INFECTED=ID_INFECTED, 
                                     ID_SUSCEPTIBLE=ID_SUSCEPTIBLE,
                                     ID_REMOVED=ID_REMOVED)
    
    # Create scatter plot for susceptible population
    scatter_s = go.Scatter(x=group_counts['simTime'],
                          y=group_counts['group-s'],
                          name='susceptible ' + os.path.basename(folder),
                          mode='lines')
    
    # Create scatter plot for infected population
    scatter_i = go.Scatter(x=group_counts['simTime'],
                          y=group_counts['group-i'],
                          name='infected ' + os.path.basename(folder),
                          mode='lines')
    
    # New addition: Create scatter plot for recovered population
    scatter_r = go.Scatter(x=group_counts['simTime'],
                          y=group_counts['group-r'],
                          name='recovered ' + os.path.basename(folder),
                          mode='lines')
    
    # Return all three scatter plots and the data
    return [scatter_s, scatter_i, scatter_r], group_counts