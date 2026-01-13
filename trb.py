import pandas as pd
import numpy as np
import os
from statsbombpy import sb
from match_finder import arg_matches, fr_matches

def analyze_time_to_recover(home_team, matches, output_dir="output"):
    """
    Analyze time to recover for matches between home team and opponents.
    Creates global variables for each match's time-to-recover dataframe.
    """    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for match_id, details in matches.items():
        print(f"\nProcessing match with Match ID: {match_id} ({details['team']})...")

        # 1. Pull events for the match
        try:
            events = sb.events(match_id=match_id)
        except NameError:
            print(f"Error: StatsBomb client 'sb' not defined. Please define it before calling this function.")
            continue

        # 2. Parse and sort events
        events['timestamp'] = pd.to_datetime(events['timestamp'], errors='coerce')
        events = events[events['period'].isin([1, 2, 3, 4])]
        
        if events.empty:
            print(f"No valid events found for Match {match_id}. Skipping...")
            continue

        events = events.sort_values(by=['period', 'timestamp']).reset_index(drop=True)

        # 3. Identify Goal and Own Goal Events
        goal_events = events[
            ((events['type'] == 'Shot') & (events['shot_outcome'] == 'Goal')) |
            (events['type'] == 'Own Goal For')
        ].copy()

        # 4. Define Conditions for Ball-Loss and Ball-Recovery Events
        ball_loss_conditions = [
            (events['type'] == 'Pass') & (events['pass_outcome'] == 'Incomplete'),
            (events['type'] == 'Pass') & (events['pass_outcome'] == 'Intercepted'),
            (events['type'] == 'Pass') & (events['pass_type'].isin(['Long Ball', 'Through Ball'])) & (events['pass_outcome'] == 'Incomplete'),
            (events['type'] == 'Carry') & (events['under_pressure'] == True),
            (events['type'] == 'Duel') & (events['duel_outcome'] == 'Lost'),
            (events['type'] == 'Dribble') & (events['dribble_outcome'] == 'Unsuccessful'),
            (events['type'] == 'Goalkeeper Distribution') & (events['goalkeeper_outcome'] == 'Unsuccessful'),
            events['type'].isin(['Dispossessed', 'Turnover', 'Interception', 'Error']),
            events['type'] == 'Foul Committed'
        ]
        combined_ball_loss_cond = np.logical_or.reduce(ball_loss_conditions)

        ball_recovery_conditions = [
            (events['type'] == 'Duel') & (events['duel_outcome'] == 'Won'),
            (events['type'] == 'Pass') & (events['pass_outcome'] == 'Successful'),
            (events['type'] == 'Interception') & (events['interception_outcome'] == 'Successful'),
            (events['type'] == 'Goalkeeper Action') & (events['goalkeeper_outcome'].isin(['Saved', 'Caught'])),
            events['type'].isin(['Ball Recovery', 'Interception', 'Tackle', 'Goalkeeper Possession']),
            events['type'] == 'Foul Won'
        ]
        combined_recovery_cond = np.logical_or.reduce(ball_recovery_conditions)

        # 5. Compute Time-to-Recover in Segments Between Goals
        period_results_list = []

        for period_val, period_df in events.groupby('period'):
            if period_val not in [1, 2, 3, 4]:
                continue

            period_df = period_df.sort_values(by='timestamp').reset_index(drop=True)

            # Get goal timestamps in this period
            goal_timestamps = goal_events.loc[goal_events['period'] == period_val, 'timestamp'].tolist()
            goal_timestamps.append(pd.Timestamp.max)  # Ensure last segment is covered

            # Split period into goal segments
            start_time = period_df['timestamp'].min()
            for goal_time in goal_timestamps:
                segment_df = period_df[(period_df['timestamp'] >= start_time) & (period_df['timestamp'] < goal_time)]

                # Identify ball-loss and ball-recovery events in this segment
                segment_loss_events = segment_df[
                    (segment_df['type'] == 'Pass') & (segment_df['pass_outcome'] == 'Incomplete') |
                    (segment_df['type'] == 'Pass') & (segment_df['pass_outcome'] == 'Intercepted') |
                    (segment_df['type'] == 'Pass') & (segment_df['pass_type'].isin(['Long Ball', 'Through Ball'])) & (segment_df['pass_outcome'] == 'Incomplete') |
                    (segment_df['type'] == 'Carry') & (segment_df['under_pressure'] == True) |
                    (segment_df['type'] == 'Duel') & (segment_df['duel_outcome'] == 'Lost') |
                    (segment_df['type'] == 'Dribble') & (segment_df['dribble_outcome'] == 'Unsuccessful') |
                    (segment_df['type'] == 'Goalkeeper Distribution') & (segment_df['goalkeeper_outcome'] == 'Unsuccessful') |
                    segment_df['type'].isin(['Dispossessed', 'Turnover', 'Interception', 'Error']) |
                    (segment_df['type'] == 'Foul Committed')
                ].copy()

                segment_recovery_events = segment_df[
                    (segment_df['type'] == 'Duel') & (segment_df['duel_outcome'] == 'Won') |
                    (segment_df['type'] == 'Pass') & (segment_df['pass_outcome'] == 'Successful') |
                    (segment_df['type'] == 'Interception') & (segment_df['interception_outcome'] == 'Successful') |
                    (segment_df['type'] == 'Goalkeeper Action') & (segment_df['goalkeeper_outcome'].isin(['Saved', 'Caught'])) |
                    segment_df['type'].isin(['Ball Recovery', 'Interception', 'Tackle', 'Goalkeeper Possession']) |
                    (segment_df['type'] == 'Foul Won')
                ].copy()

                if segment_loss_events.empty or segment_recovery_events.empty:
                    start_time = goal_time  # Move to the next segment
                    continue

                # Store results
                time_to_recover = []

                latest_recovery_time = None  # Track the last recovery time

                for _, loss_event in segment_loss_events.iterrows():
                    team = loss_event['team']
                    loss_time = loss_event['timestamp']

                    # Skip new losses if waiting for recovery
                    if latest_recovery_time and loss_time < latest_recovery_time:
                        continue  # Ignore this loss because a recovery hasn't happened yet

                    # Find the next valid recovery
                    valid_recoveries = segment_recovery_events[
                        (segment_recovery_events['team'] == team) &
                        (segment_recovery_events['timestamp'] > loss_time)
                    ]

                    if not valid_recoveries.empty:
                        recovery_event = valid_recoveries.iloc[0]
                        recovery_time = recovery_event['timestamp']
                        time_diff = (recovery_time - loss_time).total_seconds()

                        if time_diff >= 1:
                            time_to_recover.append({
                                'team': team,
                                'period': period_val,
                                'loss_time': loss_time,
                                'recovery_time': recovery_time,
                                'time_to_recover': time_diff
                            })
                            latest_recovery_time = recovery_time  # Update latest recovery time

                period_results = pd.DataFrame(time_to_recover)
                if not period_results.empty:
                    period_results_list.append(period_results)

                start_time = goal_time  # Move to the next segment after a goal

        # 6. Create a Global Variable for Each Match
        if period_results_list:
            # Get both team names and format them for the variable name and filename
            team1 = details['team'].replace(' ', '_').lower()
            team2 = home_team.replace(' ', '_').lower()
            
            # Concatenate all period results
            final_df = pd.concat(period_results_list, ignore_index=True)
            
            # Create variable name with both teams
            df_var_name = f"df_time_to_recover_{team2}_vs_{team1}"
            
            # Create a global variable with the appropriate name
            globals()[df_var_name] = final_df
            
            # Save to CSV with both team names
            csv_filename = f"{output_dir}/time_to_recover_{team2}_vs_{team1}.csv"
            final_df.to_csv(csv_filename, index=False)

            print(f"** Global variable '{df_var_name}' created for match {match_id}: {team2} vs {team1} **")
            print(f"Data also saved to {csv_filename}")
        else:
            print(f"No time-to-recover results for Match {match_id} in any period.")

analyze_time_to_recover('Argentina', arg_matches)
analyze_time_to_recover('France', fr_matches)