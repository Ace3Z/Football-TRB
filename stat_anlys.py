import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
from statsbombpy import sb
from match_finder import arg_matches, fr_matches
from trb import *



def get_cliffs_delta_explanation(cliff_d, ci_lower, ci_upper):
    """
    Provide explanation for Cliff's Delta results
    """
    try:
        cliff_d = float(cliff_d)
        ci_lower = float(ci_lower)
        ci_upper = float(ci_upper)
        
        if ci_lower > 0:
            return "Significantly faster recovery after events (CI entirely above 0)"
        elif ci_upper < 0:
            return "Significantly slower recovery after events (CI entirely below 0)"
        else:
            return "No significant difference (CI includes 0)"
    except (ValueError, TypeError):
        return "Unable to determine (invalid data)"

def apply_benjamini_hochberg_correction(mannwhitney_results, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction to Mann-Whitney test results
    """
    
    # Extract p-values
    p_values = []
    df_names = []
    
    for df_name, results in mannwhitney_results.items():
        if 'mw_p' in results and not pd.isna(results['mw_p']):
            p_values.append(results['mw_p'])
            df_names.append(df_name)
    
    if not p_values:
        print("No valid p-values found for correction.")
        return mannwhitney_results
    
    # Apply BH correction
    rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # Update results
    for i, df_name in enumerate(df_names):
        mannwhitney_results[df_name]['corrected_p_value'] = corrected_p_values[i]
        mannwhitney_results[df_name]['significant_corrected'] = rejected[i]
        mannwhitney_results[df_name]['correction_method'] = f"Benjamini-Hochberg (FDR), α={alpha}"
    
    # Print summary
    original_significant = sum(1 for p in p_values if p <= alpha)
    corrected_significant = sum(rejected)
    
    print(f"BH Correction Applied:")
    print(f"  Total tests: {len(p_values)}")
    print(f"  Originally significant (α={alpha}): {original_significant}")
    print(f"  Significant after BH correction: {corrected_significant}")
    print(f"  Tests no longer significant: {original_significant - corrected_significant}")
    
    return mannwhitney_results

def analyze_event_recovery_times(*team_dicts, output_dir="event_recovery_analysis", window_minutes=5, apply_multiple_correction=True, alpha=0.05):
    """
    Analyze recovery times around different event types in football matches.
    
    Uses separate Benjamini-Hochberg corrections for:
    - Match-level analysis: Controls FDR within each individual match
    - Team-level analysis: Controls FDR within each team's aggregated data
    
    Args:
        *team_dicts: Variable number of tuples, each containing (team_name, matches_dict)
        output_dir: Directory to save output CSVs (default: "event_recovery_analysis")
        window_minutes: Time window in minutes before and after events to analyze (default: 5)
        apply_multiple_correction: Whether to apply Benjamini-Hochberg correction (default: True)
        alpha: Significance level for corrections (default: 0.05)
    
    Returns:
        Dictionary containing dataframes with recovery times for each team and event type
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store all dataframes
    all_dataframes = {}
    
    # Dictionary to store Shapiro-Wilk test results
    normality_results = {}
    
    # Dictionary to store Mann-Whitney test results (for backward compatibility)
    mannwhitney_results = {}
    
    # Storage for separate analysis levels
    match_level_tests = []  # Individual match analyses
    team_level_tests = []   # Team-aggregated analyses
    
    # Store impact results for summary reporting
    match_impact_results = []
    team_impact_results = []
    
    print(f"\n{'='*80}")
    print("EVENT RECOVERY ANALYSIS WITH SEPARATE BH CORRECTION")
    print('='*80)
    print(f"BH Correction Strategy:")
    print(f"  Match-level: Separate correction within each individual match")
    print(f"  Team-level: Separate correction within each team's aggregated data")
    print(f"  Alpha level: {alpha}")
    print(f"  Apply correction: {apply_multiple_correction}")
    print('='*80)
    
    # Process all teams and their matches
    processed_matches = set()
    
    def cliffs_delta(group1, group2, n_bootstrap=1000, random_state=None):
        """
        Calculate Cliff's Delta effect size with bootstrap confidence intervals
        """
        
        # Convert to numpy arrays
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        n1 = len(group1)
        n2 = len(group2)
        
        if n1 == 0 or n2 == 0:
            return float('nan'), "Not applicable", float('nan'), float('nan')
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        def calculate_cliffs_delta(g1, g2):
            """Helper function to calculate Cliff's Delta for any two groups"""
            dominance = 0
            total_comparisons = len(g1) * len(g2)
            
            for x in g1:
                for y in g2:
                    if x > y:
                        dominance += 1
                    elif x < y:
                        dominance -= 1
            
            return dominance / total_comparisons
        
        # Calculate original Cliff's Delta
        delta = calculate_cliffs_delta(group1, group2)
        
        # Bootstrap procedure
        bootstrap_deltas = []
        
        for i in range(n_bootstrap):
            # Resample both groups with replacement
            bootstrap_group1 = np.random.choice(group1, size=n1, replace=True)
            bootstrap_group2 = np.random.choice(group2, size=n2, replace=True)
            
            # Calculate Cliff's Delta for bootstrap sample
            bootstrap_delta = calculate_cliffs_delta(bootstrap_group1, bootstrap_group2)
            bootstrap_deltas.append(bootstrap_delta)
        
        bootstrap_deltas = np.array(bootstrap_deltas)
        
        # Calculate 95% confidence intervals using percentile method
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
        
        # Ensure CI is within [-1, 1] bounds
        ci_lower = max(-1, ci_lower)
        ci_upper = min(1, ci_upper)
        
        # Interpret magnitude according to Romano et al. (2006)
        if abs(delta) < 0.147:
            magnitude = "Negligible"
        elif abs(delta) < 0.33:
            magnitude = "Small"
        elif abs(delta) < 0.474:
            magnitude = "Medium"
        else:
            magnitude = "Large"
        
        return delta, magnitude, ci_lower, ci_upper
        
    for team_data in team_dicts:
        # Unpack the team data
        team_name, matches_dict = team_data
        
        # Create team-specific directory
        team_output_dir = os.path.join(output_dir, team_name.replace(' ', '_').lower())
        os.makedirs(team_output_dir, exist_ok=True)
        
        # Create team-level subfolder
        team_level_dir = os.path.join(team_output_dir, "team_level")
        os.makedirs(team_level_dir, exist_ok=True)
        
        # Create stats folder in team-level directory
        team_stats_dir = os.path.join(team_level_dir, "stats")
        os.makedirs(team_stats_dir, exist_ok=True)
        
        print(f"\nProcessing matches for {team_name}...")
        
        # Initialize dictionaries for this team (aggregated across all matches)
        team_event_recoveries = {
            'all_goals': {'times': [], 'timing': [], 'count': 0, 'team': []},
            'substitutions': {'times': [], 'timing': [], 'count': 0, 'team': []},
            'tactical_shifts': {'times': [], 'timing': [], 'count': 0, 'team': []},
            'injuries': {'times': [], 'timing': [], 'count': 0, 'team': []},
            'yellow_cards': {'times': [], 'timing': [], 'count': 0, 'team': []},
            'red_cards': {'times': [], 'timing': [], 'count': 0, 'team': []}
        }
        
        # Team-specific normality results
        team_normality_results = []
        
        # Team-specific tests (to be corrected together)
        team_tests_for_this_team = []
        
        # Process all matches for this team
        for match_id, details in matches_dict.items():
            print(f"  Processing match {match_id} ({details['team']})...")
            
            # Create variable names based on team names
            team1 = details['team'].replace(' ', '_').lower()  # opposing team
            team2 = team_name.replace(' ', '_').lower()  # current team
            
            # Store the opposing team's name for reference
            opposing_team_name = details['team']
            
            # Get the time-to-recover DataFrame
            df_var_name = f"df_time_to_recover_{team2}_vs_{team1}"
            if df_var_name not in globals():
                print(f"    No time-to-recover data found for {df_var_name}")
                continue
                
            time_to_recover_df = globals()[df_var_name]
            
            # Ensure loss_time is datetime
            time_to_recover_df['loss_time'] = pd.to_datetime(time_to_recover_df['loss_time'])
            
            # Get events from the StatsBomb API
            events = sb.events(match_id=match_id)
            events['timestamp'] = pd.to_datetime(events['timestamp'], errors='coerce')
            
            # Filter for periods 1-4
            events = events[events['period'].isin([1, 2, 3, 4])]
            
            if events.empty:
                print(f"    No events found for periods 1-4 in Match {match_id}. Skipping...")
                continue
            
            # Create match-specific directory
            match_output_dir = os.path.join(team_output_dir, f"match_{match_id}")
            os.makedirs(match_output_dir, exist_ok=True)
            
            # Create stats folder in match directory
            match_stats_dir = os.path.join(match_output_dir, "stats")
            os.makedirs(match_stats_dir, exist_ok=True)
            
            # Initialize dictionaries for this match
            match_event_recoveries = {
                'all_goals': {'times': [], 'timing': [], 'count': 0, 'team': []},
                'substitutions': {'times': [], 'timing': [], 'count': 0, 'team': []},
                'tactical_shifts': {'times': [], 'timing': [], 'count': 0, 'team': []},
                'injuries': {'times': [], 'timing': [], 'count': 0, 'team': []},
                'yellow_cards': {'times': [], 'timing': [], 'count': 0, 'team': []},
                'red_cards': {'times': [], 'timing': [], 'count': 0, 'team': []}
            }
            
            # Match-specific normality results
            match_normality_results = []
            
            # Match-specific tests (to be corrected together)
            match_tests_for_this_match = []
            
            # Define event type filters
            event_types = {
                'all_goals': events[((events['type'] == 'Shot') & (events['shot_outcome'] == 'Goal')) | 
                                   (events['type'] == 'Own Goal For')].copy(),
                'substitutions': events[events['type'] == 'Substitution'].copy(),
                'tactical_shifts': events[events['type'] == 'Tactical Shift'].copy(),
                'injuries': events[events['type'] == 'Injury Stoppage'].copy(),
                'yellow_cards': events[(events['type'] == 'Foul Committed') & 
                                      (events['foul_committed_card'] == 'Yellow Card')].copy(),
                'red_cards': events[(events['type'] == 'Foul Committed') & 
                                   (events['foul_committed_card'] == 'Red Card')].copy()
            }
            
            # Process each type of event
            for event_type, event_events in event_types.items():
                # Count the number of events of this type (team-specific and match-specific)
                team_event_recoveries[event_type]['count'] += len(event_events)
                match_event_recoveries[event_type]['count'] += len(event_events)
                
                for _, event in event_events.iterrows():
                    event_time = event['timestamp']
                    event_period = event['period']
                    
                    # Define time window around the event
                    window_start = event_time - pd.Timedelta(minutes=window_minutes)
                    window_end = event_time + pd.Timedelta(minutes=window_minutes)
                    
                    # Get all ball loss and recovery events in the window
                    window_events = time_to_recover_df[
                        (time_to_recover_df['loss_time'] >= window_start) &
                        (time_to_recover_df['loss_time'] <= window_end) &
                        (time_to_recover_df['period'] == event_period) &
                        (time_to_recover_df['time_to_recover'] >= 1)
                    ].copy()
                    
                    # Label events as "before event" or "after event"
                    window_events['event_timing'] = window_events['loss_time'].apply(
                        lambda x: "before event" if x < event_time else "after event"
                    )
                    
                    # Add to both team-specific and match-specific data
                    for _, row in window_events.iterrows():
                        # Team data (aggregated)
                        team_event_recoveries[event_type]['times'].append(row['time_to_recover'])
                        team_event_recoveries[event_type]['timing'].append(row['event_timing'])
                        team_event_recoveries[event_type]['team'].append(row['team'])
                        
                        # Match data (specific to this match)
                        match_event_recoveries[event_type]['times'].append(row['time_to_recover'])
                        match_event_recoveries[event_type]['timing'].append(row['event_timing'])
                        match_event_recoveries[event_type]['team'].append(row['team'])
            
            # MATCH-LEVEL ANALYSIS: Create and analyze match-specific dataframes
            for event_type, data in match_event_recoveries.items():
                if data['times']:
                    # Create DataFrame for this event type in this match
                    df_name = f"df_{team_name.replace(' ', '_').lower()}_match_{match_id}_{event_type}"
                    
                    df = pd.DataFrame({
                        'time_to_recover': data['times'],
                        'event_timing': data['timing'],
                        'team': data['team']
                    })
                    
                    # Save the dataframe
                    globals()[df_name] = df
                    all_dataframes[df_name] = df
                    
                    # Save to match-specific directory
                    csv_path = f"{match_output_dir}/{event_type}_recovery_times.csv"
                    df.to_csv(csv_path, index=False)
                    
                    # Filter data to only include the specified team for statistical analysis
                    team_data_df = df[df['team'] == team_name]
                    team_recovery_times = team_data_df['time_to_recover'].tolist()
                    
                    # Perform Shapiro-Wilk test if we have sufficient data
                    if len(team_recovery_times) >= 8:
                        normality_results[df_name] = {}
                        
                        if len(team_recovery_times) < 5000:
                            shapiro_stat, shapiro_p = stats.shapiro(team_recovery_times)
                            normality_results[df_name]['shapiro_stat'] = shapiro_stat
                            normality_results[df_name]['shapiro_p'] = shapiro_p
                            normality_results[df_name]['shapiro_normal'] = shapiro_p > 0.05
                            normality_results[df_name]['sample_size'] = len(team_recovery_times)
                            
                            match_result = {
                                'Event Type': event_type.replace('_', ' ').title(),
                                'Sample Size': len(team_recovery_times),
                                'Shapiro-Wilk Statistic': f"{shapiro_stat:.4f}",
                                'p-value': f"{shapiro_p:.18f}",
                                'Normal Distribution': "Yes" if shapiro_p > 0.05 else "No"
                            }
                            match_normality_results.append(match_result)
                    
                    # Split data into before and after events for Mann-Whitney test
                    before_times = team_data_df[team_data_df['event_timing'] == 'before event']['time_to_recover'].tolist()
                    after_times = team_data_df[team_data_df['event_timing'] == 'after event']['time_to_recover'].tolist()
                    
                    # Calculate averages
                    avg_before = np.mean(before_times) if len(before_times) > 0 else float('nan')
                    avg_after = np.mean(after_times) if len(after_times) > 0 else float('nan')
                    faster_after = avg_after < avg_before if not (pd.isna(avg_before) or pd.isna(avg_after)) else False
                    
                    # Calculate Cliff's Delta
                    cliff_d = float('nan')
                    cliff_magnitude = "Not applicable"
                    ci_lower = float('nan')
                    ci_upper = float('nan')
                    
                    if len(before_times) > 0 and len(after_times) > 0:
                        cliff_d, cliff_magnitude, ci_lower, ci_upper = cliffs_delta(before_times, after_times, random_state=8)
                    
                    # Perform Mann-Whitney U test if there's enough data
                    if len(before_times) >= 5 and len(after_times) >= 5:
                        mw_stat, mw_p = stats.mannwhitneyu(before_times, after_times, alternative='two-sided')
                        
                        # Store match-level test result
                        match_test = {
                            'analysis_level': 'match',
                            'team': team_name,
                            'match_id': match_id,
                            'opposing_team': opposing_team_name,
                            'event_type': event_type,
                            'p_value': mw_p,
                            'test_statistic': mw_stat,
                            'before_n': len(before_times),
                            'after_n': len(after_times),
                            'avg_before': avg_before,
                            'avg_after': avg_after,
                            'faster_after': faster_after,
                            'cliffs_delta': cliff_d,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'effect_size': cliff_magnitude,
                            'significant_uncorrected': mw_p <= alpha
                        }
                        
                        match_tests_for_this_match.append(match_test)
                        match_level_tests.append(match_test)
                        
                        # Store in mannwhitney_results for backward compatibility
                        mannwhitney_results[df_name] = {
                            'mw_stat': mw_stat,
                            'mw_p': mw_p,
                            'before_sample_size': len(before_times),
                            'after_sample_size': len(after_times),
                            'avg_before': avg_before,
                            'avg_after': avg_after,
                            'faster_after': faster_after,
                            'significant': mw_p <= 0.05,
                            'cliffs_delta': cliff_d,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'effect_size': cliff_magnitude
                        }
            
            # APPLY BH CORRECTION TO MATCH'S TESTS
            if apply_multiple_correction and match_tests_for_this_match:
                print(f"    Applying BH correction to {len(match_tests_for_this_match)} tests in Match {match_id}...")
                
                match_p_values = [test['p_value'] for test in match_tests_for_this_match]
                rejected, corrected_p_values, _, _ = multipletests(
                    match_p_values, alpha=alpha, method='fdr_bh'
                )
                
                # Update match test results with correction
                for i, test in enumerate(match_tests_for_this_match):
                    test['bh_corrected_p'] = corrected_p_values[i]
                    test['significant_after_bh'] = rejected[i]
                    test['correction_scope'] = f"Within Match {match_id}"
                    test['correction_method'] = f"Benjamini-Hochberg (Match-level, α={alpha})"
                
                # Print match correction summary
                original_sig = sum(1 for p in match_p_values if p <= alpha)
                corrected_sig = sum(rejected)
                print(f"    Match {match_id}: {original_sig} → {corrected_sig} significant tests after BH correction")
                
                # Create match impact results
                for test in match_tests_for_this_match:
                    if test['significant_after_bh']:
                        match_impact = {
                            'Team': team_name,
                            'Match ID': match_id,
                            'Opposing Team': opposing_team_name,
                            'Event Type': test['event_type'].replace('_', ' ').title(),
                            'Significant Impact': 'Yes',
                            'Avg TRB Before (s)': f"{test['avg_before']:.2f}",
                            'Avg TRB After (s)': f"{test['avg_after']:.2f}",
                            'Faster After Event': 'Yes' if test['faster_after'] else 'No',
                            'Original p-value': f"{test['p_value']:.6f}",
                            'BH Corrected p-value': f"{test['bh_corrected_p']:.6f}",
                            'Correction Method': test['correction_method']
                        }
                        match_impact_results.append(match_impact)
            
            # Save match-specific normality test results
            if match_normality_results:
                match_results_df = pd.DataFrame(match_normality_results)
                match_normality_csv_path = f"{match_stats_dir}/shapiro_wilk_test_results.csv"
                match_results_df.to_csv(match_normality_csv_path, index=False)
            
            # Save match-specific test results
            if match_tests_for_this_match:
                match_test_df = pd.DataFrame(match_tests_for_this_match)
                match_test_csv_path = f"{match_stats_dir}/mann_whitney_test_results.csv"
                match_test_df.to_csv(match_test_csv_path, index=False)
        
        print(f"  Performing team-level analysis for {team_name}...")
        
        for event_type, data in team_event_recoveries.items():
            if data['times']:
                # Create DataFrame for this event type aggregated across all matches
                df_name = f"df_{team_name.replace(' ', '_').lower()}_{event_type}"
                
                df = pd.DataFrame({
                    'time_to_recover': data['times'],
                    'event_timing': data['timing'],
                    'team': data['team']
                })
                
                # Save the dataframe
                globals()[df_name] = df
                all_dataframes[df_name] = df
                
                # Save to team-level directory
                csv_path = f"{team_level_dir}/{event_type}_recovery_times.csv"
                df.to_csv(csv_path, index=False)
                
                # Filter data to only include the specified team for statistical analysis
                team_data_df = df[df['team'] == team_name]
                team_recovery_times = team_data_df['time_to_recover'].tolist()
                
                # Perform Shapiro-Wilk test if we have sufficient data
                if len(team_recovery_times) >= 8:
                    normality_results[df_name] = {}
                    
                    if len(team_recovery_times) < 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(team_recovery_times)
                        normality_results[df_name]['shapiro_stat'] = shapiro_stat
                        normality_results[df_name]['shapiro_p'] = shapiro_p
                        normality_results[df_name]['shapiro_normal'] = shapiro_p > 0.05
                        normality_results[df_name]['sample_size'] = len(team_recovery_times)
                        
                        team_result = {
                            'Event Type': event_type.replace('_', ' ').title(),
                            'Sample Size': len(team_recovery_times),
                            'Shapiro-Wilk Statistic': f"{shapiro_stat:.4f}",
                            'p-value': f"{shapiro_p:.18f}",
                            'Normal Distribution': "Yes" if shapiro_p > 0.05 else "No"
                        }
                        team_normality_results.append(team_result)
                
                # Split data into before and after events for Mann-Whitney test
                before_times = team_data_df[team_data_df['event_timing'] == 'before event']['time_to_recover'].tolist()
                after_times = team_data_df[team_data_df['event_timing'] == 'after event']['time_to_recover'].tolist()
                
                # Calculate averages
                avg_before = np.mean(before_times) if len(before_times) > 0 else float('nan')
                avg_after = np.mean(after_times) if len(after_times) > 0 else float('nan')
                faster_after = avg_after < avg_before if not (pd.isna(avg_before) or pd.isna(avg_after)) else False
                
                # Calculate Cliff's Delta
                cliff_d = float('nan')
                cliff_magnitude = "Not applicable"
                ci_lower = float('nan')
                ci_upper = float('nan')
                
                if len(before_times) > 0 and len(after_times) > 0:
                    cliff_d, cliff_magnitude, ci_lower, ci_upper = cliffs_delta(before_times, after_times, random_state=8)
                
                # Perform Mann-Whitney U test if there's enough data
                if len(before_times) >= 5 and len(after_times) >= 5:
                    mw_stat, mw_p = stats.mannwhitneyu(before_times, after_times, alternative='two-sided')
                    
                    # Store team-level test result
                    team_test = {
                        'analysis_level': 'team',
                        'team': team_name,
                        'match_id': 'all_matches',
                        'event_type': event_type,
                        'p_value': mw_p,
                        'test_statistic': mw_stat,
                        'before_n': len(before_times),
                        'after_n': len(after_times),
                        'avg_before': avg_before,
                        'avg_after': avg_after,
                        'faster_after': faster_after,
                        'cliffs_delta': cliff_d,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'effect_size': cliff_magnitude,
                        'significant_uncorrected': mw_p <= alpha
                    }
                    # Store in mannwhitney_results for backward compatibility
                    mannwhitney_results[df_name] = {
                        'mw_stat': mw_stat,
                        'mw_p': mw_p,
                        'before_sample_size': len(before_times),
                        'after_sample_size': len(after_times),
                        'avg_before': avg_before,
                        'avg_after': avg_after,
                        'faster_after': faster_after,
                        'significant': mw_p <= 0.05,
                        'cliffs_delta': cliff_d,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'effect_size': cliff_magnitude
                    }
                    team_tests_for_this_team.append(team_test)
                    team_level_tests.append(team_test)
        
        # APPLY BH CORRECTION TO THIS TEAM'S TESTS
        if apply_multiple_correction and team_tests_for_this_team:
            print(f"  Applying BH correction to {len(team_tests_for_this_team)} team-level tests for {team_name}...")
            
            team_p_values = [test['p_value'] for test in team_tests_for_this_team]
            rejected, corrected_p_values, _, _ = multipletests(
                team_p_values, alpha=alpha, method='fdr_bh'
            )
            
            # Update team test results with correction
            for i, test in enumerate(team_tests_for_this_team):
                test['bh_corrected_p'] = corrected_p_values[i]
                test['significant_after_bh'] = rejected[i]
                test['correction_scope'] = f"Within {team_name} Team"
                test['correction_method'] = f"Benjamini-Hochberg (Team-level, α={alpha})"
            
            # Print team correction summary
            original_sig = sum(1 for p in team_p_values if p <= alpha)
            corrected_sig = sum(rejected)
            print(f"  {team_name}: {original_sig} → {corrected_sig} significant tests after BH correction")
            
            # Create team impact results
            for test in team_tests_for_this_team:
                if test['significant_after_bh']:
                    team_impact = {
                        'Team': team_name,
                        'Match ID': 'All Matches',
                        'Event Type': test['event_type'].replace('_', ' ').title(),
                        'Significant Impact': 'Yes',
                        'Avg TRB Before (s)': f"{test['avg_before']:.2f}",
                        'Avg TRB After (s)': f"{test['avg_after']:.2f}",
                        'Faster After Event': 'Yes' if test['faster_after'] else 'No',
                        'Original p-value': f"{test['p_value']:.6f}",
                        'BH Corrected p-value': f"{test['bh_corrected_p']:.6f}",
                        'Correction Method': test['correction_method']
                    }
                    team_impact_results.append(team_impact)
        
        # Save team-specific normality test results
        if team_normality_results:
            team_results_df = pd.DataFrame(team_normality_results)
            team_normality_csv_path = f"{team_stats_dir}/shapiro_wilk_test_results.csv"
            team_results_df.to_csv(team_normality_csv_path, index=False)
        
        # Save team-specific test results
        if team_tests_for_this_team:
            team_test_df = pd.DataFrame(team_tests_for_this_team)
            team_test_csv_path = f"{team_stats_dir}/mann_whitney_test_results.csv"
            team_test_df.to_csv(team_test_csv_path, index=False)
    
    # Save normality test results
    if normality_results:
        results_data = []
        for df_name, results in normality_results.items():
            team_name = df_name.split('_')[1] if len(df_name.split('_')) > 1 else "Unknown"
            
            if "match_" in df_name:
                match_id = df_name.split('_match_')[1].split('_')[0]
                event_type = '_'.join(df_name.split(f'_match_{match_id}_')[1:])
                
                if event_type != "all_events":
                    results_data.append({
                        'Team': team_name.replace('_', ' ').title(),
                        'Match ID': match_id,
                        'Event Type': event_type.replace('_', ' ').title(),
                        'Sample Size': results.get('sample_size', "N/A"),
                        'Shapiro-Wilk Statistic': f"{results.get('shapiro_stat', 'N/A'):.4f}" if isinstance(results.get('shapiro_stat'), (float, int)) else "N/A",
                        'p-value': f"{results.get('shapiro_p', 'N/A'):.18f}" if isinstance(results.get('shapiro_p'), (float, int)) else "N/A",
                        'Normal Distribution': "Yes" if results.get('shapiro_normal', False) else "No"
                    })
            else:
                event_type = '_'.join(df_name.split('_')[2:])
                
                if event_type != "all_events":
                    results_data.append({
                        'Team': team_name.replace('_', ' ').title(),
                        'Match ID': "All Matches",
                        'Event Type': event_type.replace('_', ' ').title(),
                        'Sample Size': results.get('sample_size', "N/A"),
                        'Shapiro-Wilk Statistic': f"{results.get('shapiro_stat', 'N/A'):.4f}" if isinstance(results.get('shapiro_stat'), (float, int)) else "N/A",
                        'p-value': f"{results.get('shapiro_p', 'N/A'):.18f}" if isinstance(results.get('shapiro_p'), (float, int)) else "N/A",
                        'Normal Distribution': "Yes" if results.get('shapiro_normal', False) else "No"
                    })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            normality_csv_path = f"{output_dir}/shapiro_wilk_test_results.csv"
            results_df.to_csv(normality_csv_path, index=False)
    
    # Save match-level results
    if match_level_tests:
        match_df = pd.DataFrame(match_level_tests)
        match_csv_path = f"{output_dir}/match_level_mann_whitney_results.csv"
        match_df.to_csv(match_csv_path, index=False)
        
        # Save significant match results
        if apply_multiple_correction:
            sig_match = match_df[match_df['significant_after_bh'] == True]
        else:
            sig_match = match_df[match_df['significant_uncorrected'] == True]
        
        if not sig_match.empty:
            sig_match_csv = f"{output_dir}/significant_match_level_results.csv"
            sig_match.to_csv(sig_match_csv, index=False)
    
    # Save team-level results
    if team_level_tests:
        team_df = pd.DataFrame(team_level_tests)
        team_csv_path = f"{output_dir}/team_level_mann_whitney_results.csv"
        team_df.to_csv(team_csv_path, index=False)
        
        # Save significant team results
        if apply_multiple_correction:
            sig_team = team_df[team_df['significant_after_bh'] == True]
        else:
            sig_team = team_df[team_df['significant_uncorrected'] == True]
        
        if not sig_team.empty:
            sig_team_csv = f"{output_dir}/significant_team_level_results.csv"
            sig_team.to_csv(sig_team_csv, index=False)
    
    # Save impact results
    all_impacts = match_impact_results + team_impact_results
    if all_impacts:
        all_impacts_df = pd.DataFrame(all_impacts)
        all_impacts_csv = f"{output_dir}/all_significant_impacts.csv"
        all_impacts_df.to_csv(all_impacts_csv, index=False)
        
        # Save separate files for match and team impacts
        if match_impact_results:
            match_impacts_df = pd.DataFrame(match_impact_results)
            match_impacts_csv = f"{output_dir}/significant_match_impacts.csv"
            match_impacts_df.to_csv(match_impacts_csv, index=False)
        
        if team_impact_results:
            team_impacts_df = pd.DataFrame(team_impact_results)
            team_impacts_csv = f"{output_dir}/significant_team_impacts.csv"
            team_impacts_df.to_csv(team_impacts_csv, index=False)

    print(f"\n{'='*80}")
    print("FINAL ANALYSIS SUMMARY")
    print('='*80)
    
    if match_level_tests:
        match_original = sum(1 for test in match_level_tests if test['significant_uncorrected'])
        match_corrected = sum(1 for test in match_level_tests if test.get('significant_after_bh', False))
        print(f"Match-level tests: {len(match_level_tests)} total")
        print(f"  Originally significant: {match_original}")
        print(f"  Significant after BH correction: {match_corrected}")
    
    if team_level_tests:
        team_original = sum(1 for test in team_level_tests if test['significant_uncorrected'])
        team_corrected = sum(1 for test in team_level_tests if test.get('significant_after_bh', False))
        print(f"Team-level tests: {len(team_level_tests)} total")
        print(f"  Originally significant: {team_original}")
        print(f"  Significant after BH correction: {team_corrected}")
    
    # Print significant findings
    if apply_multiple_correction:
        if team_impact_results:
            print(f"\nSIGNIFICNT TEAM-LEVEL FINDINGS (After BH Correction):")
            for impact in team_impact_results:
                result_type = "FASTER" if impact['Faster After Event'] == 'Yes' else "SLOWER"
                print(f"  • {impact['Team']} recovered {result_type} after {impact['Event Type']} (p={impact['BH Corrected p-value']})")
        
        if match_impact_results:
            print(f"\nSIGNIFICANT MATCH-LEVEL FINDINGS (After BH Correction):")
            for impact in match_impact_results:
                result_type = "FASTER" if impact['Faster After Event'] == 'Yes' else "SLOWER"
                print(f"  • {impact['Team']} recovered {result_type} after {impact['Event Type']} in Match {impact['Match ID']} (p={impact['BH Corrected p-value']})")
    
    print('='*80)
    
    return all_dataframes, normality_results, mannwhitney_results, match_level_tests, team_level_tests

dataframes, normality_results, mannwhitney_results, match_tests, team_tests = analyze_event_recovery_times(
    ("Argentina", arg_matches), 
    ("France", fr_matches),
    apply_multiple_correction=True,
    alpha=0.05
)