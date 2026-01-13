import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch
from statsbombpy import sb
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats
import json
import pycountry
from datetime import datetime
import os
import matplotlib as mpl
import matplotlib.colors as mcolors

from match_finder import arg_matches, fr_matches
from trb import *
from stat_anlys import *
from game_timeline import *


def test_recovery_time_normality(*team_dicts, output_dir="visuals"): 
    """
    Performs normality tests on recovery time data for matches involving teams in parameters.
    Creates QQ plots and KDE plots for each match and period.
    Features sequential team-match filtering where match options depend on team selection.
    
    Args:
        *team_dicts: Variable number of tuples, each containing (team_name, team_color, matches_dict)
        output_dir: Directory to save the output plots and reports
    
    Returns:
        Dictionary with normality test results for each match and period
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define periods to consider, including "All" for combined data
    numeric_periods = [1, 2, 3, 4]
    periods = numeric_periods + ["All"]
    
    # Dictionary to store test results
    normality_results = {}
    
    # Create a set of teams we're explicitly analyzing (from parameters)
    target_teams = {team_name for team_name, _, _ in team_dicts}
    print(f"Analyzing only matches involving these teams: {', '.join(target_teams)}")
    
    # Get team colors
    team_colors = {team_name: team_color for team_name, team_color, _ in team_dicts}
    
    # Define a set of distinct colors for mean lines that are visually different from typical team colors
    mean_line_colors = [
        "#FF4500",  # OrangeRed
        "#32CD32",  # LimeGreen
        "#9932CC",  # DarkOrchid
        "#FF8C00",  # DarkOrange
        "#1E90FF",  # DodgerBlue
        "#FF1493",  # DeepPink
        "#8A2BE2",  # BlueViolet
        "#FF6347",  # Tomato
        "#20B2AA",  # LightSeaGreen
        "#CD5C5C",  # IndianRed
    ]

    # Create a dictionary to store the mean line color for each team
    team_mean_colors = {}

    # Assign colors to teams in a deterministic way
    for i, team_name in enumerate(sorted(target_teams)):
        team_mean_colors[team_name] = mean_line_colors[i % len(mean_line_colors)]
    
    # Define a set of different dash styles for each period
    period_dash_styles = {
        "All": "solid",
        1: "dash",
        2: "dot",
        3: "dashdot",
        4: "longdash"
    }

    # Define line widths for better visual differentiation
    period_line_widths = {
        "All": 3,
        1: 2,
        2: 2,
        3: 2,
        4: 2
    }
    
    # Store match data for the filtering feature
    match_data = {}  # Will store all match information
    team_matches = {}  # Will map teams to their matches
    results_data = []  # Will store all result rows
    
    # Process all teams and their matches
    processed_matches = set()
    
    for team_name, team_color, matches_dict in team_dicts:
        print(f"\nProcessing normality tests for {team_name}...")
        
        # Initialize team_matches entry
        if team_name not in team_matches:
            team_matches[team_name] = []
        
        # Process all matches for this team
        for match_id, details in matches_dict.items():
            # Skip if we've already processed this match
            if match_id in processed_matches:
                print(f"Skipping already processed Match ID: {match_id}")
                continue
                
            try:
                print(f"Processing Match ID: {match_id} ({details['team']})...")
                
                # Mark this match as processed
                processed_matches.add(match_id)
                
                # Get the team names
                primary_team = details['team']
                
                # Create variable names based on team names
                team1 = primary_team.replace(' ', '_').lower()
                team2 = team_name.replace(' ', '_').lower()
                
                # Initialize match results in the dictionary
                if match_id not in normality_results:
                    normality_results[match_id] = {
                        'primary_team': primary_team,
                        'team_name': team_name,
                        'periods': {}
                    }
                
                # Retrieve the main team data 
                df_main = globals()[f"df_time_to_recover_{team2}_vs_{team1}"]
                
                # Get all teams in this match
                match_teams = df_main['team'].unique()
                match_teams_in_target = [t for t in match_teams if t in target_teams]
                
                # Store match information for filtering
                opponent_team = [t for t in match_teams if t in target_teams and t != primary_team]
                opponent_name = opponent_team[0] if opponent_team else "Unknown"
                match_name = f"{primary_team} vs {opponent_name}"
                
                # Store match data
                match_data[match_id] = {
                    'match_id': match_id,
                    'match_name': match_name,
                    'teams': match_teams_in_target
                }
                
                # Add this match to each team's list
                for t in match_teams_in_target:
                    if t not in team_matches:
                        team_matches[t] = []
                    team_matches[t].append(match_id)
                
                # Create output folder for this match
                match_folder = os.path.join(output_dir, f"match_{match_id}")
                os.makedirs(match_folder, exist_ok=True)
                
                # Create KDE plot for the entire match (all teams, all periods)
                # First gather all recovery time data
                all_data = []
                team_period_data = {}
                
                for current_team in match_teams:
                    # Skip teams not in our target list
                    if current_team not in target_teams:
                        continue
                    
                    # Initialize team data
                    team_period_data[current_team] = {}
                    
                    # Get the team color
                    team_color = team_colors.get(current_team, "#808080")
                    
                    # Process each numeric period
                    combined_data = pd.DataFrame()
                    
                    for p in numeric_periods:
                        period_data = df_main[(df_main['team'] == current_team) & (df_main['period'] == p)]
                        
                        # Store period data if available
                        if len(period_data) >= 3:
                            recovery_times = period_data['time_to_recover'].dropna().values
                            team_period_data[current_team][p] = recovery_times
                            combined_data = pd.concat([combined_data, period_data])
                            
                            # Add to all data collection for match-wide KDE
                            all_data.append({
                                'team': current_team,
                                'period': p,
                                'color': team_color,
                                'data': recovery_times
                            })
                    
                    # Add combined "All periods" data if we have anything
                    if not combined_data.empty and len(combined_data) >= 3:
                        recovery_times = combined_data['time_to_recover'].dropna().values
                        team_period_data[current_team]["All"] = recovery_times
                        # Add to all data collection
                        all_data.append({
                            'team': current_team,
                            'period': "All",
                            'color': team_color,
                            'data': recovery_times
                        })
                
                # Create match-wide KDE plot comparing all periods for all teams
                if all_data:
                    # Sort data entries to ensure consistent order (teams, then periods)
                    all_data.sort(key=lambda x: (x['team'], 0 if x['period'] == "All" else x['period']))
                    
                    # Create KDE figure
                    fig_kde = go.Figure()
                    
                    # Process each team
                    for team in set(entry['team'] for entry in all_data):
                        team_color = next(entry['color'] for entry in all_data if entry['team'] == team)
                        team_data = [entry for entry in all_data if entry['team'] == team]
                        
                        # First add histogram with low opacity for each team's "All Periods" data only
                        all_periods_data = next((entry for entry in team_data if entry['period'] == "All"), None)
                        if all_periods_data:
                            fig_kde.add_trace(go.Histogram(
                                x=all_periods_data['data'],
                                histnorm='probability density',
                                name=f"{team} (Histogram)",
                                marker=dict(
                                    color=team_color, 
                                    opacity=0.2,
                                    line=dict(color=team_color, width=1)
                                ),
                                legendgroup=team,
                                legendgrouptitle=dict(text=team),
                                showlegend=True
                            ))
                        
                        # Then update the part where KDE traces are added for each period:
                        for data_entry in team_data:
                            period = data_entry['period']
                            period_label = "All Periods" if period == "All" else f"Period {period}"
                            
                            # Create KDE
                            x_range = np.linspace(min(data_entry['data']) - 0.5, max(data_entry['data']) + 0.5, 1000)
                            kde = stats.gaussian_kde(data_entry['data'])
                            
                            # Use the defined dash styles and line widths based on period
                            dash_style = period_dash_styles[period]
                            line_width = period_line_widths[period]
                            
                            fig_kde.add_trace(go.Scatter(
                                x=x_range,
                                y=kde(x_range),
                                mode='lines',
                                name=f"{team} - {period_label}",
                                line=dict(color=team_color, width=line_width, dash=dash_style),
                                legendgroup=team
                            ))
                            
                            # Add data points as scatter for "All Periods" only
                            if period == "All":
                                # Create y-positions that follow the KDE curve
                                scatter_y = []
                                for t in data_entry['data']:
                                    # Find closest point in x_range
                                    idx = np.abs(x_range - t).argmin()
                                    # Get corresponding y-value from KDE and adjust with random component
                                    y_val = kde(x_range)[idx] * np.random.uniform(0.05, 0.15)
                                    scatter_y.append(y_val)
                                
                                fig_kde.add_trace(go.Scatter(
                                    x=data_entry['data'],
                                    y=scatter_y,
                                    mode='markers',
                                    name=f"{team} - Data Points",  # Include team name
                                    marker=dict(
                                        color=team_color, 
                                        size=8, 
                                        symbol='circle-open',
                                        line=dict(width=1.5, color=team_color)
                                    ),
                                    legendgroup=team,
                                    showlegend=True  # Show in legend
                                ))

                        # Add vertical line at mean for each team's "All Periods" data
                        all_periods_data = next((entry for entry in team_data if entry['period'] == "All"), None)
                        if all_periods_data:
                            mean_val = np.mean(all_periods_data['data'])
                            
                            # Create KDE for this data to get max height
                            x_range = np.linspace(min(all_periods_data['data']) - 0.5, max(all_periods_data['data']) + 0.5, 1000)
                            kde = stats.gaussian_kde(all_periods_data['data'])
                            max_kde_height = max(kde(x_range))
                            
                            # Use the assigned mean color for this team
                            mean_color = team_mean_colors.get(team, "#808080")  # Default to gray if team not found
                            
                            fig_kde.add_trace(go.Scatter(
                                x=[mean_val, mean_val],
                                y=[0, max_kde_height],
                                mode='lines',
                                name=f"{team} - Mean",
                                line=dict(color=mean_color, width=2, dash='dot'),
                                legendgroup=team,
                                showlegend=True
                            ))
                    
                    # Update layout
                    fig_kde.update_layout(
                        title=f"Recovery Time Distributions: {match_name}",
                        xaxis_title="Recovery Time (seconds)",
                        yaxis_title="Density",
                        legend_title="Teams and Periods",
                        template="plotly_white",
                        hovermode="closest",
                        width=1100,
                        height=700,
                        legend=dict(
                            groupclick="toggleitem",  # Changed from togglegroup to toggleitem
                            tracegroupgap=10,
                            orientation="h",
                            y=-0.2,  # Adjusted position to accommodate more legend items
                            x=0.5,
                            xanchor="center"
                        )
                    )
                    
                    # Save the KDE plot
                    kde_file = os.path.join(match_folder, f"match_{match_id}_kde_plot.html")
                    pio.write_html(fig_kde, kde_file)
                    print(f"Created KDE plot for Match {match_id}")
                    
                # Process each team in the match
                for current_team in match_teams:
                    # Skip teams not in our target list
                    if current_team not in target_teams:
                        print(f"Skipping {current_team} (not in target teams list)")
                        continue
                        
                    # Get the team color
                    team_color = team_colors.get(current_team, "#808080")  # Default to gray if not found
                    
                    # Prepare combined data across all periods
                    combined_data = pd.DataFrame()
                    
                    # Find available periods with data
                    available_periods = []
                    
                    for p in numeric_periods:
                        period_data = df_main[(df_main['team'] == current_team) & (df_main['period'] == p)]
                        
                        # Only include periods with enough data
                        has_data = len(period_data) >= 3
                        
                        if has_data:
                            available_periods.append(p)
                            combined_data = pd.concat([combined_data, period_data])
                    
                    # Add "All" periods if we have data for any period
                    if len(combined_data) >= 3:
                        available_periods.append("All")
                    
                    # Sort periods (numeric first, then "All")
                    available_periods.sort(key=lambda x: float('inf') if x == "All" else x)
                    
                    if not available_periods:
                        print(f"No data available for {current_team} in any period in Match ID {match_id}")
                        continue
                    
                    # Create team-specific KDE comparing all periods
                    if current_team in team_period_data and len(team_period_data[current_team]) > 0:
                        fig_team_kde = go.Figure()
                        
                        # First add histograms (with lower opacity)
                        for period, data in team_period_data[current_team].items():
                            period_label = "All Periods" if period == "All" else f"Period {period}"
                            
                            # Add histogram with low opacity
                            fig_team_kde.add_trace(go.Histogram(
                                x=data,
                                histnorm='probability density',
                                name=f'Histogram: {period_label}',
                                marker=dict(
                                    color=team_color, 
                                    opacity=0.2,
                                    line=dict(color=team_color, width=1)
                                ),
                                legendgroup=f'period_{period}',
                                legendgrouptitle=dict(text=period_label),
                                showlegend=(period == "All")  # Only show histogram for "All" to reduce legend clutter
                            ))
                        
                        # Then add KDE curves
                        for period, data in team_period_data[current_team].items():
                            # Create KDE
                            x_range = np.linspace(min(data) - 0.5, max(data) + 0.5, 1000)
                            kde = stats.gaussian_kde(data)
                            
                            period_label = "All Periods" if period == "All" else f"Period {period}"
                            
                            # Use solid line for "All Periods", dashed for individual periods
                            dash_style = 'solid' if period == "All" else 'dash'
                            line_width = 3 if period == "All" else 2
                            
                            fig_team_kde.add_trace(go.Scatter(
                                x=x_range,
                                y=kde(x_range),
                                mode='lines',
                                name=f'KDE: {period_label}',
                                line=dict(color=team_color, width=line_width, dash=dash_style),
                                legendgroup=f'period_{period}'
                            ))
                            
                            # Add scatter points for actual data
                            # Create y-positions for scatter that follow the KDE curve
                            scatter_y = []
                            for t in data:
                                # Find closest point in x_range
                                idx = np.abs(x_range - t).argmin()
                                # Get corresponding y-value from KDE and adjust with a small random component for visibility
                                y_val = kde(x_range)[idx] * np.random.uniform(0.05, 0.15)  
                                scatter_y.append(y_val)
                            
                            fig_team_kde.add_trace(go.Scatter(
                                x=data,
                                y=scatter_y,
                                mode='markers',
                                name=f'Data Points: {period_label}',
                                marker=dict(
                                    color=team_color, 
                                    size=7, 
                                    symbol='circle-open',
                                    line=dict(width=1, color=team_color)
                                ),
                                legendgroup=f'period_{period}',
                                showlegend=(period == "All")  # Only show for "All" to reduce legend clutter
                            ))
                            
                            # Add normal distribution overlay for comparison
                            mean_val = np.mean(data)
                            std_dev = np.std(data)
                            normal_y = stats.norm.pdf(x_range, mean_val, std_dev)
                            
                            # Scale normal distribution to match KDE height
                            scale_factor = max(kde(x_range)) / max(normal_y)
                            
                            fig_team_kde.add_trace(go.Scatter(
                                x=x_range,
                                y=normal_y * scale_factor,
                                mode='lines',
                                name=f'Normal Dist: {period_label}',
                                line=dict(color=team_color, width=1, dash='dot'),
                                legendgroup=f'period_{period}'
                            ))
                            
                            # Add vertical line at mean for main periods only
                            if period == "All":  # Only add vertical lines for All Periods to avoid clutter
                                fig_team_kde.add_trace(go.Scatter(
                                    x=[mean_val, mean_val],
                                    y=[0, max(kde(x_range))],
                                    mode='lines',
                                    name=f'Mean: {mean_val:.2f}s',
                                    line=dict(color='green', dash='dot', width=2),
                                    legendgroup='stats',
                                    legendgrouptitle=dict(text="Statistics")
                                ))
                                
                                # Add vertical line at median
                                median_val = np.median(data)
                                fig_team_kde.add_trace(go.Scatter(
                                    x=[median_val, median_val],
                                    y=[0, max(kde(x_range))],
                                    mode='lines',
                                    name=f'Median: {median_val:.2f}s',
                                    line=dict(color='blue', dash='dot', width=2),
                                    legendgroup='stats'
                                ))
                        
                        # Update layout
                        fig_team_kde.update_layout(
                            title=f"Recovery Time Distributions: {current_team} ({match_name})",
                            xaxis_title="Recovery Time (seconds)",
                            yaxis_title="Density",
                            legend_title="Period",
                            template="plotly_white",
                            width=1000,
                            height=600,
                            legend=dict(
                                groupclick="togglegroup",
                                tracegroupgap=10,
                                orientation="h",
                                y=-0.2,
                                x=0.5,
                                xanchor="center"
                            )
                        )
                        
                        # Save the team KDE plot
                        team_kde_file = os.path.join(
                            match_folder, 
                            f"{current_team.replace(' ', '_').lower()}_kde_plot.html"
                        )
                        pio.write_html(fig_team_kde, team_kde_file)
                        print(f"Created KDE plot for {current_team} in Match {match_id}")
                        
                    # Process each period
                    for period in available_periods:
                        # Initialize period results
                        if period not in normality_results[match_id]['periods']:
                            normality_results[match_id]['periods'][period] = {}
                        
                        # Get data for this period
                        if period == "All":
                            team_data = combined_data
                        else:
                            team_data = df_main[(df_main['team'] == current_team) & (df_main['period'] == period)]
                        
                        # Skip if no data for this period
                        if len(team_data) < 3:
                            print(f"No data or insufficient data for {current_team} in Period {period}")
                            normality_results[match_id]['periods'][period][current_team] = {
                                'has_data': False,
                                'sample_size': len(team_data) if not team_data.empty else 0
                            }
                            continue
                        
                        # Get recovery time data
                        recovery_times = team_data['time_to_recover'].dropna().values
                        
                        if len(recovery_times) < 3:
                            print(f"Too few samples for {current_team} in Period {period} (n={len(recovery_times)})")
                            normality_results[match_id]['periods'][period][current_team] = {
                                'has_data': False,
                                'sample_size': len(recovery_times)
                            }
                            continue
                        
                        # Perform Shapiro-Wilk test
                        shapiro_test = stats.shapiro(recovery_times)
                        is_normal = shapiro_test.pvalue > 0.05
                        
                        # Calculate statistics
                        mean_val = np.mean(recovery_times)
                        median_val = np.median(recovery_times)
                        std_dev = np.std(recovery_times)
                        
                        # Store results
                        normality_results[match_id]['periods'][period][current_team] = {
                            'has_data': True,
                            'shapiro_stat': shapiro_test.statistic,
                            'shapiro_pvalue': shapiro_test.pvalue,
                            'is_normal': is_normal,
                            'sample_size': len(recovery_times),
                            'mean': mean_val,
                            'median': median_val,
                            'std_dev': std_dev
                        }
                        
                        # Create QQ plot data
                        qq = stats.probplot(recovery_times, dist="norm")
                        x = qq[0][0]  # x-points
                        y = qq[0][1]  # y-points
                        slope = qq[1][0]
                        intercept = qq[1][1]
                        
                        # Create combined QQ plot and KDE plot for this period and team
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=["QQ Plot", "Density Plot (KDE)"],
                            column_widths=[0.45, 0.55]  # Give more space to KDE plot
                        )
                        
                        # Add QQ plot data points
                        fig.add_trace(
                            go.Scatter(
                                x=x, 
                                y=y, 
                                mode='markers',
                                name='Data Points',
                                marker=dict(color=team_color, size=8),
                                legendgroup='qq',
                                legendgrouptitle=dict(text="QQ Plot")
                            ),
                            row=1, col=1
                        )
                        
                        # Add QQ plot reference line
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=slope * x + intercept,
                                mode='lines',
                                name='Reference Line',
                                line=dict(color='red', width=2),
                                legendgroup='qq',
                            ),
                            row=1, col=1
                        )
                        
                        # Create KDE for the second subplot
                        x_range = np.linspace(min(recovery_times) - 0.5, max(recovery_times) + 0.5, 1000)
                        kde = stats.gaussian_kde(recovery_times)
                        
                        # Add KDE curve
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=kde(x_range),
                                mode='lines',
                                name='KDE (Density)',
                                line=dict(color=team_color, width=3),
                                legendgroup='kde',
                                legendgrouptitle=dict(text="Density Plot")
                            ),
                            row=1, col=2
                        )
                        
                        # Add histogram of actual data points under the KDE
                        fig.add_trace(
                            go.Histogram(
                                x=recovery_times,
                                histnorm='probability density',
                                name='Data Histogram',
                                marker=dict(color=team_color, opacity=0.3),
                                legendgroup='kde',
                                showlegend=True
                            ),
                            row=1, col=2
                        )
                        
                        # Add normal distribution curve for comparison
                        normal_y = stats.norm.pdf(x_range, mean_val, std_dev)
                        
                        # Scale normal distribution to match KDE height
                        scale_factor = max(kde(x_range)) / max(normal_y)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=normal_y * scale_factor,
                                mode='lines',
                                name='Normal Distribution',
                                line=dict(color='black', dash='dash', width=2),
                                legendgroup='kde',
                            ),
                            row=1, col=2
                        )
                        
                        # Add vertical line at mean
                        fig.add_trace(
                            go.Scatter(
                                x=[mean_val, mean_val],
                                y=[0, max(kde(x_range))],
                                mode='lines',
                                name=f'Mean: {mean_val:.2f}s',
                                line=dict(color='green', dash='dot', width=2),
                                legendgroup='stats',
                                legendgrouptitle=dict(text="Statistics")
                            ),
                            row=1, col=2
                        )
                        
                        # Add vertical line at median
                        fig.add_trace(
                            go.Scatter(
                                x=[median_val, median_val],
                                y=[0, max(kde(x_range))],
                                mode='lines',
                                name=f'Median: {median_val:.2f}s',
                                line=dict(color='blue', dash='dot', width=2),
                                legendgroup='stats',
                            ),
                            row=1, col=2
                        )
                        
                        # Add data points as a scatter plot on top of KDE
                        # Create y-positions for scatter plot that follow the KDE curve
                        scatter_y = []
                        for t in recovery_times:
                            # Find closest point in x_range
                            idx = np.abs(x_range - t).argmin()
                            # Get corresponding y-value from KDE and adjust with a small random component for visibility
                            y_val = kde(x_range)[idx] * np.random.uniform(0.1, 0.3)  
                            scatter_y.append(y_val)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=recovery_times,
                                y=scatter_y,
                                mode='markers',
                                name='Individual Data Points',
                                marker=dict(color=team_color, size=9, symbol='circle-open', line=dict(width=2, color=team_color)),
                                legendgroup='kde',
                            ),
                            row=1, col=2
                        )
                        
                        # Update layout
                        period_label = "All Periods" if period == "All" else f"Period {period}"
                        fig.update_layout(
                            title=f"Normality Analysis: {current_team} - {period_label} (p-value: {shapiro_test.pvalue:.4f})",
                            showlegend=True,
                            height=600,
                            width=1100,
                            template="plotly_white",
                            legend=dict(
                                groupclick="toggleitem",
                                tracegroupgap=5,
                                orientation="h",
                                y=-0.2,
                                x=0.5,
                                xanchor="center"
                            )
                        )
                        
                        # Update subplot axes
                        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
                        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
                        fig.update_xaxes(title_text="Recovery Time (seconds)", row=1, col=2)
                        fig.update_yaxes(title_text="Density", row=1, col=2)
                        
                        # Save the individual plot
                        period_str = "all_periods" if period == "All" else f"period_{period}"
                        plot_file = os.path.join(
                            match_folder, 
                            f"{current_team.replace(' ', '_').lower()}_{period_str}_qq_kde_plot.html"
                        )
                        pio.write_html(fig, plot_file)
                        
                        # Determine status based on normality and sample size
                        status = ""
                        if len(recovery_times) < 8:
                            status = "warning"  # Small sample size (yellow)
                        elif is_normal:
                            status = "normal"   # Normal distribution (green)
                        else:
                            status = "not-normal"  # Not normal (red)
                        
                        # Get opponent team
                        other_teams = [t for t in match_teams if t != current_team]
                        opponent = other_teams[0] if other_teams else "Unknown"
                        
                        # Store all result data for the filtering feature
                        results_data.append({
                            'match_id': match_id,
                            'match_name': match_name,
                            'team': current_team,
                            'team_color': team_color,
                            'opponent': opponent,
                            'period': "All" if period == "All" else period,
                            'sample_size': len(recovery_times),
                            'shapiro_pvalue': shapiro_test.pvalue,
                            'is_normal': is_normal,
                            'status': status,
                            'mean': mean_val,
                            'median': median_val,
                            'std_dev': std_dev,
                            'plot_path': plot_file.replace(output_dir + os.sep, ''),
                            'kde_match_path': f"match_{match_id}/match_{match_id}_kde_plot.html",
                            'match_plots_path': f"match_{match_id}/match_{match_id}_all_periods.html",
                            'match_stats_path': f"match_{match_id}/{current_team.replace(' ', '_').lower()}_analysis.html" 
                        })
                
            except Exception as e:
                print(f"Error processing Match ID {match_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Create final HTML with sequential filtering
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recovery Time Normality Test Results</title>
        <style>
        :root {
            --primary: #1e88e5;
            --secondary: #26a69a;
            --success: #43a047;
            --danger: #e53935;
            --warning: #ffb300;
            --light: #f5f5f5;
            --dark: #2c3e50;
            --background: #ffffff;
        }
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: var(--dark);
                background-color: var(--light);
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: var(--background);
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                border-radius: 8px;
                padding: 25px;
            }
            button, .view-btn {
                background-color: var(--primary);
                transition: all 0.2s;
            }
            button:hover, .view-btn:hover {
                background-color: #1565c0;
                transform: translateY(-2px);
            }
            .normal { background-color: var(--success); }
            .not-normal { background-color: var(--danger); }
            .warning { background-color: var(--warning); }
            h1, h2, h3 {
                color: #2c3e50;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            tbody tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tbody tr:hover {
                background-color: #f1f1f1;
            }
            .team-color {
                display: inline-block;
                width: 15px;
                height: 15px;
                margin-right: 5px;
                border-radius: 50%;
                vertical-align: middle;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                margin-right: 5px;
                border-radius: 50%;
            }


            label {
                display: inline-block;
                width: 80px;
                font-weight: bold;
            }
            .filters {
                background-color: #f8f9fa;
                padding: 20px;
                margin-bottom: 25px;
                border-radius: 8px;
                border: none;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .filter-group {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            select {
                padding: 8px 12px;
                border-radius: 6px;
                border: 1px solid #ddd;
                min-width: 220px;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
                transition: border-color 0.2s;
            }
            select:focus {
                border-color: var(--primary);
                outline: none;
            }
            #reset-filters {
                padding: 8px 18px;
                font-weight: 600;
                border-radius: 6px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            .legend {
                display: flex;
                margin: 15px 0;
                flex-wrap: wrap;
            }
            .legend-item {
                display: flex;
                align-items: center;
                margin-right: 20px;
                margin-bottom: 5px;
            }
            .disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .step-indicator {
                display: inline-block;
                background-color: #ccc;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                text-align: center;
                line-height: 20px;
                margin-right: 5px;
            }
            .current-step {
                background-color: #4CAF50;
            }
            .btn-group {
                margin-top: 10px;
            }
            .btn-group button {
                margin-right: 10px;
            }
            .button-legend {
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                border: none;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .button-info {
                display: flex;
                align-items: flex-start;
                margin-bottom: 15px;
            }
            .button-sample {
                background-color: var(--primary);
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                margin-right: 20px;
                font-size: 12px;
                min-width: 90px;
                text-align: center;
                font-weight: 600;
            }
            .button-description {
                font-size: 14px;
                color: #555;
                line-height: 1.5;
            }
            .dashboard-header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            .logo {
                font-size: 32px;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                font-size: 16px;
                margin-top: 5px;
            }
            .stats-cards {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                display: flex;
                align-items: center;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .stat-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            }
            .stat-icon {
                font-size: 28px;
                margin-right: 15px;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: var(--primary);
            }
            .stat-label {
                color: #666;
                font-size: 14px;
            }
            #reset-filters, 
            .reset-filters, 
            button[id="reset-filters"], 
            button.reset-filters {
                background-color: var(--primary, #1e88e5);
                color: white;
                border: none;
                transition: all 0.2s;
            }

            #reset-filters:hover, 
            .reset-filters:hover, 
            button[id="reset-filters"]:hover, 
            button.reset-filters:hover {
                background-color: #1565c0;
                transform: translateY(-2px);
            }
            .step-indicator {
                display: inline-block;
                background-color: #ccc;  /* Inactive color remains gray */
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                text-align: center;
                line-height: 20px;
                margin-right: 5px;
            }

            /* Change active indicator from green to blue */
            .current-step {
                background-color: var(--primary, #1e88e5);  /* Use blue instead of #4CAF50 */
            }

            /* Optional: add a subtle hover effect */
            .step-indicator:hover {
                opacity: 0.9;
}
/* Enhanced Table Styling */
table#results-table {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    margin-top: 25px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    font-size: 14px;
}

#results-table th {
    background-color: var(--primary, #1e88e5);
    color: white;
    position: sticky;
    top: 0;
    z-index: 10;
    padding: 14px 16px;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 13px;
    letter-spacing: 0.5px;
    text-align: left;
    border-bottom: 2px solid rgba(0,0,0,0.05);
}

#results-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #eee;
    vertical-align: middle;
}

#results-table tbody tr:nth-child(even) {
    background-color: rgba(0,0,0,0.02);
}

#results-table tbody tr:hover {
    background-color: rgba(30, 136, 229, 0.05);
}

/* Keep specific styling for the status indicators */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-right: 8px;
    border-radius: 50%;
    vertical-align: middle;
}

/* Team color indicator enhancement */
.team-color {
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 8px;
    border-radius: 50%;
    vertical-align: middle;
    box-shadow: 0 0 4px rgba(0,0,0,0.2);
}

/* View Plots column - preserve arrangement but enhance buttons */
#results-table td:last-child {
    text-align: right;
    padding-right: 12px;
}

/* Enhance the buttons without changing arrangement */
.btn-group {
    display: inline-flex;
    gap: 6px;
}

.view-btn {
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 4px;
    border: none;
    background-color: var(--primary, #1e88e5);
    color: white;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    text-align: center;
}

.view-btn:hover {
    background-color: #1565c0;
    transform: translateY(-1px);
    box-shadow: 0 2px 5px rgba(0,0,0,0.15);
}
/* Help panel styling */
.help-container {
  position: relative;
  margin: 20px 0;
}

.help-button {
  background-color: var(--primary, #1e88e5);
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.help-button:hover {
  background-color: #1565c0;
  transform: translateY(-1px);
}

.help-icon {
  font-size: 16px;
}

.help-panel {
  position: absolute;
  top: 100%;
  right: 0;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
  width: 600px;
  max-width: 90vw;
  z-index: 100;
  max-height: 80vh;
  overflow-y: auto;
}

.help-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  border-bottom: 1px solid #eee;
}

.help-header h2 {
  margin: 0;
  font-size: 20px;
  color: var(--primary, #1e88e5);
}

.close-button,
.close-button:focus,
.close-button:hover,
.close-button:active {
  outline: none;
  box-shadow: none;
  background: transparent;   
  color: #666;               
}

.accordion {
  padding: 15px;
}

.accordion-item {
  margin-bottom: 10px;
  border: 1px solid #eee;
  border-radius: 6px;
  overflow: hidden;
}

.accordion-header {
  background-color: #f8f9fa;
  padding: 12px 15px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: background-color 0.2s;
}

.accordion-header:hover {
  background-color: #f1f3f5;
}

.accordion-icon {
  font-size: 18px;
}

.accordion-content {
  padding: 15px;
  background-color: white;
}

.help-list {
  padding-left: 20px;
  margin: 10px 0;
}

.help-list li {
  margin-bottom: 8px;
  line-height: 1.5;
}

.help-note {
  background-color: #f8f9fa;
  padding: 10px;
  border-radius: 4px;
  border-left: 4px solid var(--primary, #1e88e5);
  margin: 15px 0 5px 0;
}

/* Team Analysis Navigation */
.team-analytics-navigation {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.team-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 15px;
}
.team-btn {
    background-color: var(--primary);
    color: white;
    padding: 10px 20px;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
    min-width: 180px;
    text-align: center;
}
.team-btn:hover {
    background-color: #1565c0;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
        </style>
    </head>
    <body>
        <div class="container">
            <header class="dashboard-header">
                <div class="logo">⚽📊</div>
                <h1>Football Recovery Analytics Dashboard</h1>
                <div class="subtitle">Performance analysis of recovery times across matches and periods</div>
            </header>
            <!-- Add this where you want the help button to appear (perhaps in the header or near the filters) -->
<div class="help-container">
  <button id="toggleHelp" class="help-button">
    <i class="help-icon">❓</i> Help & Documentation
  </button>
  
  <div id="helpPanel" class="help-panel" style="display: none;">
    <div class="help-header">
      <h2>Analysis Documentation</h2>
      <button id="closeHelp" class="close-button">×</button>
    </div>
    
    <div class="accordion">
      <div class="accordion-item">
        <div class="accordion-header">
          <span class="accordion-icon">📊</span> Interpretation
        </div>
        <div class="accordion-content">
          <ul class="help-list">
            <li><strong>Shapiro-Wilk p-value > 0.05:</strong> Data can be considered normally distributed</li>
            <li><strong>Shapiro-Wilk p-value <= 0.05:</strong> Data significantly deviates from normal distribution</li>
            <li><strong>Warning:</strong> Results for small sample sizes (n < 8) should be interpreted with caution</li>
          </ul>
          <p class="help-note"><strong>Note:</strong> The "All Periods" analysis combines data across all periods to provide a more comprehensive assessment with larger sample sizes.</p>
        </div>
      </div>
      
      <div class="accordion-item">
        <div class="accordion-header">
          <span class="accordion-icon">📈</span> QQ Plots
        </div>
        <div class="accordion-content">
          <p>QQ plots compare the observed data distribution to the theoretical normal distribution:</p>
          <ul class="help-list">
            <li>Points following the red reference line closely suggest a normal distribution</li>
            <li>Systematic deviations from the line suggest non-normality</li>
            <li>S-shaped patterns indicate skewness in the data</li>
          </ul>
        </div>
      </div>
      
      <div class="accordion-item">
        <div class="accordion-header">
          <span class="accordion-icon">🔍</span> KDE Plots
        </div>
        <div class="accordion-content">
          <p>Kernel Density Estimation (KDE) plots show the estimated probability density of recovery times:</p>
          <ul class="help-list">
            <li>Bell-shaped curves suggest normal distributions</li>
            <li>Multi-modal curves (multiple peaks) suggest mixed distributions</li>
            <li>Skewed curves (asymmetric) indicate non-normal distributions</li>
            <li>Dotted line shows theoretical normal distribution for comparison</li>
            <li>Green vertical line shows mean, blue vertical line shows median</li>
          </ul>
        </div>
      </div>
      
      
      <div class="accordion-item">
        <div class="accordion-header">
          <span class="accordion-icon">⚽</span> Match Events
        </div>
        <div class="accordion-content">
          <p>These plots show match recovery time trends alongside match events:</p>
          <ul class="help-list">
            <li>Interactive timeline with team recovery times</li>
            <li>Match events (goals, cards, substitutions) overlaid on recovery data</li>
            <li>Team formation details and tactical shifts</li>
            <li>Toggle between different periods or view all periods</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>
            <div class="filters">
                <h3>Filter Results</h3>
                <div class="filter-group">
                    <span class="step-indicator current-step">1</span>
                    <label for="team-filter">Team:</label>
                    <select id="team-filter">
                        <option value="">-- Select a Team --</option>

            
    """
    
    # Add team options
    for team_name in sorted(target_teams):
        html_content += f'                        <option value="{team_name}">{team_name}</option>\n'
    
    html_content += """
                    </select>
                </div>
                <div class="filter-group">
                    <span class="step-indicator">2</span>
                    <label for="match-filter">Match:</label>
                    <select id="match-filter" disabled>
                        <option value="">-- Select a Match --</option>
                    </select>
                </div>
                <div class="filter-group">
                    <span class="step-indicator">3</span>
                    <label for="period-filter">Period:</label>
                    <select id="period-filter" disabled>
                        <option value="all">All Periods</option>
                        <option value="1">Period 1</option>
                        <option value="2">Period 2</option>
                        <option value="3">Period 3</option>
                        <option value="4">Period 4</option>
                        <option value="All">Combined (All Periods)</option>
                    </select>
                </div>
                <button id="reset-filters">Reset Filters</button>
            </div>
            <div class="team-analytics-navigation">
                <h3>Team Analysis Reports</h3>
                <p>Access detailed analytics reports for each team:</p>
                <div class="team-buttons">
"""

    # Add buttons for each team
    for team_name in sorted(target_teams):
        team_slug = team_name.replace(' ', '_').lower()
        html_content += f"""
                    <button onclick="window.open('team_analysis/{team_slug}_analysis.html', '_blank')" class="view-btn team-btn">
                        {team_name} Analysis
                    </button>
        """

    html_content += """
                </div>
            </div>
            <div class="button-legend">
                <h4>Visualization Buttons</h4>
                <div class="button-info">
                    <span class="button-sample">QQ+KDE</span>
                    <span class="button-description">Shows QQ plots and density plots for assessing normality of team recovery times for a specific period</span>
                </div>
                <div class="button-info">
                    <span class="button-sample">Match Events</span>
                    <span class="button-description">Shows interactive timeline plot and table with match events, team formations, and recovery time plots</span>
                </div>
                <div class="button-info">
                    <span class="button-sample">Match Stats</span>
                    <span class="button-description">Shows detailed statistical analysis for the team, including event impact analysis, normality tests, and significance of recovery time changes</span>
                </div>
            </div>
            <div class="stats-cards">
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-content">
            <div class="stat-value" id="total-records">0</div>
            <div class="stat-label">Total Records</div>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">✅</div>
        <div class="stat-content">
            <div class="stat-value" id="normal-count">0</div>
            <div class="stat-label">Normal Distributions</div>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⚠️</div>
        <div class="stat-content">
            <div class="stat-value" id="warning-count">0</div>
            <div class="stat-label">Small Samples</div>
        </div>
    </div>
</div>
            <div class="legend">
                <div class="legend-item">
                    <span class="status-indicator normal"></span> Normal Distribution (p > 0.05)
                </div>
                <div class="legend-item">
                    <span class="status-indicator not-normal"></span> Not Normal (p <= 0.05)
                </div>
                <div class="legend-item">
                    <span class="status-indicator warning"></span> Small Sample (n < 8)
                </div>
            </div>
            
            <table id="results-table">
                <thead>
                    <tr>
                        <th>Match</th>
                        <th>Period</th>
                        <th>Team</th>
                        <th>Sample Size</th>
                        <th>Shapiro-Wilk p-value</th>
                        <th>Is Normal?</th>
                        <th>Mean (sec)</th>
                        <th>Median (sec)</th>
                        <th>Std Dev (sec)</th>
                        <th>View Plots and Tables</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add rows for each result
    for result in results_data:
        # Create the team slug for the file path
        team_slug = result['team'].replace(' ', '_').lower()
        # Update the match_stats_path to point to the team-specific analysis HTML
        stats_path = f"match_{result['match_id']}/{team_slug}_analysis.html"
        
        html_content += f"""
                        <tr data-match="{result['match_id']}" data-team="{result['team']}" data-period="{result['period']}">
                            <td>{result['match_name']}</td>
                            <td>{result['period']}</td>
                            <td>
                                <span class="team-color" style="background-color: {result['team_color']};"></span>
                                {result['team']}
                            </td>
                            <td>{result['sample_size']}</td>
                            <td>{result['shapiro_pvalue']:.4f}</td>
                            <td>
                                <span class="status-indicator {result['status']}"></span>
                                {"Yes" if result['is_normal'] else "No"}{" (small sample)" if result['sample_size'] < 8 else ""}
                            </td>
                            <td>{result['mean']:.2f}</td>
                            <td>{result['median']:.2f}</td>
                            <td>{result['std_dev']:.2f}</td>
                            <td>
                                <div class="btn-group">
                                    <button onclick="window.open('{result['plot_path']}', '_blank')" class="view-btn">QQ+KDE</button>
                                    <button onclick="window.open('{result['match_plots_path']}', '_blank')" class="view-btn">Match Events</button>
                                    <button onclick="window.open('{stats_path}', '_blank')" class="view-btn" title="View detailed team analytics">Match Stats</button>
                                </div>
                            </td>
                        </tr>
            """


    html_content += """
                </tbody>
            </table>
            
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Get filter elements
                const teamFilter = document.getElementById('team-filter');
                const matchFilter = document.getElementById('match-filter');
                const periodFilter = document.getElementById('period-filter');
                const resetButton = document.getElementById('reset-filters');
                const table = document.getElementById('results-table');
                const rows = table.querySelectorAll('tbody tr');
                const stepIndicators = document.querySelectorAll('.step-indicator');
                
                // Store team-match relationships
                const teamMatches = JSON.parse('""" + json.dumps(team_matches) + """');
                const matchData = JSON.parse('""" + json.dumps(match_data) + """');
                // Help panel functionality
                const toggleHelpBtn = document.getElementById('toggleHelp');
                const closeHelpBtn = document.getElementById('closeHelp');
                const helpPanel = document.getElementById('helpPanel');


                toggleHelpBtn.addEventListener('click', function() {
                    helpPanel.style.display = helpPanel.style.display === 'none' ? 'block' : 'none';
                });
                
                closeHelpBtn.addEventListener('click', function() {
                    helpPanel.style.display = 'none';
                });

                
                // Team filter change event
                teamFilter.addEventListener('change', function() {
                    // Reset and disable subsequent filters
                    resetMatchFilter();
                    resetPeriodFilter();
                    
                    const selectedTeam = this.value;
                    
                    if (!selectedTeam) {
                        // Reset all to initial state if no team selected
                        resetAllFilters(false);
                        return;
                    }
                    
                    // Populate match filter with matches for this team
                    const matches = teamMatches[selectedTeam] || [];
                    
                    // Enable match filter
                    matchFilter.disabled = false;
                    matchFilter.innerHTML = '<option value="">-- Select a Match --</option>';
                    
                    // Add match options
                    matches.forEach(matchId => {
                        const match = matchData[matchId];
                        const option = document.createElement('option');
                        option.value = matchId;
                        option.textContent = match.match_name;
                        matchFilter.appendChild(option);
                    });
                    
                    // Update step indicators
                    stepIndicators[0].classList.remove('current-step');
                    stepIndicators[1].classList.add('current-step');
                    
                    // Filter table to show only rows with the selected team
                    filterTable();
                });
                
                // Match filter change event
                matchFilter.addEventListener('change', function() {
                    resetPeriodFilter();
                    
                    const selectedMatch = this.value;
                    
                    if (!selectedMatch) {
                        // If no match selected, only filter by team
                        periodFilter.disabled = true;
                        stepIndicators[1].classList.add('current-step');
                        stepIndicators[2].classList.remove('current-step');
                    } else {
                        // Enable period filter
                        periodFilter.disabled = false;
                        stepIndicators[1].classList.remove('current-step');
                        stepIndicators[2].classList.add('current-step');
                    }
                    
                    // Filter table
                    filterTable();
                });
                
                // Period filter change event
                periodFilter.addEventListener('change', function() {
                    // Filter table
                    filterTable();
                });
                
                // Reset button click event
                resetButton.addEventListener('click', function() {
                    resetAllFilters(true);
                });
                
                // Filter table based on current selections
                function filterTable() {
                    const selectedTeam = teamFilter.value;
                    const selectedMatch = matchFilter.value;
                    const selectedPeriod = periodFilter.value;
                    
                    rows.forEach(row => {
                        let showRow = true;
                        
                        // Apply team filter
                        if (selectedTeam && row.getAttribute('data-team') !== selectedTeam) {
                            showRow = false;
                        }
                        
                        // Apply match filter
                        if (selectedMatch && row.getAttribute('data-match') !== selectedMatch) {
                            showRow = false;
                        }
                        
                        // Apply period filter
                        if (selectedPeriod && selectedPeriod !== 'all' && row.getAttribute('data-period') !== selectedPeriod) {
                            showRow = false;
                        }
                        
                        row.style.display = showRow ? '' : 'none';
                    });
                    
                    // Update stats after filtering
                    updateStats();
                }
                
                // Reset match filter
                function resetMatchFilter() {
                    matchFilter.innerHTML = '<option value="">-- Select a Match --</option>';
                    matchFilter.disabled = true;
                }
                
                // Reset period filter
                function resetPeriodFilter() {
                    periodFilter.value = 'all';
                    periodFilter.disabled = true;
                }
                
                // Reset all filters
                function resetAllFilters(includeTeam) {
                    if (includeTeam) {
                        teamFilter.value = '';
                    }
                    
                    resetMatchFilter();
                    resetPeriodFilter();
                    
                    // Reset step indicators
                    stepIndicators.forEach((indicator, index) => {
                        indicator.classList.remove('current-step');
                        if (index === 0) {
                            indicator.classList.add('current-step');
                        }
                    });
                    
                    // Show all rows
                    rows.forEach(row => {
                        row.style.display = '';
                    });
                    
                    // Update stats after resetting
                    updateStats();
                }
                
                // Make rows clickable to open plots
                rows.forEach(row => {
                    row.style.cursor = 'pointer';
                    row.addEventListener('click', function(e) {
                        if (e.target.tagName !== 'BUTTON' && e.target.parentNode.tagName !== 'BUTTON') {
                            const qqKdeBtn = this.querySelector('.view-btn');
                            if (qqKdeBtn) {
                                window.open(qqKdeBtn.getAttribute('onclick').match(/'([^']+)'/)[1], '_blank');
                            }
                        }
                    });
                });
                
                // Function to update the stats cards
                function updateStats() {
                    const rows = document.querySelectorAll('#results-table tbody tr:not([style*="display: none"])');
                    const totalRecords = rows.length;
                    const normalCount = document.querySelectorAll('#results-table tbody tr:not([style*="display: none"]) .normal').length;
                    const warningCount = document.querySelectorAll('#results-table tbody tr:not([style*="display: none"]) .warning').length;
                    
                    document.getElementById('total-records').textContent = totalRecords;
                    document.getElementById('normal-count').textContent = normalCount;
                    document.getElementById('warning-count').textContent = warningCount;
                }
                
                // Initialize stats when page loads
                updateStats();
            });
            </script>
        </div>
    </body>
    </html>
    """
    
    # Write the final HTML file
    index_file = os.path.join(output_dir, "dashboard.html")
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nNormality test report created at: {index_file}")
    print("All normality tests completed!")
    add_explanations_to_all_plots(output_dir)
    
    return normality_results


def generate_html_files(team_dicts, all_dataframes, normality_results, mannwhitney_results, impact_results, match_level_tests, team_level_tests, output_dir="visuals"):
    """
    Generate HTML files for the analysis results
    
    Args:
        team_dicts: Dictionary of team names and their match dictionaries
        all_dataframes: Dictionary of all dataframes from analyze_event_recovery_times
        normality_results: Dictionary of normality test results from analyze_event_recovery_times
        mannwhitney_results: Dictionary of Mann-Whitney test results from analyze_event_recovery_times
        impact_results: DataFrame of impact analysis results from analyze_recovery_impact
        match_level_tests: List of match-level test results with BH correction
        team_level_tests: List of team-level test results with BH correction
        output_dir: Directory to save HTML files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create team_analysis directory
    team_dir = os.path.join(output_dir, "team_analysis")
    os.makedirs(team_dir, exist_ok=True)
    
    # Get list of teams and matches
    teams = [team_name for team_name, _ in team_dicts]
    
    # Dictionary to store match IDs and their details
    all_matches = {}
    for team_name, matches_dict in team_dicts:
        for match_id, match_details in matches_dict.items():
            if match_id not in all_matches:
                all_matches[match_id] = {
                    'teams': [team_name, match_details['team']]
                }
            
            # Create individual match directory
            match_dir = os.path.join(output_dir, f"match_{match_id}")
            os.makedirs(match_dir, exist_ok=True)
    
    # Create dictionaries to quickly look up BH correction results
    match_bh_lookup = {}
    team_bh_lookup = {}
    
    # Build lookup for match-level BH corrections
    for test in match_level_tests:
        key = f"{test['team'].replace(' ', '_').lower()}_match_{test['match_id']}_{test['event_type']}"
        match_bh_lookup[key] = {
            'bh_corrected_p': test.get('bh_corrected_p', 'N/A'),
            'significant_after_bh': test.get('significant_after_bh', False)
        }
    
    # Build lookup for team-level BH corrections
    for test in team_level_tests:
        key = f"{test['team'].replace(' ', '_').lower()}_{test['event_type']}"
        team_bh_lookup[key] = {
            'bh_corrected_p': test.get('bh_corrected_p', 'N/A'),
            'significant_after_bh': test.get('significant_after_bh', False)
        }
    
    # Define CSS styles
    css_styles = """
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
        }
        h1 {
            color: #1d3557;
            border-bottom: 2px solid #457b9d;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        h2 {
            color: #1d3557;
            margin-top: 25px;
        }
        h3 {
            color: #457b9d;
            margin-top: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            table-layout: auto;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
            white-space: nowrap;
        }
        th {
            background-color: #457b9d;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .positive {
            color: #2a9d8f;
            font-weight: bold;
        }
        .negative {
            color: #e63946;
            font-weight: bold;
        }
        .highlight {
            background-color: #ffffd9;
        }
        .highlight-bh {
            background-color: #e8f5e9;
        }
        .summary-box {
            background-color: #e1f3f8;
            border-left: 5px solid #457b9d;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }
        .event-card {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 0.9em;
            color: #777;
        }
        .chart-container {
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            height: 400px;
            position: relative;
        }
        /* Help button styles */
        .help-button {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background-color: #457b9d;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .help-button:hover {
            background-color: #1d3557;
        }
        .help-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0,0,0,0.5);
            z-index: 1001;
            align-items: center;
            justify-content: center;
        }
        .help-content {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .help-content h2 {
            color: #1d3557;
            margin-top: 0;
        }
        .help-close {
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 24px;
            cursor: pointer;
            color: #666;
        }
        .help-close:hover {
            outline: none;
            box-shadow: none;
            background: transparent;
            color: #666;
        }
        .term-definition {
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        .term-definition h3 {
            margin-bottom: 5px;
        }
        .team-flag {
            font-size: 24px;
            margin-right: 8px;
            vertical-align: middle;
        }
        /* Remove table wrapper constraints */
        .table-wrapper {
            width: 100%;
            overflow-x: auto;
        }
        /* Make tables responsive but use full width */
        @media screen and (max-width: 1200px) {
            .table-wrapper {
                overflow-x: scroll;
            }
        }
        .no-data {
            color: #999;
            font-style: italic;
            background-color: #f9f9f9;
        }
    </style>
    """
    
    # JavaScript for help button and modal
    help_js = """
    <script>
        // Wait for the DOM to be fully loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Get modal elements
            var helpModal = document.getElementById('helpModal');
            var helpBtn = document.getElementById('helpBtn');
            var closeBtn = document.getElementById('closeHelp');
            
            // Open modal when help button is clicked
            helpBtn.onclick = function() {
                helpModal.style.display = 'flex';
            }
            
            // Close modal when close button is clicked
            closeBtn.onclick = function() {
                helpModal.style.display = 'none';
            }
            
            // Close modal when clicking outside the content
            window.onclick = function(event) {
                if (event.target == helpModal) {
                    helpModal.style.display = 'none';
                }
            }
        });
    </script>
    """
    
    # Helper function to format event type names for display
    def format_event_type(event_type):
        """Format event type for display, with special handling for tactical shifts"""
        if event_type in ("tactical_shifts", "shifts"):
            return "Tactical Shifts"
        else:
            return event_type.replace('_', ' ').title()
            
    # Help button and modal HTML with updated BH correction information
    def create_help_button_and_modal():
        help_html = """
        <!-- Help Button -->
        <button id="helpBtn" class="help-button">?</button>
        
        <!-- Help Modal -->
        <div id="helpModal" class="help-modal">
            <div class="help-content">
                <span id="closeHelp" class="help-close">&times;</span>
                <h2>Football Analysis Guide</h2>
                <p>This guide explains the statistical terms and analysis used in this dashboard.</p>
                
                <div class="term-definition">
                    <h3>TRB (Time to Recover Ball)</h3>
                    <p>The time (in seconds) it takes for a team to regain possession after losing the ball. Lower times indicate better defensive pressing or recovery.</p>
                </div>
                
                <div class="term-definition">
                    <h3>Event Types</h3>
                    <p><strong>All Goals:</strong> Includes goals scored by either team</p>
                    <p><strong>Substitutions:</strong> Player replacements during the match</p>
                    <p><strong>Tactical Shifts:</strong> Changes in team formation or strategy</p>
                    <p><strong>Injuries:</strong> Stoppages due to player injuries</p>
                    <p><strong>Yellow/Red Cards:</strong> Disciplinary actions by the referee</p>
                </div>
                
                <div class="term-definition">
                    <h3>Shapiro-Wilk Test</h3>
                    <p>A statistical test to check if data follows a normal distribution (bell curve).</p>
                    <p><strong>p-value:</strong> If this number is less than 0.05, the data is NOT normally distributed.</p>
                    <p><strong>What it means for coaching:</strong> Non-normal data suggests there are inconsistencies in recovery times - some very fast recoveries and some very slow ones, rather than most being around the average.</p>
                </div>
                
                <div class="term-definition">
                    <h3>Mann-Whitney U Test</h3>
                    <p>A statistical test that compares recovery times before and after events to see if there's a significant difference.</p>
                    <p><strong>p-value:</strong> If this number is less than 0.05, there IS a significant difference in recovery times before vs. after the event.</p>
                    <p><strong>BH Corrected p-value:</strong> When multiple tests are performed, the Benjamini-Hochberg (BH) correction adjusts p-values to control the false discovery rate. This helps avoid finding false significant results.</p>
                    <p><strong>Significant After BH:</strong> Shows whether the result remains significant after applying the BH correction. This is a more conservative and reliable indicator of true effects.</p>
                    <p><strong>Cliff's Delta:</strong> Measures the size of the effect:</p>
                    <ul>
                        <li>Negligible: Less than 0.147</li>
                        <li>Small: 0.147 to 0.33</li>
                        <li>Medium: 0.33 to 0.474</li>
                        <li>Large: Greater than 0.474</li>
                    </ul>
                    <p><strong>Direction of Cliff's Delta:</strong></p>
                    <ul>
                        <li><strong>Positive values (+):</strong> Recovery times BEFORE the event are generally higher than after the event. This means players take LESS time to recover the ball following the event (faster recovery).</li>
                        <li><strong>Negative values (-):</strong> Recovery times BEFORE the event are generally lower than after the event. This means players take MORE time to recover the ball following the event (slower recovery).</li>
                    </ul>
                    <p><strong>Confidence Interval (CI):</strong></p>
                    <ul>
                        <li>The CI Lower and CI Upper values show the 95% confidence interval for Cliff's Delta.</li>
                        <li>This means we're 95% confident the true effect size is somewhere between these two values.</li>
                        <li><strong>For Coaches:</strong> If the CI includes zero (e.g., -0.1 to +0.2), the effect might not be reliable. If it excludes zero (e.g., +0.1 to +0.3), you can be more confident in the effect's direction.</li>
                        <li>Wider intervals indicate less certainty about the exact effect size.</li>
                        <li>Narrower intervals indicate more reliable estimates of the true effect.</li>
                    </ul>
                    <p><strong>What it means for coaching:</strong> Focus on results that are significant after BH correction, as these are the most reliable findings. These indicate events that truly affect your team's recovery patterns.</p>
                </div>
                
                <div class="term-definition">
                    <h3>Recovery Impact Analysis</h3>
                    <p>Shows whether teams recover the ball faster or slower after specific events.</p>
                    <p><strong>Avg TRB Before/After:</strong> Average recovery time in seconds before and after events</p>
                    <p><strong>Faster After Event:</strong> "Yes" means recovery was quicker after the event</p>
                    <p><strong>Significant Impact:</strong> "Yes" means the difference is statistically meaningful</p>
                    <p><strong>What it means for coaching:</strong> Focus on events with significant impacts. If recovery is faster after goals, your team might be more motivated. If slower after cards, you may need to adjust tactics after disciplinary actions.</p>
                </div>
                
                <div class="term-definition">
                    <h3>Highlighted Rows</h3>
                    <p>Yellow highlighted rows indicate statistically significant findings that deserve special attention from coaching staff.</p>
                </div>
                
                <div class="term-definition">
                    <h3>How To Use This Analysis</h3>
                    <p><strong>For Training:</strong> Focus on improving recovery times after events where your team slows down</p>
                    <p><strong>For Tactics:</strong> Consider formation changes after events that negatively impact recovery</p>
                    <p><strong>For Psychology:</strong> Address mental aspects if certain events (like conceding goals) consistently slow recovery</p>
                    <p><strong>For Opposition Analysis:</strong> Look for patterns in how opponents' recovery times change after events</p>
                </div>
            </div>
        </div>
        """
        return help_html
        
    # Helper function to get significant findings for a team
    def get_significant_findings(team_name):
        findings = []
        
        # Filter impact results for this team
        if not impact_results.empty:
            team_impacts = impact_results[impact_results['Team'] == team_name]
            significant_impacts = team_impacts[team_impacts['Significant Impact'] == 'Yes']
            
            for _, row in significant_impacts.iterrows():
                event_type = row['Event Type']
                faster = row['Faster After Event'] == 'Yes'
                effect = "faster" if faster else "slower"
                
                findings.append(f"<strong>{event_type}:</strong> Recovery times are significantly {effect} after {event_type.lower()}.")
        
        return findings
    
    # Generate team-level HTML files
    for team_name in teams:
        team_slug = team_name.replace(' ', '_').lower()
        flag_emoji = get_flag_emoji(team_name)
        
        # Debug: Print available dataframes for this team
        print(f"\nGenerating HTML for {team_name}...")
        team_dataframes = [df for df in mannwhitney_results.keys() if team_slug in df and "match_" not in df]
        print(f"Found {len(team_dataframes)} team-level dataframes for {team_name}: {team_dataframes}")
        
        # Get significant findings for this team
        significant_findings = get_significant_findings(team_name)

        # Create team level analysis HTML file
        with open(os.path.join(team_dir, f"{team_slug}_analysis.html"), "w") as f:
            f.write(f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{team_name} - Team Analysis</title>
            {css_styles}
            {help_js}
        </head>
        <body style="max-width: 100%; width: 100%;">
            <h1><span class="team-flag">{flag_emoji}</span> {team_name} - Team Analysis</h1>
                        
            <div class="summary-box">
                <h3>Key Findings</h3>
            """)
            
            if significant_findings:
                # Show the significant findings first
                f.write("<ul>\n")
                for finding in significant_findings:
                    f.write(f"    <li>{finding}</li>\n")
                f.write("</ul>\n")
                
                # Add detailed event-specific analysis
                f.write("<h4>Detailed Event Analysis (All Matches)</h4>\n")
                
                # Add team-level impact analysis details
                if not impact_results.empty:
                    team_impacts = impact_results[
                        (impact_results['Team'] == team_name) & 
                        (impact_results['Match ID'] == 'All Matches')
                    ]
                    
                    for _, row in team_impacts.iterrows():
                        event_type = row['Event Type']
                        avg_before = row['Avg TRB Before (s)']
                        avg_after = row['Avg TRB After (s)']
                        faster = row['Faster After Event'] == "Yes"
                        p_value = row['p-value']
                        significant = row['Significant Impact'] == "Yes"
                        
                        # Create event analysis
                        effect_word = "faster" if faster else "slower"
                        significance_phrase = "significantly " if significant else ""
                        effect_size = ""
                        
                        # Convert to internal event type format for consistent display
                        internal_event_type = event_type.lower().replace(' ', '_')
                        event_display = format_event_type(internal_event_type)
                        
                        # Find the effect size from Mann-Whitney results if available
                        for df_name, mw_results in mannwhitney_results.items():
                            # Find team-level results (not match-specific)
                            if "match_" not in df_name and team_slug in df_name:
                                parts = df_name.split('_')
                                if len(parts) > 2:
                                    test_event_type = '_'.join(parts[2:])
                                    if test_event_type == internal_event_type:
                                        if 'effect_size' in mw_results:
                                            effect_size = f" with a {mw_results['effect_size'].lower()} effect size (Cliff's Delta: {mw_results.get('cliffs_delta', 'N/A'):.4f})" if isinstance(mw_results.get('cliffs_delta'), (float, int)) else ""
                                        break
                        
                        # Write event analysis
                        f.write(f"""
                        <div style="margin-left: 20px; margin-bottom: 15px; padding: 10px; background-color: #f9f9f9; border-left: 3px solid {'#4CAF50' if significant else '#ddd'}; border-radius: 4px;">
                            <strong>{event_display}:</strong>
                    """)
                        
                        if significant:
                            f.write(f"""
                            Across all matches, the team has {significance_phrase}{effect_word} recovery times after {event_type.lower()}{effect_size}. 
                            Average recovery time {'decreased' if faster else 'increased'} from {avg_before} seconds before to {avg_after} seconds after {event_type.lower()}.
                            """)
                        else:
                            f.write(f"""
                            No significant overall impact found for {event_type.lower()}, though there was a {'decrease' if faster else 'increase'} 
                            in average recovery time ({avg_before} to {avg_after}).
                            """)
                        
                        f.write("</div>\n")
            else:
                f.write("    <p>No significant findings were identified for this team.</p>\n")
            
            f.write("""
                        </div>

                        <h2>Recovery Times - Team Level</h2>
                    """)
            
            f.write("""
                <h3>Normality Test Results</h3>
                <div class="table-wrapper">
                <table>
                    <tr>
                        <th>Event Type</th>
                        <th>Sample Size</th>
                        <th>Shapiro-Wilk Statistic</th>
                        <th>p-value</th>
                        <th>Normal Distribution</th>
                    </tr>
            """)
            
            # Define all event types to ensure complete table
            event_types = ['all_goals', 'substitutions', 'tactical_shifts', 'injuries', 'yellow_cards', 'red_cards']
            
            # Collect all rows first
            rows_with_data = []
            rows_without_data = []
            
            # Process each event type
            for event_type in event_types:
                found = False
                
                # Look for exact dataframe name pattern: df_[team_slug]_[event_type]
                expected_df_name = f"df_{team_slug}_{event_type}"
                
                # Find team-level normality results for this event type
                if expected_df_name in normality_results:
                    results = normality_results[expected_df_name]
                    found = True
                    
                    # Format for display
                    event_display = format_event_type(event_type)
                    
                    # Check for normality
                    is_normal = results.get('shapiro_normal', False)
                    normal_class = "positive" if is_normal else "negative"
                    normal_text = "Yes" if is_normal else "No"
                    
                    # Sample size
                    sample_size = results.get('sample_size', "N/A")
                    
                    # Test statistic
                    shapiro_stat = results.get('shapiro_stat', "N/A")
                    if isinstance(shapiro_stat, (float, int)):
                        shapiro_stat = f"{shapiro_stat:.4f}"
                        
                    # p-value
                    shapiro_p = results.get('shapiro_p', "N/A")
                    if isinstance(shapiro_p, (float, int)):
                        shapiro_p = f"{shapiro_p:.6f}"
                    
                    row_html = f"""
                <tr>
                    <td>{event_display}</td>
                    <td>{sample_size}</td>
                    <td>{shapiro_stat}</td>
                    <td>{shapiro_p}</td>
                    <td class="{normal_class}">{normal_text}</td>
                </tr>
                    """
                    rows_with_data.append(row_html)
                

            
            # Write rows with data first, then rows without data
            for row in rows_with_data:
                f.write(row)
            # for row in rows_without_data:
            #     f.write(row)
            
            f.write("""
                </table>
                </div>
            """)
            
            # 2. Mann-Whitney Test Results with BH Correction
            f.write("""
                <h3>Mann-Whitney U Test Results</h3>
                <div class="table-wrapper">
                <table>
                    <tr>
                        <th>Event Type</th>
                        <th>Before Sample Size</th>
                        <th>After Sample Size</th>
                        <th>Mann-Whitney U</th>
                        <th>p-value</th>
                        <th>Significant Difference</th>
                        <th>BH Corrected p-value</th>
                        <th>Significant After BH</th>
                        <th>Cliff's Delta</th>
                        <th>CI Lower</th>
                        <th>CI Upper</th>
                        <th>Effect Size</th>
                        <th>Explanation</th>
                    </tr>
            """)
            
            # Define all event types to ensure complete table
            event_types = ['all_goals', 'substitutions', 'tactical_shifts', 'injuries', 'yellow_cards', 'red_cards']
            
            # Collect all rows first
            rows_with_data = []
            rows_without_data = []
            
            # Process each event type
            for event_type in event_types:
                found = False
                
                # Look for exact dataframe name pattern: df_[team_slug]_[event_type]
                expected_df_name = f"df_{team_slug}_{event_type}"
                
                # Search through Mann-Whitney results
                if expected_df_name in mannwhitney_results:
                    results = mannwhitney_results[expected_df_name]
                    found = True
                    
                    # Format for display
                    event_display = format_event_type(event_type)
                    
                    # Look up BH correction results
                    lookup_key = f"{team_slug}_{event_type}"
                    bh_data = team_bh_lookup.get(lookup_key, {})
                    bh_corrected_p = bh_data.get('bh_corrected_p', 'N/A')
                    significant_after_bh = bh_data.get('significant_after_bh', False)
                    
                    # Check for significance
                    is_significant = results.get('significant', False)
                    sig_class = "positive" if is_significant else "negative"
                    sig_text = "Yes" if is_significant else "No"
                    
                    # BH significance class
                    bh_sig_class = "positive" if significant_after_bh else "negative"
                    bh_sig_text = "Yes" if significant_after_bh else "No"
                    
                    # Highlight row based on BH significance
                    row_class = 'class="highlight"' if significant_after_bh else ''
                    
                    # Sample sizes
                    before_size = results.get('before_sample_size', "N/A")
                    after_size = results.get('after_sample_size', "N/A")
                    
                    # Test statistic
                    mw_stat = results.get('mw_stat', "N/A")
                    if isinstance(mw_stat, (float, int)):
                        mw_stat = f"{mw_stat:.4f}"
                        
                    # p-value
                    mw_p = results.get('mw_p', "N/A")
                    if isinstance(mw_p, (float, int)):
                        mw_p = f"{mw_p:.6f}"
                    
                    # Format BH corrected p-value
                    if isinstance(bh_corrected_p, (float, int)):
                        bh_corrected_p_str = f"{bh_corrected_p:.6f}"
                    else:
                        bh_corrected_p_str = str(bh_corrected_p)
                        
                    # Cliff's Delta
                    cliff_d = results.get('cliffs_delta', "N/A")
                    if isinstance(cliff_d, (float, int)):
                        cliff_d = f"{cliff_d:.4f}"
                    
                    # Cliff's Delta CI Lower
                    ci_lower = results.get('ci_lower', "N/A")
                    if isinstance(ci_lower, (float, int)):
                        ci_lower = f"{ci_lower:.4f}"

                    # Cliff's Delta CI Upper
                    ci_upper = results.get('ci_upper', "N/A")
                    if isinstance(ci_upper, (float, int)):
                        ci_upper = f"{ci_upper:.4f}"
                        
                    # Effect size
                    effect_size = results.get('effect_size', "N/A")
                    
                    row_html = f"""
                        <tr {row_class}>
                            <td>{event_display}</td>
                            <td>{before_size}</td>
                            <td>{after_size}</td>
                            <td>{mw_stat}</td>
                            <td>{mw_p}</td>
                            <td class="{sig_class}">{sig_text}</td>
                            <td>{bh_corrected_p_str}</td>
                            <td class="{bh_sig_class}">{bh_sig_text}</td>
                            <td>{cliff_d}</td>
                            <td>{ci_lower}</td>
                            <td>{ci_upper}</td>
                            <td>{effect_size}</td>
                            <td>{get_cliffs_delta_explanation(results.get('cliffs_delta', 'N/A'), 
                                    results.get('ci_lower', 'N/A'), 
                                    results.get('ci_upper', 'N/A'))}</td>
                        </tr>
                    """
                    rows_with_data.append(row_html)
                

            
            # Write rows with data first, then rows without data
            for row in rows_with_data:
                f.write(row)
            # for row in rows_without_data:
            #     f.write(row)
            
            f.write("""
                </table>
                </div>
            """)
            
            # 3. Recovery Impact Analysis
            f.write("""
                <h3>Recovery Impact Analysis</h3>
                <div class="table-wrapper">
                <table>
                    <tr>
                        <th>Event Type</th>
                        <th>Avg TRB Before (s)</th>
                        <th>Avg TRB After (s)</th>
                        <th>Faster After Event</th>
                    </tr>
            """)
            
            # Define all event types to ensure complete table
            event_types_display = {
                'all_goals': 'All Goals',
                'substitutions': 'Substitutions', 
                'tactical_shifts': 'Tactical Shifts',
                'injuries': 'Injuries',
                'yellow_cards': 'Yellow Cards',
                'red_cards': 'Red Cards'
            }
            
            # Collect all rows first
            rows_with_data = []
            rows_without_data = []
            
            # Process each event type
            for event_type, event_display in event_types_display.items():
                found = False
                
                # Find team-level impact results
                if not impact_results.empty:
                    team_impacts = impact_results[
                        (impact_results['Team'] == team_name) & 
                        (impact_results['Match ID'] == 'All Matches')
                    ]
                    
                    for _, row in team_impacts.iterrows():
                        if row['Event Type'] == event_display:
                            found = True
                            
                            avg_before = row['Avg TRB Before (s)']
                            avg_after = row['Avg TRB After (s)']
                            faster = row['Faster After Event']
                            p_value = row['p-value']
                            significant = row['Significant Impact']
                            
                            # CSS classes
                            faster_class = "positive" if faster == "Yes" else "negative"
                            sig_class = "positive" if significant == "Yes" else "negative"
                            
                            # Highlight row if significant
                            row_class = 'class="highlight"' if significant == "Yes" else ''
                            
                            row_html = f"""
                            <tr {row_class}>
                                <td>{event_display}</td>
                                <td>{avg_before}</td>
                                <td>{avg_after}</td>
                                <td class="{faster_class}">{faster}</td>
                            </tr>
                            """
                            rows_with_data.append(row_html)
                            break
                

            
            # Write rows with data first, then rows without data
            for row in rows_with_data:
                f.write(row)
            # for row in rows_without_data:
            #     f.write(row)
            
            f.write("""
                </table>
                </div>
            """)
            
            # Generate match list for this team
            f.write("""
                <h2>Match-specific Analysis</h2>
                <div class="event-card">
                    <p>Select a match to view detailed analysis:</p>
                    <ul>
            """)
            
            # Get matches for this team
            team_matches = {}
            for team_name_data, matches_dict in team_dicts:
                if team_name_data == team_name:
                    for match_id, match_details in matches_dict.items():
                        team_matches[match_id] = match_details
            
            for match_id, match_details in team_matches.items():
                opposing_team = match_details['team']
                opposing_flag = get_flag_emoji(opposing_team)
                # Update link to point to match-specific directory
                f.write(f'        <li><a href="../match_{match_id}/{team_slug}_analysis.html"><span class="team-flag">{flag_emoji}</span> {team_name} vs <span class="team-flag">{opposing_flag}</span> {opposing_team} (Match ID: {match_id})</a></li>\n')
            
            f.write("""
                    </ul>
                </div>
                
                <div class="footer">
                </div>
                
                """ + create_help_button_and_modal() + """
            </body>
            </html>
            """)
    
    # Helper function to get significant findings for a team in a specific match
    def get_match_significant_findings(team_name, match_id):
        findings = []
        
        # Filter impact results for this team and match
        if not impact_results.empty:
            match_impacts = impact_results[
                (impact_results['Team'] == team_name) & 
                (impact_results['Match ID'] == match_id)
            ]
            significant_impacts = match_impacts[match_impacts['Significant Impact'] == 'Yes']
            
            for _, row in significant_impacts.iterrows():
                event_type = row['Event Type']
                faster = row['Faster After Event'] == 'Yes'
                effect = "faster" if faster else "slower"
                
                findings.append(f"<strong>{event_type}:</strong> Recovery times are significantly {effect} after {event_type.lower()} in this match.")
        
        return findings
    
    # Generate match-specific HTML files
    for team_name, matches_dict in team_dicts:
        team_slug = team_name.replace(' ', '_').lower()
        flag_emoji = get_flag_emoji(team_name)
        
        for match_id, match_details in matches_dict.items():
            opposing_team = match_details['team']
            opposing_flag = get_flag_emoji(opposing_team)
            
            # Get significant findings for this match
            match_findings = get_match_significant_findings(team_name, match_id)
            match_dir = os.path.join(output_dir, f"match_{match_id}")
            
            # Create match-specific HTML file in the match directory
            with open(os.path.join(match_dir, f"{team_slug}_analysis.html"), "w") as f:
                f.write(f"""<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{team_name} vs. {opposing_team} - Match Analysis</title>
                {css_styles}
                {help_js}
            </head>
            <body style="max-width: 100%; width: 100%;">
                <h1><span class="team-flag">{flag_emoji}</span> {team_name} vs <span class="team-flag">{opposing_flag}</span> {opposing_team} - Match Analysis</h1>
                <p>Match ID: {match_id}</p>
                
                <div class="summary-box">
                    <h3>Key Findings</h3>
            """)
                
                if match_findings:
                    # Show the significant findings first
                    f.write("<ul>\n")
                    for finding in match_findings:
                        f.write(f"    <li>{finding}</li>\n")
                    f.write("</ul>\n")
                    
                    # Add detailed event-specific analysis
                    f.write("<h4>Detailed Event Analysis</h4>\n")
                    
                    # Find match-specific impact results for event details
                    if not impact_results.empty:
                        match_impacts = impact_results[
                            (impact_results['Team'] == team_name) & 
                            (impact_results['Match ID'] == match_id)
                        ]
                        
                        for _, row in match_impacts.iterrows():
                            event_type = row['Event Type']
                            avg_before = row['Avg TRB Before (s)']
                            avg_after = row['Avg TRB After (s)']
                            faster = row['Faster After Event'] == "Yes"
                            p_value = float(row['p-value']) if isinstance(row['p-value'], str) else row['p-value']
                            significant = row['Significant Impact'] == "Yes"
                            
                            # Create event analysis
                            effect_word = "faster" if faster else "slower"
                            significance_phrase = "significantly " if significant else ""
                            effect_size = ""
                            
                            # Use the formatted event type display
                            event_display = format_event_type(event_type.lower().replace(' ', '_'))
                            
                            # extract the full event type from the dataframe name
                            for df_name, mw_results in mannwhitney_results.items():
                                if f"match_{match_id}" in df_name and team_slug in df_name:
                                    # Extract the full event type from the dataframe name
                                    prefix = f"df_{team_slug}_match_{match_id}_"
                                    if df_name.startswith(prefix):
                                        extracted_event_type = df_name[len(prefix):]
                                        # Compare with the current event type
                                        if extracted_event_type == event_type.lower().replace(' ', '_'):
                                            if 'effect_size' in mw_results:
                                                effect_size = f" with a {mw_results['effect_size'].lower()} effect size (Cliff's Delta: {mw_results.get('cliffs_delta', 'N/A'):.4f})" if isinstance(mw_results.get('cliffs_delta'), (float, int)) else ""
                                            break
                            
                            # Write event analysis
                            f.write(f"""
                            <div style="margin-left: 20px; margin-bottom: 15px; padding: 10px; background-color: #f9f9f9; border-left: 3px solid {'#4CAF50' if significant else '#ddd'}; border-radius: 4px;">
                                <strong>{event_display}:</strong>
                        """)
                            
                            if significant:
                                f.write(f"""
                                The team has {significance_phrase}{effect_word} recovery times after {event_type.lower()} in this match{effect_size}. 
                                Average recovery time {'decreased' if faster else 'increased'} from {avg_before} seconds before to {avg_after} seconds after {event_type.lower()}.
                                """)
                            else:
                                f.write(f"""
                                No significant impact found for {event_type.lower()}, though there was a {'decrease' if faster else 'increase'} 
                                in average recovery time ({avg_before} to {avg_after}).
                                """)
                            
                            f.write("</div>\n")
                else:
                    f.write("    <p>No significant findings were identified for this match.</p>\n")
                
                f.write("""
                    </div>

                    <h2>Recovery Times - Match Level</h2>
                """)
                
                # Generate tables for match-level analysis
                
                # 1. Normality Test Results
                f.write("""
                    <h3>Normality Test Results</h3>
                    <div class="table-wrapper">
                    <table>
                        <tr>
                            <th>Event Type</th>
                            <th>Sample Size</th>
                            <th>Shapiro-Wilk Statistic</th>
                            <th>p-value</th>
                            <th>Normal Distribution</th>
                        </tr>
                """)
                
                # Define all event types to ensure complete table
                event_types = ['all_goals', 'substitutions', 'tactical_shifts', 'injuries', 'yellow_cards', 'red_cards']
                
                # Process each event type for match-specific normality results
                for event_type in event_types:
                    found = False
                    
                    # Find match-specific normality results
                    for df_name, results in normality_results.items():
                        # Only include match-specific results for this match
                        if f"match_{match_id}" in df_name and team_slug in df_name:
                            # Get event type
                            parts = df_name.split('_')
                            if len(parts) > 4:  
                                match_idx = df_name.find(f"_match_{match_id}_")
                                if match_idx != -1:
                                    test_event_type = df_name[match_idx + len(f"_match_{match_id}_"):]
                                else:
                                    test_event_type = parts[-1]
                                
                                if test_event_type == event_type:
                                    found = True
                                    
                                    # Format for display
                                    event_display = format_event_type(event_type)

                                    # Check for normality
                                    is_normal = results.get('shapiro_normal', False)
                                    normal_class = "positive" if is_normal else "negative"
                                    normal_text = "Yes" if is_normal else "No"
                                    
                                    # Sample size
                                    sample_size = results.get('sample_size', "N/A")
                                    
                                    # Test statistic
                                    shapiro_stat = results.get('shapiro_stat', "N/A")
                                    if isinstance(shapiro_stat, (float, int)):
                                        shapiro_stat = f"{shapiro_stat:.4f}"
                                        
                                    # p-value
                                    shapiro_p = results.get('shapiro_p', "N/A")
                                    if isinstance(shapiro_p, (float, int)):
                                        shapiro_p = f"{shapiro_p:.6f}"
                                    
                                    f.write(f"""
                                <tr>
                                    <td>{event_display}</td>
                                    <td>{sample_size}</td>
                                    <td>{shapiro_stat}</td>
                                    <td>{shapiro_p}</td>
                                    <td class="{normal_class}">{normal_text}</td>
                                </tr>
                                    """)
                                    break
                    
                    # # If no results found for this event type, add empty row
                    # if not found:
                    #     event_display = format_event_type(event_type)
                    #     f.write(f"""
                    #         <tr>
                    #             <td>{event_display}</td>
                    #             <td colspan="4" style="text-align: center; color: #999;">No data available</td>
                    #         </tr>
                    #     """)
                
                f.write("""
                    </table>
                    </div>
                """)
                
                # 2. Mann-Whitney Test Results with BH Correction
                f.write("""
                    <h3>Mann-Whitney U Test Results</h3>
                    <div class="table-wrapper">
                    <table>
                        <tr>
                            <th>Event Type</th>
                            <th>Before Sample Size</th>
                            <th>After Sample Size</th>
                            <th>Mann-Whitney U</th>
                            <th>p-value</th>
                            <th>Significant Difference</th>
                            <th>BH Corrected p-value</th>
                            <th>Significant After BH</th>
                            <th>Cliff's Delta</th>
                            <th>CI Lower</th>
                            <th>CI Upper</th>
                            <th>Effect Size</th>
                            <th>Explanation</th>
                        </tr>
                """)
                
                # Process each event type for match-specific Mann-Whitney results
                for event_type in event_types:
                    found = False
                    
                    # Find match-specific Mann-Whitney results
                    for df_name, results in mannwhitney_results.items():
                        # Only include match-specific results for this match
                        if f"match_{match_id}" in df_name and team_slug in df_name:
                            # Get event type
                            parts = df_name.split('_')
                            if len(parts) > 4:  
                                match_idx = df_name.find(f"_match_{match_id}_")
                                if match_idx != -1:
                                    test_event_type = df_name[match_idx + len(f"_match_{match_id}_"):]
                                else:
                                    test_event_type = parts[-1]
                                
                                if test_event_type == event_type:
                                    found = True
                                    
                                    # Format for display
                                    event_display = format_event_type(event_type)
                                    
                                    # Look up BH correction results
                                    lookup_key = f"{team_slug}_match_{match_id}_{event_type}"
                                    bh_data = match_bh_lookup.get(lookup_key, {})
                                    bh_corrected_p = bh_data.get('bh_corrected_p', 'N/A')
                                    significant_after_bh = bh_data.get('significant_after_bh', False)
                                    
                                    # Check for significance
                                    is_significant = results.get('significant', False)
                                    sig_class = "positive" if is_significant else "negative"
                                    sig_text = "Yes" if is_significant else "No"
                                    
                                    # BH significance class
                                    bh_sig_class = "positive" if significant_after_bh else "negative"
                                    bh_sig_text = "Yes" if significant_after_bh else "No"
                                    
                                    # Highlight row based on BH significance
                                    row_class = 'class="highlight"' if significant_after_bh else ''
                                    
                                    # Sample sizes
                                    before_size = results.get('before_sample_size', "N/A")
                                    after_size = results.get('after_sample_size', "N/A")
                                    
                                    # Test statistic
                                    mw_stat = results.get('mw_stat', "N/A")
                                    if isinstance(mw_stat, (float, int)):
                                        mw_stat = f"{mw_stat:.4f}"
                                        
                                    # p-value
                                    mw_p = results.get('mw_p', "N/A")
                                    if isinstance(mw_p, (float, int)):
                                        mw_p = f"{mw_p:.6f}"
                                    
                                    # Format BH corrected p-value
                                    if isinstance(bh_corrected_p, (float, int)):
                                        bh_corrected_p_str = f"{bh_corrected_p:.6f}"
                                    else:
                                        bh_corrected_p_str = str(bh_corrected_p)
                                        
                                    # Cliff's Delta
                                    cliff_d = results.get('cliffs_delta', "N/A")
                                    if isinstance(cliff_d, (float, int)):
                                        cliff_d = f"{cliff_d:.4f}"
                                        
                                    # Cliff's Delta CI Lower
                                    ci_lower = results.get('ci_lower', "N/A")
                                    if isinstance(ci_lower, (float, int)):
                                        ci_lower = f"{ci_lower:.4f}"

                                    # Cliff's Delta CI Upper
                                    ci_upper = results.get('ci_upper', "N/A")
                                    if isinstance(ci_upper, (float, int)):
                                        ci_upper = f"{ci_upper:.4f}"
                                    
                                    # Effect size
                                    effect_size = results.get('effect_size', "N/A")
                                    
                                    f.write(f"""
                                        <tr {row_class}>
                                            <td>{event_display}</td>
                                            <td>{before_size}</td>
                                            <td>{after_size}</td>
                                            <td>{mw_stat}</td>
                                            <td>{mw_p}</td>
                                            <td class="{sig_class}">{sig_text}</td>
                                            <td>{bh_corrected_p_str}</td>
                                            <td class="{bh_sig_class}">{bh_sig_text}</td>
                                            <td>{cliff_d}</td>
                                            <td>{ci_lower}</td>
                                            <td>{ci_upper}</td>
                                            <td>{effect_size}</td>
                                            <td>{get_cliffs_delta_explanation(results.get('cliffs_delta', 'N/A'), 
                                                results.get('ci_lower', 'N/A'), 
                                                results.get('ci_upper', 'N/A'))}</td>
                                        </tr>
                                    """)
                                    break
                    
                    # # If no results found for this event type, add empty row
                    # if not found:
                    #     event_display = format_event_type(event_type)
                    #     f.write(f"""
                    #         <tr>
                    #             <td>{event_display}</td>
                    #             <td colspan="12" style="text-align: center; color: #999;">No data available</td>
                    #         </tr>
                    #     """)
                
                f.write("""
                    </table>
                    </div>
                """)
                
                # 3. Recovery Impact Analysis
                f.write("""
                    <h3>Recovery Impact Analysis</h3>
                    <div class="table-wrapper">
                    <table>
                        <tr>
                            <th>Event Type</th>
                            <th>Avg TRB Before (s)</th>
                            <th>Avg TRB After (s)</th>
                            <th>Faster After Event</th>
                        </tr>
                """)
                
                # Define all event types to ensure complete table
                event_types_display = {
                    'all_goals': 'All Goals',
                    'substitutions': 'Substitutions', 
                    'tactical_shifts': 'Tactical Shifts',
                    'injuries': 'Injuries',
                    'yellow_cards': 'Yellow Cards',
                    'red_cards': 'Red Cards'
                }
                
                # Process each event type for match-specific impact results
                for event_type, event_display in event_types_display.items():
                    found = False
                    
                    # Find match-specific impact results
                    if not impact_results.empty:
                        match_impacts = impact_results[
                            (impact_results['Team'] == team_name) & 
                            (impact_results['Match ID'] == match_id)
                        ]
                        
                        for _, row in match_impacts.iterrows():
                            if row['Event Type'] == event_display:
                                found = True
                                
                                avg_before = row['Avg TRB Before (s)']
                                avg_after = row['Avg TRB After (s)']
                                faster = row['Faster After Event']
                                p_value = row['p-value']
                                significant = row['Significant Impact']
                                
                                # CSS classes
                                faster_class = "positive" if faster == "Yes" else "negative"
                                sig_class = "positive" if significant == "Yes" else "negative"
                                
                                # Highlight row if significant
                                row_class = 'class="highlight"' if significant == "Yes" else ''
                                
                                f.write(f"""
                                <tr {row_class}>
                                    <td>{event_display}</td>
                                    <td>{avg_before}</td>
                                    <td>{avg_after}</td>
                                    <td class="{faster_class}">{faster}</td>
                                </tr>
                                """)
                                break
                    
                    # # If no data found, add empty row
                    # if not found:
                    #     f.write(f"""
                    #         <tr>
                    #             <td>{event_display}</td>
                    #             <td colspan="5
                    #             " style="text-align: center; color: #999;">No data available</td>
                    #         </tr>
                    #     """)
                
                f.write("""
                    </table>
                    </div>
                """)
                
                f.write("""
                    <div class="footer">
                    </div>
                    
                    """ + create_help_button_and_modal() + """
                </body>
                </html>
                """)
    
    print(f"HTML files generated in {output_dir}")
    return "HTML generation complete"


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


def run_football_analysis(team_dicts, output_dir="visuals"):
    """
    Main function to run the football analysis and generate HTML files
    
    Args:
        team_dicts: List of tuples containing team names and match dictionaries
        output_dir: Directory to save HTML files
    """
    # First, run the analyze_event_recovery_times function
    print("Running analyze_event_recovery_times...")
    all_dataframes, normality_results, mannwhitney_results, match_level_tests, team_level_tests = analyze_event_recovery_times(*team_dicts)
    
    # Extract impact results from the test results
    print("Extracting impact results...")
    
    impact_data = []
    
    # Process match-level tests
    for test in match_level_tests:
        if test.get('before_n', 0) >= 5 and test.get('after_n', 0) >= 5:
            impact_row = {
                'Team': test['team'],
                'Match ID': test['match_id'],
                'Event Type': test['event_type'].replace('_', ' ').title(),
                'Avg TRB Before (s)': f"{test['avg_before']:.2f}" if not pd.isna(test['avg_before']) else "N/A",
                'Avg TRB After (s)': f"{test['avg_after']:.2f}" if not pd.isna(test['avg_after']) else "N/A",
                'Faster After Event': "Yes" if test.get('faster_after', False) else "No",
                'p-value': f"{test['p_value']:.6f}" if not pd.isna(test['p_value']) else "N/A",
                'Significant Impact': "Yes" if test.get('significant_after_bh', test.get('significant_uncorrected', False)) else "No"
            }
            impact_data.append(impact_row)
    
    # Process team-level tests
    for test in team_level_tests:
        if test.get('before_n', 0) >= 5 and test.get('after_n', 0) >= 5:
            impact_row = {
                'Team': test['team'],
                'Match ID': 'All Matches',
                'Event Type': test['event_type'].replace('_', ' ').title(),
                'Avg TRB Before (s)': f"{test['avg_before']:.2f}" if not pd.isna(test['avg_before']) else "N/A",
                'Avg TRB After (s)': f"{test['avg_after']:.2f}" if not pd.isna(test['avg_after']) else "N/A",
                'Faster After Event': "Yes" if test.get('faster_after', False) else "No",
                'p-value': f"{test['p_value']:.6f}" if not pd.isna(test['p_value']) else "N/A",
                'Significant Impact': "Yes" if test.get('significant_after_bh', test.get('significant_uncorrected', False)) else "No"
            }
            impact_data.append(impact_row)
    
    # Create DataFrame
    impact_results = pd.DataFrame(impact_data)
    
    # Sort by team, then match ID, then event type
    if not impact_results.empty:
        impact_results = impact_results.sort_values(['Team', 'Match ID', 'Event Type'])
    
    # Finally, generate the HTML files with the new parameters
    print("Generating HTML files...")
    result = generate_html_files(team_dicts, all_dataframes, normality_results, mannwhitney_results, 
                                impact_results, match_level_tests, team_level_tests, output_dir)
    
    return result


def integrated_football_analysis(team_dicts, output_dir="visuals"):
    """
    Run both the recovery time normality analysis and the football analytics
    functions to generate a complete dashboard with linked HTML files.
    
    Args:
        team_dicts: List of tuples, each containing (team_name, team_color, matches_dict)
        output_dir: Directory to save the output plots and reports
    
    Returns:
        Result of both analyses
    """
    # First, run the recovery time normality test
    print("Running recovery time normality analysis...")
    normality_results = test_recovery_time_normality(*team_dicts, output_dir=output_dir)
    
    # Convert team_dicts format for the run_football_analysis function
    football_team_dicts = [(team_name, matches_dict) for team_name, team_color, matches_dict in team_dicts]
    
    # Then, run the football analytics function
    print("Running football analytics...")
    football_results = run_football_analysis(football_team_dicts, output_dir=output_dir)
    
    print("All analyses complete!")
    return {
        "normality_results": normality_results,
        "football_results": football_results
    }

print("Starting integrated football analysis...")
integrated_football_analysis([
    ("Argentina", "#89CFF0", arg_matches), 
    ("France", "#0055A4", fr_matches)
], output_dir="visuals")