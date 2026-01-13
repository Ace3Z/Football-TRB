import pandas as pd
import numpy as np
from statsbombpy import sb
import plotly.graph_objs as go
import plotly.io as pio
import json
import pycountry
import os
import matplotlib.colors as mcolors

from match_finder import arg_matches, fr_matches
from trb import *


def add_explanation_buttons_to_html(html_file_path):
    """
    Adds explanation buttons and modals to plot HTML files.
    
    Args:
        html_file_path (str): Path to the HTML file to modify
    
    Returns:
        None: Modifies the file in place
    """
    
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Detect plot types in this file
    filename = os.path.basename(html_file_path).lower()
    has_qq = 'qq' in filename or 'QQ Plot' in html_content
    has_kde = 'kde' in filename or 'Density Plot' in html_content
    has_match_kde = ('match' in filename and 'kde' in filename) or ('Recovery Time Distributions' in html_content)
    has_match_events = 'match_events' in filename or 'Match Events' in html_content or 'Game Time (minutes)' in html_content
    
    # Create the HTML for the explanation button container
    explanation_button_html = """
    <div class="plot-explanation-container" style="position: absolute; top: 10px; right: 10px; z-index: 100;">
    """
    
    # Add buttons based on detected plot types
    if has_match_events:
        explanation_button_html += """
        <button id="explain-plot" class="explanation-button" data-plot-type="match-events">
            <span class="explanation-icon">❓</span> Explain This Plot
        </button>
        """
    elif has_qq and has_kde and not has_match_kde:
        # Combined QQ and KDE plot page
        explanation_button_html += """
        <button id="explain-plot" class="explanation-button" data-plot-type="combined">
            <span class="explanation-icon">❓</span> Explain This Plot
        </button>
        """
    elif has_match_kde:
        explanation_button_html += """
        <button id="explain-plot" class="explanation-button" data-plot-type="match-kde">
            <span class="explanation-icon">❓</span> Explain This Plot
        </button>
        """
    elif has_kde:
        explanation_button_html += """
        <button id="explain-plot" class="explanation-button" data-plot-type="kde">
            <span class="explanation-icon">❓</span> Explain This Plot
        </button>
        """
    elif has_qq:
        explanation_button_html += """
        <button id="explain-plot" class="explanation-button" data-plot-type="qq">
            <span class="explanation-icon">❓</span> Explain This Plot
        </button>
        """
    
    explanation_button_html += """
    </div>
    """
    
    # Create the HTML for the explanation modal
    explanation_modal_html = """
    <div id="explanation-modal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h2 id="modal-title">Plot Explanation</h2>
          <span class="close-modal">&times;</span>
        </div>
        <div class="modal-body">
          <div id="combined-explanation" class="explanation-content">
            <h3>Understanding QQ and KDE Plots</h3>
            
            <div class="explanation-section">
              <h4>Left Side: QQ Plot</h4>
              <p>This plot helps us understand if recovery times follow a "normal" pattern or if they show unusual trends:</p>
              <ul>
                <li><strong>Red line:</strong> This is the "expected" pattern if all players were recovering at similar rates</li>
                <li><strong>Blue dots:</strong> Each dot represents actual player recovery times</li>
                <li><strong>What to look for:</strong> 
                  <ul>
                    <li>Dots following the line closely = recovery times are consistent across the team</li>
                    <li>Dots curving away from the line = some players recover much differently than others</li>
                    <li>Dots above the line at right side = some players taking much longer to recover</li>
                  </ul>
                </li>
                <li><strong>P-value in title:</strong> Below 0.05 means the team has inconsistent recovery patterns that may need attention</li>
              </ul>
              <p><strong>Football insights:</strong></p>
              <ul>
                <li>Identify players who don't fit the team's recovery profile</li>
                <li>Spot if your team has distinct "fast recoverers" and "slow recoverers" that need different training approaches</li>
                <li>Track if tactical changes impact recovery consistency</li>
              </ul>
            </div>
            
            <div class="explanation-section">
              <h4>Right Side: KDE Plot</h4>
              <p>This plot shows the distribution of how quickly players recover after losing possession:</p>
              <ul>
                <li><strong>Blue curve:</strong> Shows the frequency of different recovery times - peaks indicate most common recovery times</li>
                <li><strong>Blue histogram:</strong> Actual recovery time frequencies</li>
                <li><strong>Blue circles:</strong> Individual player recovery instances</li>
                <li><strong>Vertical lines:</strong> Median (blue) and average (green) recovery times</li>
              </ul>
              <p><strong>Football insights:</strong></p>
              <ul>
                <li>Lower peaks (shorter recovery times) indicate more effective pressing/counterpress</li>
                <li>Wide spread indicates inconsistent recovery approach across the team</li>
                <li>Long "tail" to the right suggests specific situations where recovery breaks down</li>
                <li>Big difference between average and median suggests a few problematic recovery scenarios affecting overall performance</li>
                <li>Use to compare recovery effectiveness across different formations, opponents, or pitch zones</li>
              </ul>
            </div>
          </div>
          
          <div id="kde-explanation" class="explanation-content">
            <h3>Understanding Recovery Time Distribution</h3>
            <p>This density plot (KDE) shows how quickly your team recovers possession after losing the ball:</p>
            <ul>
              <li><strong>Blue curve:</strong> Shows the frequency of different recovery times - the peak shows your team's most common recovery time</li>
              <li><strong>Blue histogram:</strong> The actual count of recovery instances at each time point</li>
              <li><strong>Blue circles:</strong> Individual recovery instances from the match data</li>
              <li><strong>Vertical lines:</strong> Show the median (blue) and average (green) recovery times</li>
            </ul>
            <p><strong>What to look for:</strong></p>
            <ul>
              <li><strong>Peak position:</strong> Lower values (left-side peaks) mean faster recoveries - better pressing efficiency</li>
              <li><strong>Curve width:</strong> Narrow peaks mean consistent recovery approach; wide spread suggests variable recovery tactics</li>
              <li><strong>Right-side tail:</strong> Long tail means some situations lead to much longer recovery times</li>
              <li><strong>Multiple peaks:</strong> Could indicate different recovery strategies in different game situations</li>
              <li><strong>Average vs Median gap:</strong> Large gap suggests a few problematic recovery situations affecting overall performance</li>
            </ul>
            <p><strong>Football applications:</strong></p>
            <ul>
              <li>Compare recovery performance across different matches, periods, or opponents</li>
              <li>Identify if tactical changes are improving recovery times</li>
              <li>Spot if fatigue is affecting recovery (later periods showing longer times)</li>
              <li>Evaluate effectiveness of pressing strategies</li>
              <li>Identify if specific player rotations improve recovery performance</li>
            </ul>
          </div>
          
          <div id="qq-explanation" class="explanation-content">
            <h3>Understanding Recovery Time Patterns</h3>
            <p>This QQ plot helps coaches identify if recovery times follow expected patterns or show unusual trends:</p>
            <ul>
              <li><strong>Red line:</strong> Represents "normal" distribution - what we'd expect if recovery was consistent</li>
              <li><strong>Blue dots:</strong> Each dot represents actual recovery time instances</li>
              <li><strong>P-value:</strong> Statistical measure of how "normal" your recovery patterns are (below 0.05 means unusual patterns)</li>
            </ul>
            <p><strong>What to look for:</strong></p>
            <ul>
              <li><strong>Dots following the line:</strong> Team recovers consistently across all situations</li>
              <li><strong>Dots curving above line (right side):</strong> Some recovery instances taking much longer than expected</li>
              <li><strong>Dots curving below line (left side):</strong> Some unusually quick recoveries</li>
              <li><strong>S-shaped curve:</strong> Team has mixed recovery performance - very quick in some situations, very slow in others</li>
            </ul>
            <p><strong>Football applications:</strong></p>
            <ul>
              <li>Identify if you have distinct "recovery profiles" within your team that need different training</li>
              <li>Spot if your recovery approach changes dramatically in different match situations</li>
              <li>Evaluate if your team's recovery times are predictable (following the line) or highly variable</li>
              <li>Determine if recovery performance is being affected by specific opposition tactics</li>
              <li>Use to decide if you need one unified pressing strategy or multiple approaches for different scenarios</li>
            </ul>
          </div>
          
          <div id="match-kde-explanation" class="explanation-content">
            <h3>Understanding Match Recovery Comparison</h3>
            <p>This plot compares ball recovery times between teams and across different match periods:</p>
            <ul>
              <li><strong>Different colored lines:</strong> Each color represents a different team</li>
              <li><strong>Line styles:</strong> Solid lines show all periods combined; dashed/dotted lines show individual periods</li>
              <li><strong>Peaks:</strong> The most common recovery time for each team</li>
              <li><strong>Vertical lines:</strong> Show mean recovery times for each team</li>
            </ul>
            <p><strong>What to look for:</strong></p>
            <ul>
              <li><strong>Left-shifted peaks:</strong> Teams with peaks further left recover possession faster</li>
              <li><strong>Peak height:</strong> Taller, narrower peaks indicate more consistent recovery times</li>
              <li><strong>Period differences:</strong> Compare how recovery times change across periods to spot fatigue or tactical shifts</li>
              <li><strong>Curve overlap:</strong> Where curves overlap, teams have similar recovery performance</li>
              <li><strong>Multiple peaks:</strong> Could indicate different recovery scenarios (e.g., high press vs. low block)</li>
            </ul>
            <p><strong>Football applications:</strong></p>
            <ul>
              <li>Directly compare your pressing efficiency against opponents</li>
              <li>Identify which periods your team had recovery advantage/disadvantage</li>
              <li>Spot if opponent's tactics affected your recovery performance</li>
              <li>See if substitutions improved recovery times in later periods</li>
              <li>Evaluate if recovery performance correlates with match outcome</li>
              <li>Determine when to increase/decrease pressing intensity based on recovery trends</li>
            </ul>
          </div>
          
          <div id="match-events-explanation" class="explanation-content">
            <h3>Understanding Match Events Plot</h3>
            <p>This timeline visualization shows recovery times alongside key match events, helping coaches connect tactical moments with recovery performance:</p>
            <ul>
              <li><strong>Recovery time lines:</strong> The colored lines show recovery times of each team throughout the match</li>
              <li><strong>Y-axis:</strong> Shows time taken to recover possession (in seconds) - lower values indicate faster recovery</li>
              <li><strong>X-axis:</strong> Match timeline in minutes</li>
              <li><strong>Event icons:</strong> Shows goals ⚽, injuries 🤕, substitutions 🔄, tactical shifts 🔄, and cards 🟨/🟥</li>
              <li><strong>Hover tooltips:</strong> Reveal detailed information about specific recovery instances or match events</li>
            </ul>
            
            <p><strong>What to look for:</strong></p>
            <ul>
              <li><strong>Recovery time patterns:</strong> Identify periods of effective and poor recovery performance</li>
              <li><strong>Event impact:</strong> See how events like goals, substitutions or tactical shifts affect recovery times</li>
              <li><strong>Pre/post injury:</strong> Compare recovery performance before and after player injuries</li>
              <li><strong>Formation influence:</strong> Notice how formation changes impact recovery efficiency</li>
              <li><strong>Period-to-period changes:</strong> Track how fatigue affects recovery across match periods</li>
            </ul>
            
            <p><strong>Football applications:</strong></p>
            <ul>
              <li>Identify which in-game events create recovery vulnerabilities</li>
              <li>Assess if substitutions improve or disrupt recovery patterns</li>
              <li>Determine optimal times for tactical adjustments based on recovery trends</li>
              <li>Analyze which formations deliver most consistent recovery performance</li>
              <li>Spot connections between recovery breakdown and conceded goals</li>
              <li>Understand how specific player injuries impact team-wide recovery efficiency</li>
            </ul>
            
            <p>The detailed match events table below the plot provides additional context about each event, and clicking on formation details reveals full team positioning information.</p>
          </div>
        </div>
      </div>
    </div>
    """
    
    # Create the CSS styles with BLUE button
    explanation_css = """
    <style>
    /* Plot Explanation Button */
    .explanation-button {
      background-color: #1e88e5; /* Changed to blue to match your dashboard */
      color: white;
      border: none;
      border-radius: 4px;
      padding: 8px 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      transition: all 0.2s;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .explanation-button:hover {
      background-color: #1565C0; /* Darker blue on hover */
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .explanation-icon {
      font-size: 16px;
    }
    
    /* Modal Styles */
    .modal {
      display: none;
      position: fixed;
      z-index: 1000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.6);
    }
    
    .modal-content {
      background-color: white;
      margin: 5% auto;
      width: 80%;
      max-width: 900px;
      border-radius: 8px;
      box-shadow: 0 5px 30px rgba(0,0,0,0.3);
      animation: modalFadeIn 0.3s;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
    }
    
    @keyframes modalFadeIn {
      from {opacity: 0; transform: translateY(-20px);}
      to {opacity: 1; transform: translateY(0);}
    }
    
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 15px 20px;
      border-bottom: 1px solid #eee;
    }
    
    .modal-header h2 {
      margin: 0;
      color: #1e88e5;
      font-size: 22px;
    }
    
    .close-modal {
      font-size: 28px;
      font-weight: bold;
      color: #666;
      cursor: pointer;
    }
    
    .close-modal:hover {
      color: #000;
    }
    
    .modal-body {
      padding: 20px;
      overflow-y: auto;
    }
    
    .explanation-content {
      display: none;
      line-height: 1.6;
    }
    
    .explanation-content.active {
      display: block;
    }
    
    .explanation-section {
      margin-bottom: 30px;
      padding-bottom: 20px;
      border-bottom: 1px solid #eee;
    }
    
    .explanation-section:last-child {
      border-bottom: none;
      margin-bottom: 0;
      padding-bottom: 0;
    }
    
    .explanation-content h3 {
      margin-top: 0;
      color: #1e88e5;
      font-size: 20px;
      margin-bottom: 15px;
    }
    
    .explanation-content h4 {
      color: #1565C0;
      font-size: 18px;
      margin-top: 0;
      margin-bottom: 10px;
    }
    
    .explanation-content p {
      margin-bottom: 15px;
    }
    
    .explanation-content ul {
      padding-left: 20px;
      margin-bottom: 15px;
    }
    
    .explanation-content li {
      margin-bottom: 8px;
    }
    
    /* Media query for smaller screens */
    @media (max-width: 768px) {
      .modal-content {
        width: 95%;
        margin: 5% auto;
      }
    }
    </style>
    """
    
    # Create the JavaScript code
    explanation_js = """
    <script>
    // Wait for page to fully load before initializing
    window.addEventListener('load', function() {
        const modal = document.getElementById('explanation-modal');
        const modalTitle = document.getElementById('modal-title');
        const closeModal = document.querySelector('.close-modal');
        const combinedExplanation = document.getElementById('combined-explanation');
        const kdeExplanation = document.getElementById('kde-explanation');
        const qqExplanation = document.getElementById('qq-explanation');
        const matchKdeExplanation = document.getElementById('match-kde-explanation');
        const matchEventsExplanation = document.getElementById('match-events-explanation');
        const explainButton = document.getElementById('explain-plot');
        
        // Set the active explanation based on plot type
        function showExplanation(plotType) {
            // Hide all explanations first
            combinedExplanation.classList.remove('active');
            kdeExplanation.classList.remove('active');
            qqExplanation.classList.remove('active');
            matchKdeExplanation.classList.remove('active');
            matchEventsExplanation.classList.remove('active');
            
            // Show the appropriate explanation
            if (plotType === 'combined') {
                combinedExplanation.classList.add('active');
                modalTitle.textContent = 'Understanding QQ and KDE Plots';
            } else if (plotType === 'kde') {
                kdeExplanation.classList.add('active');
                modalTitle.textContent = 'Understanding KDE Plots';
            } else if (plotType === 'qq') {
                qqExplanation.classList.add('active');
                modalTitle.textContent = 'Understanding QQ Plots';
            } else if (plotType === 'match-kde') {
                matchKdeExplanation.classList.add('active');
                modalTitle.textContent = 'Understanding Match KDE Plots';
            } else if (plotType === 'match-events') {
                matchEventsExplanation.classList.add('active');
                modalTitle.textContent = 'Understanding Match Events Plot';
            }
            
            // Show the modal
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden'; // Prevent scrolling behind modal
        }
        
        // Add click event to the explain button
        if (explainButton) {
            explainButton.addEventListener('click', function() {
                const plotType = this.getAttribute('data-plot-type');
                showExplanation(plotType);
            });
        }
        
        // Close modal when clicking the X
        if (closeModal) {
            closeModal.addEventListener('click', function() {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto'; // Restore scrolling
            });
        }
        
        // Close modal when clicking outside of it
        window.addEventListener('click', function(event) {
            if (event.target === modal) {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && modal.style.display === 'block') {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
    });
    </script>
    """
    
    # Add content before the closing body tag
    if '</body>' in html_content:
        modified_html = html_content.replace('</body>', 
                                            f'{explanation_button_html}\n{explanation_modal_html}\n{explanation_css}\n{explanation_js}\n</body>')
    else:
        # If there's no closing body tag, append at the end
        modified_html = html_content + f'\n{explanation_button_html}\n{explanation_modal_html}\n{explanation_css}\n{explanation_js}'
    
    # Write the modified content back to the file
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(modified_html)

def add_explanations_to_all_plots(directory):
    """
    Recursively adds explanation buttons to all plot HTML files in a directory.
    
    Args:
        directory (str): Path to the directory containing HTML plot files
        
    Returns:
        int: Number of files modified
    """
    
    count = 0
    
    # Walk through all files in the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                
                # Skip the dashboard file
                if file == 'dashboard.html':
                    continue
                
                # Check if file contains plot data (simple check)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(2000)  # Read first 2000 chars to check
                        if ('plotly' in content or 'Plotly' in content) and ('plot' in content.lower()):
                            # Add explanation buttons to the file
                            add_explanation_buttons_to_html(file_path)
                            count += 1
                except Exception as e:
                    print(f"Error checking file {file_path}: {e}")
    
    return count





def preprocess_game_data(df, selected_team):
    """
    Preprocess game data for a selected team.
    Filters the DataFrame for the selected team, converts time-related columns to the proper
    data types, and computes the game time in minutes from the match start (defined as the first
    loss event). Returns the processed DataFrame or None if no valid data is available.
    """
    team_data = df[df['team'] == selected_team].copy()
    team_data['loss_time'] = pd.to_datetime(team_data['loss_time'], errors='coerce')
    team_data['time_to_recover'] = pd.to_numeric(team_data['time_to_recover'], errors='coerce')
    team_data = team_data.dropna(subset=['loss_time', 'time_to_recover'])
    
    if team_data.empty:
        return None  

    # Set match start time to the first ball loss event
    match_start_time = team_data['loss_time'].min() 
    print(match_start_time)
    # Ensure match_start_time is valid
    if pd.isna(match_start_time):
        return None  

    # Compute game time in minutes from match start
    team_data['game_time_minutes'] = (team_data['loss_time'] - match_start_time).dt.total_seconds() / 60

    return team_data

# Periods to process
periods = [1, 2, 3, 4]

def get_flag_emoji(country_name: str) -> str:
    name = country_name.strip().lower()
    if name in ("england", "scotland", "wales", "northern ireland"):
        return "".join(chr(0x1F1E6 + (ord(c) - ord('A'))) for c in "GB")
    try:
        country = pycountry.countries.lookup(country_name)
        code = country.alpha_2.upper()
        return "".join(chr(0x1F1E6 + (ord(c) - ord('A'))) for c in code)
    except Exception:
        return ""

def generate_match_plots(*team_dicts, output_dir="visuals"):  
    """
    Generate interactive recovery plots for football matches with match event tables
    and an index HTML page linking to them.
    This function considers only periods 1, 2, 3, and 4, and will skip any period with no data.
    The dropdown menu now shows only individual periods, with Period 1 selected by default.
    Can accept multiple team dictionaries at once.
    
    Args:
        *team_dicts: Variable number of tuples, each containing (team_name, team_color, matches_dict)
    """
    # Define periods to consider
    periods = [1, 2, 3, 4]
    
    # Define emoji mapping for event types
    emoji_map = {
        "Goal": "⚽",
        "Own Goal": "🚫⚽",
        "Substitution": "🔄",
        "Tactical Shift": "🔀",
        "Injury": "🤕",
        "Yellow Card": "🟨",
        "Red Card": "🟥"
    }
    
    # Define emojis for player substitutions
    player_in_emoji = "⬆️"  # Player coming in
    player_out_emoji = "⬇️"  # Player going out
    
    def get_recovery_value(event_time, team_data):
        """
        Given an event time and the team's recovery data (with 'game_time_minutes' and 'time_to_recover'),
        return an interpolated recovery value at that event time.
        """
        if team_data.empty:
            return None
        return np.interp(event_time, team_data['game_time_minutes'], team_data['time_to_recover'])
    
    def extract_player_name(row):
        """
        Extract player name from event data in a consistent way.
        """
        if row.get('type') == 'Substitution':
            player_out = row.get('player', {}).get('name', 'Unknown') if isinstance(row.get('player'), dict) else row.get('player', 'Unknown')
            player_in = row.get('substitution_replacement', {}).get('name', 'Unknown') if isinstance(row.get('substitution_replacement'), dict) else row.get('substitution_replacement', 'Unknown')
            return f"{player_in_emoji} {player_in} {player_out_emoji} {player_out}"
        return row.get('player', {}).get('name', 'Unknown') if isinstance(row.get('player'), dict) else row.get('player', 'Unknown')
    
    def process_match(match_id, matches_dict):
        """
        Process match events to create a structured dataframe for display.
        Only includes events from periods 1, 2, 3, and 4.
        Tracks formation changes for tactical shifts.
        Ensures scores are calculated correctly in chronological order.
        
        Args:
            match_id: The ID of the match to process
            matches_dict: Dictionary containing the match details
            
        Returns:
            A dictionary containing match information and processed dataframe
        """
        try:
            # Get the match details
            details = matches_dict.get(match_id)
            if not details:
                print(f"No details found for Match ID {match_id}")
                return None
                
            # Fetch event data for the match
            events_df = sb.events(match_id=match_id)
            if events_df.empty:
                print(f"No event data found for Match ID {match_id}")
                return None
            
            # Get the team names
            primary_team = details['team']
            
            # Determine the opposing team from events
            all_teams = events_df['team'].dropna().unique()
            opponent_teams = [team for team in all_teams if team != primary_team]
            opponent_team = opponent_teams[0] if opponent_teams else "Unknown Team"
            team_flags[opponent_team] = get_flag_emoji(opponent_team)

            # Process events into a display-friendly format
            processed_events = []
            
            # Filter events to only include periods 1, 2, 3, and 4
            valid_periods = [1, 2, 3, 4]
            events_df = events_df[events_df['period'].isin(valid_periods)]
            
            # Sort events by period and then by timestamp within each period
            # This ensures proper chronological ordering for score calculation
            events_df = events_df.sort_values(by=['period', 'timestamp'])
            
            # Remove duplicate events if present
            if 'id' in events_df.columns:
                events_df = events_df.drop_duplicates(subset=['id'])
            
            # Initialize score counters
            primary_score, opponent_score = 0, 0
            
            # Create dictionaries to track formations by team and period
            team_formations = {
                primary_team: {period: [] for period in valid_periods},
                opponent_team: {period: [] for period in valid_periods}
            }
            
            # Extract starting formation data
            starting_xi_events = events_df[events_df['type'] == 'Starting XI']
            for _, event in starting_xi_events.iterrows():
                team = event.get('team', 'Unknown')
                tactics = event.get('tactics') or {}
                formation = tactics.get('formation', 'Unknown')
                lineup = tactics.get('lineup', [])
                
                # Store the initial formation in period 1
                if team in team_formations:
                    team_formations[team][1].append({
                        'timestamp': event.get('timestamp'),
                        'minute': event.get('minute', 0),
                        'formation': formation,
                        'lineup': lineup
                    })
            
            # Process events
            for _, event in events_df.iterrows():
                event_type = event['type']
                team = event.get('team', 'Unknown')
                period = event.get('period', '')
                minute = event.get('minute', '')
                second = event.get('second', 0)
                time_display = f"{minute}" if pd.notna(minute) else ""
                
                evt_id = event.get('id')

                # Extract details based on event type
                details = ""
                if event_type == 'Shot' and event.get('shot_outcome') == 'Goal':
                    # Update score for the correct team
                    if team == primary_team:
                        primary_score += 1
                    else:
                        opponent_score += 1
                    event_type = 'Goal'
                    player_name = extract_player_name(event)
                    details = f"{extract_player_name(event)}: <strong> scored </strong>"

                elif event_type == 'Own Goal Against':
                    # Get the original team (the one that actually scored the own goal)
                    own_goal_by_team = team
                    own_goal_by_player = extract_player_name(event)
                    
                    # For own goals, credit the opposing team
                    if team == primary_team:
                        opponent_score += 1  # Primary team scored an own goal, opponent gets the point
                        benefiting_team = opponent_team
                    else:
                        primary_score += 1  # Opponent scored an own goal, primary team gets the point
                        benefiting_team = primary_team
                    
                    # Set the display team to the team that gets the point
                    team = benefiting_team
                    event_type = 'Own Goal'
                    
                    # Create clearer details with info about who scored the own goal
                    details = f"{own_goal_by_player} ({own_goal_by_team}): <strong>OWN GOAL ⚽ (point for {benefiting_team})</strong>"
                    
                    # Store this information for hover text later
                    event['own_goal_by_team'] = own_goal_by_team
                    event['own_goal_by_player'] = own_goal_by_player
                    event['own_goal_benefiting_team'] = benefiting_team

                elif event_type == 'Substitution':
                    player_name = extract_player_name(event)
                    details = player_name
                elif event_type == 'Injury Stoppage':
                    event_type = 'Injury'
                    player_name = extract_player_name(event)
                    details = f"{player_name}: <strong> injured </strong>"
                elif event_type == 'Tactical Shift':
                    # Get the new formation details
                    new_formation = event.get('tactics', {}).get('formation', 'Unknown')
                    lineup_data = event.get('tactics', {}).get('lineup', [])
                    
                    # Store the new formation in our tracking dictionary for this period
                    if team in team_formations and period in team_formations[team]:
                        team_formations[team][period].append({
                            'timestamp': event.get('timestamp'),
                            'minute': minute,
                            'formation': new_formation,
                            'lineup': lineup_data
                        })
                    
                    # Get the previous formation for this team in this period
                    prev_formation = None
                    prev_lineup = None
                    
                    if team in team_formations and period in team_formations[team]:
                        period_formations = team_formations[team][period]
                        
                        if len(period_formations) > 1:
                            # Get the previous formation from the same period
                            prev_idx = len(period_formations) - 2
                            prev_formation = period_formations[prev_idx]['formation']
                            prev_lineup = period_formations[prev_idx]['lineup']
                        elif period > 1:
                            # If this is the first formation change in this period,
                            # try to get the last formation from the previous period
                            for prev_period in range(period-1, 0, -1):
                                prev_period_formations = team_formations[team][prev_period]
                                if prev_period_formations:
                                    prev_formation = prev_period_formations[-1]['formation']
                                    prev_lineup = prev_period_formations[-1]['lineup']
                                    break
                    
                    # Create the JSON data for the lineups
                    lineup_json = json.dumps(lineup_data).replace('"', '&quot;')
                    prev_lineup_json = json.dumps(prev_lineup).replace('"', '&quot;') if prev_lineup else ""
                    
                    # Get team styling information
                    team_color = team_colors.get(team.strip().title(), 'gray')
                    flag = team_flags.get(team, "")
                    team_label = f"{flag} {team}"
                    
                    # Create the clickable span with all necessary data attributes
                    if prev_formation:
                        details = (
                            f'<span class="formation-details" '
                            f'data-lineup="{lineup_json}" '
                            f'data-prev-lineup="{prev_lineup_json}" '
                            f'data-team-color="{team_color}" '
                            f'data-team-name="{team_label}" '
                            f'data-formation="{new_formation}" '
                            f'data-prev-formation="{prev_formation}" '
                            f'data-period="{period}">'
                            f'Formation change: {prev_formation} → {new_formation} (click for details)</span>'
                        )
                    else:
                        details = (
                            f'<span class="formation-details" '
                            f'data-lineup="{lineup_json}" '
                            f'data-team-color="{team_color}" '
                            f'data-team-name="{team_label}" '
                            f'data-formation="{new_formation}" '
                            f'data-period="{period}">'
                            f'New formation: {new_formation} (click for details)</span>'
                        )
                elif event_type == 'Foul Committed':
                    card = event.get('foul_committed_card', '')
                    if card == 'Yellow Card':
                        event_type = 'Yellow Card'
                        player_name = extract_player_name(event)
                        details = f"{player_name}: <strong> fouled </strong>"
                    elif card == 'Red Card':
                        event_type = 'Red Card'
                        player_name = extract_player_name(event)
                        details = f"{player_name}: <strong> fouled </strong>"
                    else:
                        continue  
                else:
                    continue  

                # Add the event with the current score
                processed_events.append({
                    'Period': period,
                    'Event': event_type,
                    'Team': team,
                    'Minute': time_display,
                    'Score': f"{opponent_score}–{primary_score}",  # Current score after this event
                    'Details': details
                })
                
            # Create DataFrame from processed events
            match_df = pd.DataFrame(processed_events)
            
            # Ensure events are in correct order for display
            if not match_df.empty:
                # Convert Period to numeric for sorting
                match_df['Period_num'] = pd.to_numeric(match_df['Period'], errors='coerce')
                # Convert Minute to numeric for sorting (handle format like "45:00" by extracting first part)
                match_df['Minute_num'] = match_df['Minute'].apply(
                    lambda x: float(x.split(':')[0]) if isinstance(x, str) and ':' in x else float(x) if pd.notna(x) else 0
                )
                # Sort by Period and then by Minute
                match_df = match_df.sort_values(by=['Period_num', 'Minute_num'])
                # Drop the helper columns used for sorting
                match_df = match_df.drop(columns=['Period_num', 'Minute_num'])
            
            # Return match info, processed data, and formation tracking
            return {
                'match_id': match_id,
                'primary_team': primary_team,
                'opponent_team': opponent_team,
                'match_df': match_df,
                'team_formations': team_formations
            }
            
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            import traceback
            traceback.print_exc()
            return None


    def generate_match_table_html(match_info, team_colors):
        """
        Generate HTML for the match event table, including kickoff formations
        shown like tactical shifts (clickable), with consistent team-color based styling.
        Works with both hex color codes and color names.
        """
        if match_info is None or match_info['match_df'].empty:
            return "<p>No match data available to display.</p>"

        match_df = match_info['match_df']
        primary_team = match_info['primary_team']
        opponent_team = match_info['opponent_team']
        match_id = match_info['match_id']
        team_formations = match_info.get('team_formations', {})
        

        def ensure_hex_color(color):
            """
            Convert any CSS‐style color name or hex code into normalized #RRGGBB.
            Falls back to light gray (#888888) if parsing fails.
            """
            try:
                # If it's already a hex code, normalize #RGB → #RRGGBB
                if color.startswith('#'):
                    hex_code = color.lstrip('#')
                    if len(hex_code) == 3:
                        hex_code = ''.join(c*2 for c in hex_code)
                    return '#' + hex_code.upper()
                # Otherwise let matplotlib parse the name (supports all CSS4/X11 names)
                rgb = mcolors.to_rgb(color)
                return mcolors.to_hex(rgb).upper()
            except Exception:
                return "#888888"

        # --- pull Kick-off lineup & formation from sb.events ---
        try:
            all_events  = sb.events(match_id=match_id)
            starting_xi = all_events[all_events['type'] == 'Starting XI']
        except Exception:
            starting_xi = []

        initial_formations = {}
        initial_lineups    = {}
        for _, ev in starting_xi.iterrows():
            team    = ev.get('team', 'Unknown')
            tactics = ev.get('tactics') or {}
            fv      = tactics.get('formation')
            form_str = str(fv) if fv else 'Unknown'
            initial_formations[team] = form_str
            initial_lineups[team]    = tactics.get('lineup', [])

        p_form = initial_formations.get(primary_team,  'Unknown')
        o_form = initial_formations.get(opponent_team, 'Unknown')
        p_line = initial_lineups.get(primary_team,    [])
        o_line = initial_lineups.get(opponent_team,   [])

        flag_p = team_flags.get(primary_team, "")
        flag_o = team_flags.get(opponent_team, "")
        
        # the opacity level for team color backgrounds
        opacity = "33"  
        
        # build HTML header (6 columns)
        html = f"""
        <div class="match-table-container">
        <h2>Match Events: {flag_p}{primary_team} vs {flag_o}{opponent_team}</h2>
        <table class="match-table">
            <thead>
            <tr>
                <th>Period</th>
                <th>Event</th>
                <th>Team</th>
                <th>Minute</th>
                <th>Score</th>
                <th>Details</th>
            </tr>
            </thead>
            <tbody>

            <!-- Kick-off formations -->
            <tr class="period-header">
                <td colspan="6"><strong>Kick-off Formations</strong></td>
            </tr>
            
        """

        # Primary kickoff row
        p_lineup_json = json.dumps(p_line).replace('"', '&quot;')
        p_team_color_raw = team_colors.get(primary_team, '#888888')
        p_team_color = ensure_hex_color(p_team_color_raw)
        p_bg_color = f"background-color: {p_team_color}{opacity};"
        
        span_p = (
            f'<span class="formation-details" '
            f'data-lineup="{p_lineup_json}" '
            f'data-team-color="{p_team_color}" '
            f'data-team-name="{flag_p} {primary_team}" '
            f'data-formation="{p_form}" '
            f'data-period="1">'
            f'<strong>Starting formation: {p_form} (click for details)</strong></span>'
        )
        html += f"""
            <tr style="{p_bg_color}">
                <td>1</td>
                <td>Formation</td>
                <td><span style="
                color:#fff;
                background-color:{p_team_color};
                padding:2px 6px;
                border-radius:4px;
                font-weight:bold;
                ">{flag_p} {primary_team}</span></td>
                <td>Kick-off</td>
                <td>0–0</td>
                <td class="tactical-shift">{span_p}</td>
            </tr>
        """

        # Opponent kickoff row
        o_lineup_json = json.dumps(o_line).replace('"', '&quot;')
        o_team_color_raw = team_colors.get(opponent_team, '#888888')
        o_team_color = ensure_hex_color(o_team_color_raw)
        o_bg_color = f"background-color: {o_team_color}{opacity};"
        
        span_o = (
            f'<span class="formation-details" '
            f'data-lineup="{o_lineup_json}" '
            f'data-team-color="{o_team_color}" '
            f'data-team-name="{flag_o} {opponent_team}" '
            f'data-formation="{o_form}" '
            f'data-period="1">'
            f'<strong>Starting formation: {o_form} (click for details)</strong></span>'
        )
        html += f"""
            <tr style="{o_bg_color}">
                <td>1</td>
                <td>Formation</td>
                <td><span style="
                color:#fff;
                background-color:{o_team_color};
                padding:2px 6px;
                border-radius:4px;
                font-weight:bold;
                ">{flag_o} {opponent_team}</span></td>
                <td>Kick-off</td>
                <td>0–0</td>
                <td class="tactical-shift">{span_o}</td>
            </tr>
        """

        # Event rows with consistent team-colored backgrounds 
        current_period = None
        for _, row in match_df.iterrows():
            period = row['Period']
            if period != current_period:
                current_period = period
                html += f"""
            <tr class="period-header">
                <td colspan="6"><strong>Period {period}</strong></td>
            </tr>
            """
            # Get team info and color
            team = row['Team']
            team_color_raw = team_colors.get(team.strip().title(), '#888888')
            team_color = ensure_hex_color(team_color_raw)
            # Create consistent background color with the specified opacity
            bg_color = f"background-color: {team_color}{opacity};"
            
            # Team label styling
            team_label_style = (
                "color:#fff;"
                f"background-color:{team_color};"
                "padding:2px 6px;"
                "border-radius:4px;"
                "font-weight:bold;"
            )

            # Special formatting for certain event types
            if row['Event'] == 'Tactical Shift':
                details_td = f'<td class="tactical-shift">{row["Details"]}</td>'
            else:
                details_td = f'<td>{row["Details"]}</td>'

            html += f"""
            <tr style="{bg_color}">
                <td><strong>{row['Period']}</strong></td>
                <td>{row['Event']}</td>
                <td>
                    <span style="{team_label_style}">
                        {team_flags.get(team, "")} {team}
                    </span>
                </td>
                <td>{row['Minute']}</td>
                <td>{row['Score']}</td>
                {details_td}
            </tr>
        """

        html += """
            </tbody>
        </table>
        </div>
        """
        
        return html

    # Dictionary to store match information for the index page
    match_info = {}
    match_tables = {}  # Store processed match tables
    team_colors = {}  # Store team colors for styling
    all_teams = []
    team_flags = {}  

    for team_name, team_color, matches_dict in team_dicts:
        all_teams.append(team_name)
        team_colors[team_name] = team_color
        team_flags[team_name]  = get_flag_emoji(team_name)

        for match_id, details in matches_dict.items():
            opp = details['team']
            team_flags[opp] = get_flag_emoji(opp)
        
            opp_color_data = details.get('color')
            if isinstance(opp_color_data, dict):
                team_colors[opp] = opp_color_data.get('opposing',
                                                    opp_color_data.get('main',
                                                                        team_color))
            else:
                team_colors[opp] = opp_color_data or team_color

        for match_id, details in matches_dict.items():
            try:
                print(f"\nProcessing Match ID: {match_id} ({details['team']})...")
                
                # Use the team_name parameter instead of a hard-coded team name
                team1 = details['team'].replace(' ', '_').lower()
                team2 = team_name.replace(' ', '_').lower()
                
                # Retrieve the main team data from a global DataFrame variable based on a naming convention
                df_main = globals()[f"df_time_to_recover_{team2}_vs_{team1}"]
                
                # Determine the opposing team from the main team's DataFrame
                opposing_team = df_main[df_main['team'] != details['team']]['team'].unique()[0]

                flag_main = team_flags.get(details['team'], '')
                flag_opp  = team_flags.get(opposing_team, '')

                # Process match events for table display
                match_data = process_match(match_id, matches_dict)
                if match_data:
                    match_tables[match_id] = match_data
                
                # Save match info for later
                match_info[match_id] = {
                    'main_team': details['team'],
                    'opposing_team': opposing_team,
                    'match_name': f"{details['team']} vs {opposing_team}"
                }
                
                # Extract colors for each team with special handling for the specified team
                if details['team'] == team_name:
                    main_color = team_color
                elif isinstance(details.get('color'), dict):
                    main_color = details['color'].get('main', '#0000FF')  # default blue if not provided
                else:
                    main_color = details.get('color', '#0000FF')
                
                if opposing_team == team_name:
                    opposing_color = team_color
                else:
                    # Look up color for opposing team across all provided dictionaries
                    opposing_color = 'gray'  # default
                    for _, _, match_dict in team_dicts:
                        for _, d in match_dict.items():
                            if d['team'] == opposing_team:
                                if isinstance(d.get('color'), dict):
                                    opposing_color = d['color'].get('opposing', 'gray')
                                else:
                                    opposing_color = d.get('color', 'gray')
                                break
                
                # Store team colors for HTML styling
                team_colors[details['team']] = main_color
                team_colors[opposing_team] = opposing_color
                
                # Process data for both teams using the helper function
                main_team_data = preprocess_game_data(df_main, details['team'])
                opposing_team_data = preprocess_game_data(df_main, opposing_team)
                
                if main_team_data is None or opposing_team_data is None:
                    print(f"Missing team data for Match ID {match_id}. Skipping...")
                    continue
                
                # Fetch event data for the match
                events = sb.events(match_id=match_id)
                if events.empty:
                    print(f"No event data found for Match ID {match_id}. Skipping...")
                    continue
                
                # Convert timestamps and filter events for valid periods (1,2,3,4)
                events['timestamp'] = pd.to_datetime(events['timestamp'], errors='coerce')
                events = events[events['period'].isin(periods)]
                match_start_time = events['timestamp'].min()
                events['event_time_minutes'] = (events['timestamp'] - match_start_time).dt.total_seconds() / 60
                mask = events['type'] == 'Own Goal Against'

                if mask.any():
                    # 2) Rename them to plain “Own Goal”
                    events.loc[mask, 'type'] = 'Own Goal'
                    
                    # 3) Make sure your team‐column is a clean string
                    events['team'] = events['team'].astype(str).str.strip()
                    
                    # 4) Vectorized swap: if original team == primary_team → opposing_team, 
                    teams = events.loc[mask, 'team']
                    events.loc[mask, 'team'] = np.where(
                        teams == details['team'],
                        opposing_team,
                        np.where(
                            teams == opposing_team,
                            details['team'],
                            teams
                        )
                    )
                # Create a folder for match outputs
                match_folder = os.path.join(output_dir, f"match_{match_id}")
                os.makedirs(match_folder, exist_ok=True)
                
                # Determine which periods have data for either team or events
                available_periods = []
                for period in periods:
                    main_period_data = main_team_data[main_team_data['period'] == period]
                    opp_period_data = opposing_team_data[opposing_team_data['period'] == period]
                    period_events = events[events['period'] == period]
                    if (not main_period_data.empty or not opp_period_data.empty or not period_events.empty):
                        available_periods.append(period)
                
                if not available_periods:
                    print(f"No valid period data found for Match ID {match_id}. Skipping...")
                    continue
                
                print(f"Available periods for Match ID {match_id}: {available_periods}")
                match_info[match_id]['available_periods'] = available_periods
                match_info[match_id]['filename'] = f"match_{match_id}_all_periods.html"
                
                # Create one figure for the entire match with all available periods
                fig = go.Figure()
                
                # Dictionary to store trace indices for each period
                period_trace_indices = {period: {"main": None, "opposing": None, "events": []} for period in available_periods}
                
                # Add traces for the main team's data for each available period
                for period in available_periods:
                    period_data = main_team_data[main_team_data['period'] == period]
                    if not period_data.empty:
                        fig.add_trace(go.Scatter(
                            x=period_data['game_time_minutes'],
                            y=period_data['time_to_recover'],
                            mode='lines+markers',
                            name=f"{flag_main} {details['team']} - Period {period}",
                            line=dict(color=main_color, width=2.5),
                            marker=dict(symbol='circle', size=8),
                            hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.0f} min<br>Recovery: %{y:.2f} sec<extra></extra>',
                            legendgroup=f"{details['team']}_p{period}",
                            showlegend=True,
                            visible=(period == available_periods[0])
                        ))
                        # Store the trace index for this period's main team data
                        period_trace_indices[period]["main"] = len(fig.data) - 1
                
                # Add traces for the opposing team's data for each available period
                for period in available_periods:
                    period_data = opposing_team_data[opposing_team_data['period'] == period]
                    if not period_data.empty:
                        fig.add_trace(go.Scatter(
                            x=period_data['game_time_minutes'],
                            y=period_data['time_to_recover'],
                            mode='lines+markers',
                            name=f"{flag_opp} {opposing_team} - Period {period}",
                            line=dict(color=opposing_color, width=2.5),
                            marker=dict(symbol='square', size=8),
                            hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.0f} min<br>Recovery: %{y:.2f} sec<extra></extra>',
                            legendgroup=f"{opposing_team}_p{period}",
                            showlegend=True,
                            visible=(period == available_periods[0])
                        ))
                        # Store the trace index for this period's opposing team data
                        period_trace_indices[period]["opposing"] = len(fig.data) - 1
                
                # Add event markers for each available period and team
                for period in available_periods:
                    period_events = events[events['period'] == period]
                    
                    key_events = {
                        "Goal": period_events[(period_events['type'] == 'Shot') & (period_events['shot_outcome'] == 'Goal')],
                        "Own Goal": period_events[period_events['type'] == 'Own Goal'],
                        "Substitution": period_events[period_events['type'] == 'Substitution'],
                        "Tactical Shift": period_events[period_events['type'] == 'Tactical Shift'],
                        "Injury": period_events[period_events['type'] == 'Injury Stoppage'],
                        "Yellow Card": period_events[(period_events['type'] == 'Foul Committed') & (period_events['foul_committed_card'] == 'Yellow Card')],
                        "Red Card": period_events[(period_events['type'] == 'Foul Committed') & (period_events['foul_committed_card'] == 'Red Card')]
                    }
                    
                    main_period_data = main_team_data[main_team_data['period'] == period]
                    opp_period_data = opposing_team_data[opposing_team_data['period'] == period]
                    
                    for event_type, events_df in key_events.items():
                        if events_df.empty:
                            continue
                        
                        events_df = events_df.copy()
                        events_df['player_name'] = events_df.apply(extract_player_name, axis=1)
                        
                        # For tactical shifts, extract the new formation
                        if event_type == "Tactical Shift":
                            events_df['new_formation'] = events_df.apply(
                                lambda row: row.get('tactics', {}).get('formation', 'Unknown'), axis=1
                            )
                        
                        if "team" in events_df.columns:
                            main_events = events_df[events_df['team'].str.lower() == details['team'].lower()]
                            opp_events = events_df[events_df['team'].str.lower() == opposing_team.lower()]
                            other_events = events_df[~events_df['team'].str.lower().isin([details['team'].lower(), opposing_team.lower()])]
                        else:
                            main_events = events_df.copy()
                            opp_events = pd.DataFrame()
                            other_events = pd.DataFrame()
                        
                        emoji = emoji_map.get(event_type, "❓")
                        
                        def create_hover_text(row, base_team, period, team_label):
                            # Special handling for Own Goal events
                            if event_type == "Own Goal":
                                # Get the actual team that scored the own goal if available
                                if 'own_goal_by_team' in row:
                                    own_goal_by_team = row['own_goal_by_team']
                                    own_goal_by_player = row['own_goal_by_player']
                                    benefiting_team = row['own_goal_benefiting_team']
                                else:
                                    # Fallback if we don't have the special fields
                                    own_goal_by_team = "Unknown"
                                    own_goal_by_player = row['player_name']
                                    benefiting_team = team_label.split(' ')[-1]
                                
                                # Get flag emojis for the teams
                                own_goal_flag = team_flags.get(own_goal_by_team, "")
                                benefiting_flag = team_flags.get(benefiting_team, "")
                                
                                return (
                                    f"<b>OWN GOAL 🚫⚽</b><br>" +
                                    f"Time: {int(row['event_time_minutes'])} min<br>" +
                                    f"Own goal by: {own_goal_by_player}<br>" +
                                    f"Point awarded to: {benefiting_flag} {benefiting_team}<br>" +
                                    f"Period: {period}"
                                )
                            
                            # Regular handling for other event types
                            base_text = (
                                f"<b>{event_type}</b><br>" +
                                f"Time: {int(row['event_time_minutes'])} min<br>" +
                                f"Team: {team_label}<br>" +
                                f"Period: {period}"
                            )
                            
                            if event_type == "Tactical Shift":
                                formation_text = f"<br>New Formation: {row['new_formation']}"
                                return base_text + formation_text
                            else:
                                # For all other events, show the player name
                                player_text = f"<br>Player: {row['player_name']}"
                                return base_text + player_text
                        
                        # Plot events on the main team's curve
                        if not main_events.empty and not main_period_data.empty:
                            # Special handling for own goals to ensure we can access the additional fields
                            if event_type == "Own Goal":
                                hover_texts = []
                                for _, ev_row in main_events.iterrows():
                                    hover_texts.append(create_hover_text(ev_row, main_period_data, period, f"{flag_main} {details['team']}"))
                            else:
                                hover_texts = main_events.apply(
                                    lambda row: create_hover_text(row, main_period_data, period, f"{flag_main} {details['team']}"), 
                                    axis=1
                                )
                            
                            fig.add_trace(go.Scatter(
                                x=main_events['event_time_minutes'],
                                y=main_events['event_time_minutes'].apply(lambda t: get_recovery_value(t, main_period_data)),
                                mode='markers+text',
                                marker=dict(color=main_color, size=10, opacity=0.7, line=dict(color='black', width=1)),
                                text=[emoji] * len(main_events),
                                textposition='middle center',
                                name=f"{emoji} {event_type} ({flag_main} {details['team']})",
                                hovertemplate='%{hovertext}<extra></extra>',
                                hovertext=hover_texts,
                                legendgroup=f"{details['team']}_p{period}",
                                showlegend=True,
                                visible=(period == available_periods[0])
                            ))
                            # Store this event trace index for this period
                            period_trace_indices[period]["events"].append(len(fig.data) - 1)
                        
                        # Plot events on the opposing team's curve
                        if not opp_events.empty and not opp_period_data.empty:
                            # Special handling for own goals
                            if event_type == "Own Goal":
                                hover_texts = []
                                for _, ev_row in opp_events.iterrows():
                                    hover_texts.append(create_hover_text(ev_row, opp_period_data, period, f"{flag_opp} {opposing_team}"))
                            else:
                                hover_texts = opp_events.apply(
                                    lambda row: create_hover_text(row, opp_period_data, period, f"{flag_opp} {opposing_team}"), 
                                    axis=1
                                )
                            
                            fig.add_trace(go.Scatter(
                                x=opp_events['event_time_minutes'],
                                y=opp_events['event_time_minutes'].apply(lambda t: get_recovery_value(t, opp_period_data)),
                                mode='markers+text',
                                marker=dict(color=opposing_color, size=10, opacity=0.7, line=dict(color='black', width=1)),
                                text=[emoji] * len(opp_events),
                                textposition='middle center',
                                name=f"{emoji} {event_type} ({flag_opp} {opposing_team})",
                                hovertemplate='%{hovertext}<extra></extra>',
                                hovertext=hover_texts,
                                legendgroup=f"{opposing_team}_p{period}",
                                showlegend=True,
                                visible=(period == available_periods[0])
                            ))
                            # Store this event trace index for this period
                            period_trace_indices[period]["events"].append(len(fig.data) - 1)
                
                # Create dropdown buttons
                buttons = []
                
                # "All Periods" button: makes all traces visible.
                all_periods_visible = [True] * len(fig.data)
                buttons.append(
                    dict(
                        args=[
                            {'visible': all_periods_visible},
                            {'title': {
                                'text': f"Time to Recover Ball - {flag_main} {details['team']} vs {flag_opp} {opposing_team} (All Periods)",
                                'font': {'size': 20, 'family': 'Arial', 'color': 'black'},
                                'y': 0.95
                            }}
                        ],
                        label="All Periods",
                        method="update"
                    )
                )
                
                # Now add one button for each available period
                for period in available_periods:
                    # Set all traces to invisible first
                    period_visible = [False] * len(fig.data)
                    
                    # Make the main team trace for this period visible
                    if period_trace_indices[period]["main"] is not None:
                        period_visible[period_trace_indices[period]["main"]] = True
                    
                    # Make the opposing team trace for this period visible
                    if period_trace_indices[period]["opposing"] is not None:
                        period_visible[period_trace_indices[period]["opposing"]] = True
                    
                    # Make all event traces for this period visible
                    for event_idx in period_trace_indices[period]["events"]:
                        period_visible[event_idx] = True
                    
                    buttons.append(
                        dict(
                            args=[
                                {'visible': period_visible},
                                {'title': {
                                    'text': f"Time to Recover Ball - {flag_main} {details['team']} vs {flag_opp} {opposing_team} (Period {period})",
                                    'font': {'size': 20, 'family': 'Arial', 'color': 'black'},
                                    'y': 0.95
                                }}
                            ],
                            label=f"Period {period}",
                            method="update"
                        )
                    )
                
                # Set the active button index so that Period 1 is shown by default.
                # Since the "All Periods" button is at index 0, the Period 1 button is at index 1.
                active_button = 1 if len(buttons) > 1 else 0
                
                updatemenus = [
                    dict(
                        active=active_button,  # Default active button corresponds to Period 1.
                        buttons=buttons,
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.15,
                        yanchor="top",
                        name="period_selector"
                    )
                ]
                
                fig.update_layout(
                    title={
                        'text': f"Time to Recover Ball - {flag_main} {details['team']} vs {flag_opp}{opposing_team} (Period {available_periods[0]})",
                        'font': {'size': 20, 'family': 'Arial', 'color': 'black'},
                        'y': 0.95
                    },
                    xaxis_title={
                        'text': "Game Time (minutes)",
                        'font': {'size': 16, 'family': 'Arial'}
                    },
                    yaxis_title={
                        'text': "Time to Recover (seconds)",
                        'font': {'size': 16, 'family': 'Arial'}
                    },
                    updatemenus=updatemenus,
                    legend_title="Teams and Events",
                    hovermode="closest",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Arial", size=14),
                    legend=dict(
                        bordercolor="Black",
                        borderwidth=1,
                        bgcolor="white",
                        font=dict(size=12),
                        groupclick="toggleitem",
                        traceorder="grouped"
                    ),
                    width=1000,
                    height=600,
                    margin=dict(l=80, r=80, t=100, b=80)
                )
                
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=2,
                    linecolor='black'
                )
                
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=2,
                    linecolor='black'
                )
                
                # Create match event table HTML if available
                match_table_html = ""
                if match_id in match_tables:
                    match_table_html = generate_match_table_html(match_tables[match_id], team_colors)
                
                # Add toggle for lines visualization and custom JavaScript for synchronized toggling
                custom_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Function to wait for Plotly to be fully loaded
    function waitForPlotly() {
        if (typeof Plotly !== 'undefined' && document.querySelector('.plotly')) {
            setupSynchronizedToggle();
            setupLinesToggle();
            setupFormationComparison(); // Add our new function for formation comparison
        } else {
            setTimeout(waitForPlotly, 100);
        }
    }
    
    // Function to set up synchronized toggle for legend groups
    function setupSynchronizedToggle() {
        const gd = document.querySelector('.plotly');
        if (!gd || !gd.data) return;
        const origLegendClick = gd._context.legendClick;
        gd._context.legendClick = function(curveNumber) {
            const clickedTrace = gd.data[curveNumber];
            if (!clickedTrace || !clickedTrace.legendgroup) return origLegendClick.call(this, curveNumber);
            const legendGroup = clickedTrace.legendgroup;
            const isVisible = clickedTrace.visible === 'legendonly' ? true : 'legendonly';
            const tracesToUpdate = [];
            const visibilityValues = [];
            gd.data.forEach((trace, i) => {
                if (trace.legendgroup === legendGroup) {
                    tracesToUpdate.push(i);
                    visibilityValues.push(isVisible);
                }
            });
            if (tracesToUpdate.length > 0) {
                Plotly.restyle(gd, {'visible': visibilityValues}, tracesToUpdate);
                return false;
            }
            return origLegendClick.call(this, curveNumber);
        };
    }
    
    // Function to set up lines toggle button
    function setupLinesToggle() {
        const gd = document.querySelector('.plotly');
        if (!gd || !gd.data) return;
        
        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.innerText = 'Toggle Lines/Points';
        toggleButton.style.position = 'absolute';
        toggleButton.style.top = '20px';
        toggleButton.style.right = '20px';
        toggleButton.style.zIndex = '999';
        toggleButton.style.padding = '8px 16px';
        toggleButton.style.backgroundColor = '#f8f9fa';
        toggleButton.style.border = '1px solid #ddd';
        toggleButton.style.borderRadius = '4px';
        toggleButton.style.cursor = 'pointer';
        
        // Add button to the plot container
        const plotlyContainer = document.querySelector('.plotly-container');
        if (plotlyContainer) {
            plotlyContainer.style.position = 'relative';
            plotlyContainer.appendChild(toggleButton);
            
            // Toggle between lines+markers and markers only
            let showLines = true;
            toggleButton.addEventListener('click', function() {
                showLines = !showLines;
                
                const mainTraces = [];
                const mainModes = [];
                
                gd.data.forEach((trace, i) => {
                    if (trace.mode && trace.mode.includes('lines')) {
                        mainTraces.push(i);
                        mainModes.push(showLines ? 'lines+markers' : 'markers');
                    }
                });
                
                if (mainTraces.length > 0) {
                    Plotly.restyle(gd, {'mode': mainModes}, mainTraces);
                }
            });
        }
    }
    
    // Function to set up formation comparison functionality
    function setupFormationComparison() {
        document.querySelectorAll('.formation-details').forEach(function(el) {
            // Style the formation details elements
            el.style.cursor = 'pointer';
            el.style.color = '#0288d1';
            el.style.textDecoration = 'underline';
            
            // Add click event listener
            el.addEventListener('click', function() {
                try {
                    // Get data from attributes
                    const lineup = JSON.parse(this.getAttribute('data-lineup').replace(/&quot;/g, '"'));
                    const prevLineup = this.hasAttribute('data-prev-lineup') ? 
                      JSON.parse(this.getAttribute('data-prev-lineup').replace(/&quot;/g, '"')) : null;
                    const teamColor = this.getAttribute('data-team-color');
                    const teamName = this.getAttribute('data-team-name');
                    const formationName = this.getAttribute('data-formation') || 'Unknown';
                    const prevFormationName = this.getAttribute('data-prev-formation') || null;
                    
                    // Create modal HTML
                    let modalHTML = `
                      <div id="formationModal" style="position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); z-index:1000; display:flex; justify-content:center; align-items:center;">
                        <div style="background:white; border-radius:8px; width:90%; max-width:900px; max-height:90vh; overflow:auto; box-shadow:0 4px 15px rgba(0,0,0,0.2); position:relative;">
                          <div style="display:flex; justify-content:space-between; align-items:center; padding:15px; border-bottom:2px solid ${teamColor};">
                            <h3 style="margin:0; color:${teamColor};">${teamName} Formation Details</h3>
                            <button id="closeFormationModal" style="background:none; border:none; font-size:24px; cursor:pointer;">×</button>
                          </div>
                          <div style="padding:20px;">`;
                    
                    // If we have a previous formation to compare with
                    if (prevLineup) {
                      modalHTML += `
                        <div style="text-align:center; margin-bottom:15px;">
                          <h4 style="margin:5px 0;">Formation Change</h4>
                          <div style="font-size:18px;">
                            <span style="color:#666;">${prevFormationName}</span>
                            <span style="margin:0 10px;">→</span>
                            <span style="font-weight:bold;">${formationName}</span>
                          </div>
                        </div>
                        
                        <div style="display:flex; gap:20px; margin-bottom:20px;">
                          <!-- Left side: Previous Formation - UPDATED TO LIGHT GRAY -->
                          <div style="flex:1; border:1px solid #eee; border-radius:8px; padding:10px; background-color:#f9f9f9;">
                            <h4 style="text-align:center; margin-top:0; color:#777;">Previous Formation</h4>
                            <div style="overflow-x:auto;">
                              <table style="width:100%; border-collapse:collapse;">
                                <thead>
                                  <tr style="background-color:#e0e0e0; color:#555;">
                                    <th style="text-align:left; padding:10px;">Position</th>
                                    <th style="text-align:center; padding:10px; width:80px;">#</th>
                                    <th style="text-align:left; padding:10px;">Player</th>
                                  </tr>
                                </thead>
                                <tbody>
                      `;
                      
                      // Group previous players by position type
                      const prevPlayers = groupPlayersByPosition(prevLineup);
                      
                      // Add previous players to the table - using gray color for previous formation
                      const prevColor = "#aaaaaa"; // Light gray for previous formation
                      modalHTML += renderPlayerGroup(prevPlayers.gk, 'Goalkeepers', prevColor, {}, true);
                      modalHTML += renderPlayerGroup(prevPlayers.defenders, 'Defenders', prevColor, {}, true);
                      modalHTML += renderPlayerGroup(prevPlayers.midfielders, 'Midfielders', prevColor, {}, true);
                      modalHTML += renderPlayerGroup(prevPlayers.forwards, 'Forwards', prevColor, {}, true);
                      modalHTML += renderPlayerGroup(prevPlayers.others, 'Others', prevColor, {}, true);
                      
                      modalHTML += `
                                </tbody>
                              </table>
                            </div>
                          </div>
                          
                          <!-- Right side: New Formation -->
                          <div style="flex:1; border:1px solid ${teamColor}; border-radius:8px; padding:10px;">
                            <h4 style="text-align:center; margin-top:0; color:${teamColor};">Current Formation</h4>
                            <div style="overflow-x:auto;">
                              <table style="width:100%; border-collapse:collapse;">
                                <thead>
                                  <tr style="background-color:${teamColor}; color:white;">
                                    <th style="text-align:left; padding:10px;">Position</th>
                                    <th style="text-align:center; padding:10px; width:80px;">#</th>
                                    <th style="text-align:left; padding:10px;">Player</th>
                                  </tr>
                                </thead>
                                <tbody>
                      `;
                      
                      // Find changes between lineups
                      const changes = identifyChanges(prevLineup, lineup);
                      
                      // Group current players by position type
                      const currentPlayers = groupPlayersByPosition(lineup);
                      
                      // Add current players to the table with changes highlighted
                      modalHTML += renderPlayerGroup(currentPlayers.gk, 'Goalkeepers', teamColor, changes, false);
                      modalHTML += renderPlayerGroup(currentPlayers.defenders, 'Defenders', teamColor, changes, false);
                      modalHTML += renderPlayerGroup(currentPlayers.midfielders, 'Midfielders', teamColor, changes, false);
                      modalHTML += renderPlayerGroup(currentPlayers.forwards, 'Forwards', teamColor, changes, false);
                      modalHTML += renderPlayerGroup(currentPlayers.others, 'Others', teamColor, changes, false);
                      
                      modalHTML += `
                                </tbody>
                              </table>
                            </div>
                          </div>
                        </div>
                        
                        <div style="margin-top:15px; background-color:#f8f9fa; padding:10px; border-radius:8px;">
                          <h4 style="margin-top:0;">Legend</h4>
                          <div style="display:flex; flex-wrap:wrap; gap:15px;">
                            <div style="display:flex; align-items:center;">
                              <span style="display:inline-block; width:16px; height:16px; background-color:#FFEB3B; margin-right:5px;"></span>
                              Position changed
                            </div>
                            <div style="display:flex; align-items:center;">
                              <span style="display:inline-block; width:16px; height:16px; background-color:#4CAF50; margin-right:5px;"></span>
                              New player
                            </div>
 
                          </div>
                        </div>
                      `;
                    } else {
                      // Single formation display (for kick-off formations)
                      modalHTML += `
                        <h4 style="text-align:center; margin-top:0;">Starting formation: ${formationName}</h4>
                        <table style="width:100%; border-collapse:collapse;">
                          <caption style="caption-side:top; font-size:18px; padding:8px 0;">
                            ${teamName}
                          </caption>
                          <thead>
                            <tr style="background-color:${teamColor}; color:white;">
                              <th style="text-align:left; padding:10px;">Position</th>
                              <th style="text-align:center; padding:10px; width:80px;">#</th>
                              <th style="text-align:left; padding:10px;">Player</th>
                            </tr>
                          </thead>
                          <tbody>
                      `;
                      
                      // Group players by position type
                      const players = groupPlayersByPosition(lineup);
                      
                      // Add players to the table
                      modalHTML += renderPlayerGroup(players.gk, 'Goalkeepers', teamColor, {}, false);
                      modalHTML += renderPlayerGroup(players.defenders, 'Defenders', teamColor, {}, false);
                      modalHTML += renderPlayerGroup(players.midfielders, 'Midfielders', teamColor, {}, false);
                      modalHTML += renderPlayerGroup(players.forwards, 'Forwards', teamColor, {}, false);
                      modalHTML += renderPlayerGroup(players.others, 'Others', teamColor, {}, false);
                      
                      modalHTML += `
                          </tbody>
                        </table>
                      `;
                    }
                    
                    // Close the HTML
                    modalHTML += `
                          </div>
                        </div>
                      </div>
                    `;
                    
                    // Add modal to the document
                    const modalContainer = document.createElement('div');
                    modalContainer.innerHTML = modalHTML;
                    document.body.appendChild(modalContainer);
                    
                    // Add close event
                    document.getElementById('closeFormationModal').addEventListener('click', function() {
                      document.body.removeChild(modalContainer);
                    });
                    
                    // Close on background click
                    document.getElementById('formationModal').addEventListener('click', function(e) {
                      if (e.target.id === 'formationModal') {
                        document.body.removeChild(modalContainer);
                      }
                    });
                    
                } catch (error) {
                    console.error('Error showing formation details:', error);
                    alert('There was an error displaying the formation details.');
                }
            });
        });
    }

    // Helper function to group players by position
    function groupPlayersByPosition(lineup) {
      let gk = [];
      let defenders = [];
      let midfielders = [];
      let forwards = [];
      let others = [];
      
      if (!lineup) return { gk, defenders, midfielders, forwards, others };
      
      lineup.forEach(player => {
        const pos = player.position?.name || '';
        if (pos.includes('Goalkeeper')) gk.push(player);
        else if (pos.includes('Back') || pos.includes('Center Back')) defenders.push(player);
        else if (pos.includes('Midfield') || pos.includes('Wing')) midfielders.push(player);
        else if (pos.includes('Forward') || pos.includes('Striker')) forwards.push(player);
        else others.push(player);
      });
      
      return { gk, defenders, midfielders, forwards, others };
    }

    // Helper function to identify changes between two lineups
    function identifyChanges(oldLineup, newLineup) {
      const changes = {
        positionChanged: {},  // Players who changed positions
        newPlayers: {},       // Players who are new
        removedPlayers: {}    // Players who were removed
      };
      
      if (!oldLineup || !newLineup) return changes;
      
      // Create maps for easy lookup
      const oldPlayerMap = {};
      const newPlayerMap = {};
      
      oldLineup.forEach(player => {
        const playerId = player.player?.id || player.player?.name;
        if (playerId) {
          oldPlayerMap[playerId] = player;
        }
      });
      
      newLineup.forEach(player => {
        const playerId = player.player?.id || player.player?.name;
        if (playerId) {
          newPlayerMap[playerId] = player;
          
          // Check if player existed in old lineup
          if (oldPlayerMap[playerId]) {
            // Check if position changed
            const oldPos = oldPlayerMap[playerId].position?.name;
            const newPos = player.position?.name;
            
            if (oldPos !== newPos) {
              changes.positionChanged[playerId] = {
                oldPosition: oldPos,
                newPosition: newPos
              };
            }
          } else {
            // This is a new player
            changes.newPlayers[playerId] = player;
          }
        }
      });
      
      // 
      
      return changes;
    }

    // Helper function to render a group of players in the table
    function renderPlayerGroup(players, sectionName, teamColor, changes, isPrevious) {
      if (players.length === 0) return '';
      
      // For previous formation, use a consistent light gray for section headers
      // For current formation, use team color with opacity
      const sectionBgColor = isPrevious ? "#f0f0f0" : `${teamColor}20`;
      
      let html = `
        <tr style="background-color:${sectionBgColor};">
          <td colspan="3" style="padding:8px; font-weight:bold; color:${isPrevious ? "#666666" : "#333333"};">${sectionName}</td>
        </tr>
      `;
      
      players.forEach((player, i) => {
        // Use different background colors for previous and current formation rows
        const bgColor = isPrevious ? (i % 2 === 0 ? '#f2f2f2' : '#e9e9e9') : (i % 2 === 0 ? '#f9f9f9' : 'white');
        const playerId = player.player?.id || player.player?.name;
        let highlightStyle = '';
        let statusIcon = '';
        
        if (!isPrevious && playerId) {
          // Highlight changes in the current formation
          if (changes.positionChanged && changes.positionChanged[playerId]) {
            highlightStyle = 'background-color:#FFEB3B50;';
            statusIcon = `<span title="Position changed: ${changes.positionChanged[playerId].oldPosition} → ${changes.positionChanged[playerId].newPosition}" style="margin-left:5px;">↔️</span>`;
          } else if (changes.newPlayers && changes.newPlayers[playerId]) {
            highlightStyle = 'background-color:#4CAF5050;';
            statusIcon = '<span title="New player" style="margin-left:5px;">➕</span>';
          }
        } else if (isPrevious && playerId && changes.removedPlayers && changes.removedPlayers[playerId]) {
          // Highlight removed players in the previous formation
          highlightStyle = 'background-color:#F4433650;';
          statusIcon = '<span title="Removed" style="margin-left:5px;">❌</span>';
        }
        
        // Use different text colors for previous and current formations
        const textColor = isPrevious ? "#666666" : "#333333";
        
        html += `
          <tr style="background-color:${bgColor}; ${highlightStyle}">
            <td style="padding:8px; border-bottom:1px solid #eee; color:${textColor};">${player.position?.name || 'Unknown'}</td>
            <td style="padding:8px; border-bottom:1px solid #eee; text-align:center; font-weight:bold; color:${textColor};">${player.jersey_number || '-'}</td>
            <td style="padding:8px; border-bottom:1px solid #eee; color:${textColor};">${player.player?.name || 'Unknown Player'} ${statusIcon}</td>
          </tr>
        `;
      });
      
      return html;
    }
    
    waitForPlotly();
});
</script>
"""
                
                # Create CSS for the match table
                table_css = """
                <style>
                /* Updated CSS for match table with team-colored backgrounds */
.match-table-container {
    margin-top: 30px;
    font-family: Arial, sans-serif;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
}

.match-table-container h2 {
    margin-bottom: 15px;
    font-size: 22px;
    text-align: center;
    padding: 15px;
    background-color: #f8f9fa;
    border-bottom: 1px solid #eee;
}

.match-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 0;
    border: none;
}

.match-table th, .match-table td {
    padding: 10px 12px;
    text-align: center;
    border: 1px solid #eee;
}

.match-table th {
    background-color: #2c3e50;
    color: white;
    font-weight: bold;
    text-transform: uppercase;
    font-size: 14px;
}

/* Ensure text is clearly visible on colored backgrounds */
.match-table td {
    color: #333;
    font-weight: normal;
}

/* Make headers stand out more */
.period-header {
    background-color: #34495e !important;
    color: white;
    font-size: 16px;
    text-align: left;
}

.period-header td {
    padding: 8px 12px;
    color: white !important;
}

/* Special event formatting */
.goal-event {
    font-weight: bold;
}

.tactical-shift {
    font-weight: bold;
    cursor: pointer;
}

/* Additional styling for the team pills */
.match-table td span {
    display: inline-block;
    margin: 0 auto;
    white-space: nowrap;
}

/* Make hover effect subtle to not hide team colors */
.match-table tr:hover {
    filter: brightness(1.05);
    transition: all 0.2s ease;
}

/* Adjust formation details styling */
.formation-details {
    cursor: pointer;
    text-decoration: underline;
    color: #2980b9;
}

.formation-details:hover {
    color: #3498db;
}
                </style>
                """
                
                html_file = os.path.join(match_folder, f"match_{match_id}_all_periods.html")
                html_content = pio.to_html(fig, full_html=True, include_plotlyjs=True)
                
                # Add our custom elements to the HTML
                # First, add the CSS, JS, and wrap the plot in a container
                html_content = html_content.replace('<body>', f'<body>\n{table_css}')
                html_content = html_content.replace('<div id="', '<div class="plotly-container"><div id="')
                # Use formatted string with actual content, not literal variable names
                html_content = html_content.replace('</div>\n</body>', f'</div></div>\n{match_table_html}\n{custom_js}\n</body>')
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"Interactive plot with match table saved for Match ID {match_id} at {html_file}")
                
            except Exception as e:
                print(f"Error processing Match ID {match_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Create an index HTML page linking to all match plots if any were processed
    if match_info:
        # Create a unified index page for all teams
        index_html_title = "Football Match Recovery Analysis - " + ", ".join(all_teams)
        
        sorted_match_ids = sorted(match_info.keys(),
                                 key=lambda x: (match_info[x]['main_team'], match_info[x]['opposing_team']))
        
        index_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{index_html_title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    border-radius: 5px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .match-list {{
                    list-style-type: none;
                    padding: 0;
                }}
                .match-item {{
                    padding: 15px;
                    border-bottom: 1px solid #eee;
                    transition: background-color 0.2s;
                    cursor: pointer;
                }}
                .match-item:hover {{
                    background-color: #f5f5f5;
                }}
                .match-item.active {{
                    background-color: #e6f7ff;
                    border-left: 3px solid #1890ff;
                }}
                .match-link {{
                    display: block;
                    color: #0366d6;
                    text-decoration: none;
                    font-size: 18px;
                }}
                .match-info {{
                    margin-top: 5px;
                    font-size: 14px;
                    color: #666;
                }}
                .period-tag {{
                    display: inline-block;
                    background-color: #e1e4e8;
                    padding: 2px 8px;
                    border-radius: 10px;
                    margin-right: 5px;
                    font-size: 12px;
                }}
                iframe {{
                    width: 100%;
                    height: 800px; /* Increased height to show plot and table */
                    border: none;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{index_html_title}</h1>
                    <p>Select a match to view detailed recovery analysis</p>
                </div> 
                <div class="match-selector">
                    <ul class="match-list">
        """
        
        for match_id in sorted_match_ids:
            m_info = match_info[match_id]
            p = m_info['main_team']
            o = m_info['opposing_team']
            flag_p = team_flags.get(p, "")
            flag_o = team_flags.get(o, "")
            match_name = m_info['match_name']
            main_team_str = m_info['main_team']
            opposing_team_str = m_info['opposing_team']
            periods_str = ' '.join([f'<span class="period-tag">Period {p}</span>' for p in m_info['available_periods']])
            
            index_html += f"""
                        <li class="match-item" onclick="loadMatch('{match_id}')">
            <a href="#" class="match-link">
                {flag_p} {p} vs {flag_o} {o}
            </a>
                            <div class="match-info">
                                {main_team_str} vs {opposing_team_str} | Available periods: {periods_str}
                            </div>
                        </li>
            """
        
        index_html += """
                    </ul>
                </div>
                <div id="match-viewer">
                    <iframe id="match-frame" src="" style="display:none;"></iframe>
                    <div id="initial-message" style="text-align:center; padding:50px;">
                        <h2>Click on a match from the list above to view it</h2>
                    </div>
                </div>
            </div>
            <script>
                function loadMatch(matchId) {
                    const matchItems = document.querySelectorAll('.match-item');
                    matchItems.forEach(item => {
                        item.classList.remove('active');
                    });
                    const clickedItem = document.querySelector(`.match-item[onclick="loadMatch('${matchId}')"]`);
                    if (clickedItem) {
                        clickedItem.classList.add('active');
                    }
                    document.getElementById('initial-message').style.display = 'none';
                    const iframe = document.getElementById('match-frame');
                    iframe.style.display = 'block';
                    iframe.src = `match_${matchId}/match_${matchId}_all_periods.html`;
                }
                document.addEventListener('DOMContentLoaded', function() {
                    const links = document.querySelectorAll('.match-link');
                    links.forEach(link => {
                        link.addEventListener('click', function(e) {
                            e.preventDefault();
                        });
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Create a unified index file for all teams
        # index_file = os.path.join(output_dir, f"match_index_all_teams.html")
    #     with open(index_file, 'w', encoding='utf-8') as f:
    #         f.write(index_html)
        
    #     print(f"\nUnified index page created at: {index_file}")
    # else:
    #     print("No match data was processed successfully. Index page was not created.")
    
    print("All match plots have been generated and saved!")

generate_match_plots(
    ("Argentina", "#89CFF0", arg_matches), 
    ("France", "#0055A4", fr_matches),
    output_dir="visuals"
)

