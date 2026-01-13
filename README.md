# 📊 Exploring Ball Recovery Times in Professional Football

This repository contains the code and visualizations for the bachelor thesis **"Exploring Ball Recovery Times in Professional Football: Insights and Patterns"** by Mahbod Tajdini and supervised by Dr. Mauricio Verano Merino, submitted to the Vrije Universiteit Amsterdam in 2025.

## 🎯 Objective
To analyze how key match events (goals, substitutions, injuries, cards, etc.) affect the time it takes for elite football teams to recover ball possession. The research focuses on Argentina and France during the 2022 FIFA World Cup, using open-access StatsBomb data.

## 🔍 Features
- **Rule-based ball recovery classification** based on detailed event taxonomy.
- **Statistical testing**:
  - Shapiro-Wilk Test for normality
  - Mann-Whitney U Test for group difference
  - Cliff’s Delta for effect size and direction
- **Interactive visualizations** using Plotly:
  - Normality plots (QQ + KDE)
  - Game-timeline plots with key match events and recovery durations
  - Statistical summary tables with effect interpretations
- **Custom dashboard** to assist coaching staff and analysts in tactical evaluation.


## 📦 Tech Stack
- Python
- Pandas
- NumPy
- SciPy
- Plotly
- StatsBombPy
- Matplotlib
- Seaborn
- Statsmodels


## 📂 Repository Contents

This repository is organized as follows:

- `output/`: This folder contains CSV files of the time to recover ball possession for each match
- `visuals/`: This folder contains the HTML files generated for the visual dashboard. The main dashboard is `dashboard.html`.
- `event_recovery_analysis/`: This folder contains the CSV file outputs of the statistical analysis performed
- `team_colors.py`: Python script that assigns colors for each team in the tournament
- `match_finder.py`: Python script used to find the match IDs for each match
- `trb.py`: Python script for calculating the time to recover the ball
- `stat_anlys.py`: Python script for conducting statistical analysis
- `vis_dashboard.py`: Python script that makes the components of the final visual dashboard
- `game_timeline.py`: Python script containing helper functions for the plots and the game time curve of events
- `Poster.pdf`: The poster presented during the poster day


## 🚀 How to Run the Code

To run the code and reproduce the analysis or generate new visualizations:

### 1. **Install Required Dependencies**

Ensure you have the required libraries installed. You can install them via pip:

```bash
pip install pandas numpy scipy plotly statsbombpy matplotlib seaborn statsmodels pycountry
```

### 2. **Execution Order**

The analysis must be run in the following sequence:

#### **Step 1: Calculate Time to Recover Ball Possession**
```bash
python trb.py
```
This script:
- Processes match events from StatsBomb data
- Identifies ball loss and recovery events
- Calculates time to recover possession for each team
- Outputs CSV files to the output/ directory
- Creates global DataFrames for subsequent analysis

#### **Step 2: Perform Statistical Analysis**
```bash
python stat_anlys.py
```
This script:
- Analyzes recovery times around different event types
- Performs normality tests (Shapiro-Wilk)
- Conducts Mann-Whitney U tests for group comparisons
- Calculates Cliff's Delta effect sizes
- Applies Benjamini-Hochberg correction for multiple testing
- Outputs results to the event_recovery_analysis/ directory

#### **Step 3: Generate Match Timeline Visualizations**
```bash
python game_timeline.py
```
This script:
- Creates interactive match timeline plots
- Overlays match events (goals, cards, substitutions) on recovery data
- Generates team formation details and tactical shift analysis
- Produces match-specific HTML files in the visuals/ directory

#### **Step 4: Create Complete Visual Dashboard**
```bash
python vis_dashboard.py
```
This script:
- Generates the comprehensive interactive dashboard
- Creates normality test visualizations (QQ plots and KDE plots)
- Produces team-specific analysis reports
- Generates the main dashboard.html file
- Integrates all previous analyses into a unified interface

### 3. **Expected Output**

After running all scripts, you will have:

- **CSV Data Files**: Raw recovery time data in output/
- **Statistical Results**: Analysis outputs in event_recovery_analysis/
- **Interactive Dashboard**: Main dashboard at visuals/dashboard.html
- **Match Visualizations**: Detailed match plots in visuals/match_*/

### 4. **Viewing Results**

Open visuals/dashboard.html in your web browser to access the complete interactive dashboard. From there, you can:

- Filter results by team, match, and period
- View normality test results and statistical analyses
- Access detailed match timeline visualizations
- Explore team-specific performance reports




## 🔧 Adapting to Other Teams or Tournaments

To replicate this analysis for other teams, matches, or tournaments, **two key files must be updated**:

---

### 1. `match_finder.py`

This file defines which matches to fetch using the StatsBomb API.

To include different teams or competitions:

- Locate the block of code where match IDs are filtered (typically via `competition_id`, `season_id`, or team names).
- Replace the existing values with those corresponding to your desired tournament/teams.

---

### 2. `team_colors.py`

This file maps display colors to each team for use in plots and dashboards.

To support new teams:

- Add a new entry to the color dictionary with the team name and the color.

