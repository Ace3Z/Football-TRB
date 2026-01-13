from statsbombpy import sb
from team_colors import color_manager


# Fetch the list of competitions
competitions = sb.competitions()

# Filter for World Cup competitions and make a copy to safely modify
world_cup = competitions[competitions['competition_name'].str.contains("World Cup", case=False, na=False)].copy()

# Now it's safe to assign new columns
world_cup['year'] = world_cup['season_name'].str.extract(r'(\d{4})')

# Display the result
print(world_cup[['competition_name', 'competition_id', 'season_id', 'year']])


def get_matches_by_team(team_name, competition_id={"competition_id": 43, "season_id": 106}):
    """
    Get matches for a specific team in a competition and return as a dictionary
    with opponent details and national shirt colors.
    
    Args:
        team_name (str): Name of the team to get matches for
        competition_id (dict): Dictionary with competition_id and season_id
    
    Returns:
        dict: Dictionary with match_id as key and opponent info as value
    """
    # Define stage order for sorting
    stage_order = [
        "Group Stage",
        "Round of 16",
        "Quarter-finals",
        "Semi-finals",
        "Final"
    ]
    
    # Fetch matches for the competition
    matches = sb.matches(
        competition_id=competition_id['competition_id'],
        season_id=competition_id['season_id']
    )
    
    # Map stages to order for sorting
    matches['stage_order'] = matches['competition_stage'].map(
        lambda x: stage_order.index(x) if x in stage_order else -1
    )
    
    # Filter matches for the specific team
    team_matches = matches[
        (matches['home_team'] == team_name) | (matches['away_team'] == team_name)
    ].sort_values(by='stage_order')
    
    # Create the matches dictionary
    matches_dict = {}
    
    for _, match in team_matches.iterrows():
        match_id = match['match_id']
        
        # Determine the opponent
        if match['home_team'] == team_name:
            opponent = match['away_team']
        else:
            opponent = match['home_team']
        
        # Get national color for the opponent
        color = color_manager.get_color_for_team(opponent)
        
        # Add to dictionary
        matches_dict[match_id] = {
            'team': opponent,
            'color': color
        }
    
    return matches_dict


arg_matches = get_matches_by_team("Argentina")
fr_matches = get_matches_by_team("France")
