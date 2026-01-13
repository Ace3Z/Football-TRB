class TeamColorManager:
    """Manages color assignment based on national team shirt colors."""
    
    def __init__(self):
        self.national_colors = {
            'Argentina': '#87CEEB',  # Light blue and white stripes 
            'France': '#0055A4',     # Navy blue
            'Brazil': '#FFD700',     # Yellow
            'Netherlands': '#FF4500', # Orange
            'England': '#D3D3D3',    # Light gray 
            'Croatia': '#FF0000',    # Red and white checkered 
            'Morocco': '#C1272D',    # Red
            'Portugal': '#CE1126',   # Red
            'Spain': '#AA151B',      # Red
            'Germany': '#D3D3D3',    # Light gray 
            'Belgium': '#000000',    # Black
            'Poland': '#D3D3D3',     # Light gray 
            'Switzerland': '#FF0000', # Red
            'Uruguay': '#55AAFF',    # Light blue
            'Denmark': '#C8102E',    # Red
            'Senegal': '#00A651',    # Green
            'Australia': '#FDD835',  # Yellow 
            'Japan': '#002868',      # Navy blue
            'South Korea': '#C9151E', # Red
            'Canada': '#FF0000',     # Red
            'Mexico': '#006633',     # Green
            'Saudi Arabia': '#006C35', # Green
            'Tunisia': '#CE1126',    # Red
            'Ecuador': '#FFD100',    # Yellow
            'Iran': '#239F40',       # Green
            'Qatar': '#8A1538',      # Maroon
            'Serbia': '#C6363C',     # Red
            'Cameroon': '#007A3D',   # Green
            'Wales': '#C8102E',      # Red
            'Ghana': '#CE1126',      # Red
            'Costa Rica': '#002147', # Navy blue
            'USA': '#D3D3D3',        # Light gray 
            'United States': '#D3D3D3', # Light gray 
            'Italy': '#0066CC',      # Blue
            'Russia': '#C8102E',     # Red
            'Colombia': '#FFCD00',   # Yellow
            'Nigeria': '#008751',    # Green
            'Algeria': '#007A3D',    # Green
            'Egypt': '#CE1126',      # Red
            'Turkey': '#E30A17',     # Red
            'India': '#FF9933',      # Orange
            'Chile': '#C8102E',      # Red
            'Peru': '#D3D3D3',       # Light gray 
            'Venezuela': '#FFCE00',  # Yellow
            'Paraguay': '#CE1126',   # Red
            'Bolivia': '#007A3D',    # Green
            'China': '#DE2910',      # Red
            'Thailand': '#ED1C24',   # Red
            'Iraq': '#C8102E',       # Red
            'Vietnam': '#DA020E',    # Red
            'Indonesia': '#CE1126',  # Red
            'Norway': '#EF2B2D',     # Red
            'Sweden': '#006AA7',     # Blue
            'Finland': '#002F6C',    # Blue
            'Iceland': '#0048E0',    # Blue
            'Czech Republic': '#D7141A', # Red
            'Slovakia': '#0B4EA2',   # Blue
            'Hungary': '#436F4D',    # Green
            'Romania': '#003DA5',    # Blue
            'Bulgaria': '#00966E',   # Green
            'Ukraine': '#FFD500',    # Yellow
            'Greece': '#0D5EAF',     # Blue
            'Austria': '#C8102E',    # Red
            'Slovenia': '#007A3D',   # Green
            'North Macedonia': '#CE1126', # Red
            'Albania': '#E41E20',    # Red
            'Montenegro': '#C8102E', # Red
            'Luxembourg': '#00A1DE', # Light blue
            'Andorra': '#10069C',    # Blue
            'Malta': '#CF142B',      # Red
            'Gibraltar': '#C8102E',  # Red
            'Faroe Islands': '#ED2E38', # Red
            'San Marino': '#0099CC', # Light blue
            'Liechtenstein': '#002B7F', # Blue
        }
        
        self.fallback_colors = ['#808080', '#A0A0A0', '#909090', '#B0B0B0']
        self.fallback_index = 0
    
    def get_color_for_team(self, team_name):
        """Get the national color for a team."""
        if team_name in self.national_colors:
            return self.national_colors[team_name]
        else:
            # For unknown teams, use a fallback color and increment for next unknown team
            color = self.fallback_colors[self.fallback_index % len(self.fallback_colors)]
            self.fallback_index += 1
            print(f"Warning: No color defined for {team_name}, using fallback color {color}")
            return color
    
    def add_team_color(self, team_name, color):
        """Add or update a team's color."""
        self.national_colors[team_name] = color
    
    def get_all_defined_colors(self):
        """Get all defined team colors."""
        return self.national_colors.copy()


color_manager = TeamColorManager()