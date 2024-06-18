# import libraries
import pandas as pd
from ipywidgets import widgets
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
from mplsoccer import VerticalPitch
import io
import base64

# Load the data

# Parquet files
df_gk = pq.read_table('Parquet files/df_gk.parquet').to_pandas()
df_attack = pq.read_table('Parquet files/df_attack.parquet').to_pandas()
df_midfield = pq.read_table('Parquet files/df_midfield.parquet').to_pandas()
df_defence = pq.read_table('Parquet files/df_defence.parquet').to_pandas()

# Match data
df_match_data = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Match Data/data.csv', delimiter=',')

# Player data
df_player_defense       = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_defense.csv', delimiter=',')
df_player_gca           = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_gca.csv', delimiter=',')
df_player_keepers       = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_keepers.csv', delimiter=',')
df_player_keepersadv    = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_keepersadv.csv', delimiter=',')
df_player_misc          = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_misc.csv', delimiter=',')
df_player_passing       = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_passing.csv', delimiter=',')
df_player_passing_types = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_passing_types.csv', delimiter=',')
df_player_playingtime   = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_playingtime.csv', delimiter=',')
df_player_possession    = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_possession.csv', delimiter=',')
df_player_shooting      = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_shooting.csv', delimiter=',')
df_player_stats         = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Player Data/player_stats.csv', delimiter=',')

# Team data
df_team_data        = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Team Data/team_data.csv', delimiter=',')
df_team_group_stats = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Team Data/group_stats.csv', delimiter=',')

# Historic data
df_historic_fifa_ranking      = pd.read_csv('FIFA DataSet/Data/FIFA World Cup Historic/fifa_ranking_2022-10-06.csv', delimiter=',')
df_historic_matches_1930_2022 = pd.read_csv('FIFA DataSet/Data/FIFA World Cup Historic/matches_1930_2022.csv', delimiter=',')
df_historic_world_cup         = pd.read_csv('FIFA DataSet/Data/FIFA World Cup Historic/world_cup.csv', delimiter=',')

# Penalty shootouts
df_penalty_shootouts = pd.read_csv('FIFA DataSet/Data/FIFA World Cup Penalty Shootouts/WorldCupShootouts.csv', delimiter=',')

# Twitter data
#df_tweets_01 = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Twitter Dataset/tweets1.csv', delimiter=';')
#df_tweets_02 = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Twitter Dataset/tweets2.csv', delimiter=';')
#df_tweets = pd.concat([df_tweets_01, df_tweets_02])

# Prediction data
df_prediction_groups  = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Prediction/2022_world_cup_groups.csv', delimiter=',')
df_prediction_matches = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Prediction/2022_world_cup_matches.csv', delimiter=',')
df_prediction_international_matches = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Prediction/international_matches.csv', delimiter=',')
df_prediction_world_cup_matches = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Prediction/world_cup_matches.csv', delimiter=',')
df_prediction_world_cups = pd.read_csv('FIFA DataSet/Data/FIFA World Cup 2022 Prediction/world_cups.csv', delimiter=',')

#Player images
def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

def img_reshape(img):
    img = Image.open(img).convert('RGB')
    img = img.resize((300,300))
    img = np.asarray(img)
    return img

def showImages(group, land, player):
    images  = list_full_paths('FIFA DataSet/Data/FIFA World Cup 2022 Player Images/Images/Images/Group ' + group + '/' + land + ' Players/Images_' + player)
    img_arr = []
    
    for image in images:
        img_arr.append(img_reshape(image))
        
    rows = 5
    cols = 5
    img_count = 0
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=((5,5)))
                             
    for i in range(rows):
        for j in range(cols):
            if img_count < len(img_arr):
                axes[i,j].imshow(img_arr[img_count])
                axes[i,j].axis('off')
                img_count+=1
                
    plt.subplots_adjust(wspace=0, hspace=0)

df_player_stats_cols = ['games',
       'games_starts', 'minutes', 'minutes_90s', 'goals', 'assists',
       'goals_pens', 'pens_made', 'pens_att', 'cards_yellow', 'cards_red',
       'goals_per90', 'assists_per90', 'goals_assists_per90',
       'goals_pens_per90', 'goals_assists_pens_per90', 'xg', 'npxg',
       'xg_assist', 'npxg_xg_assist', 'xg_per90', 'xg_assist_per90',
       'xg_xg_assist_per90', 'npxg_per90', 'npxg_xg_assist_per90']

# Renaming some countries to reduce ambiguity
df_prediction_groups['Team'] = df_prediction_groups['Team'].replace('South Korea', 'Korea Republic')
df_prediction_groups['Team'] = df_prediction_groups['Team'].replace('Iran', 'IR Iran')

# make the age column an integer
df_player_stats['age'] = df_player_stats['age'].str.split('-').str[0].astype(int)

# Making the dataframe for the radar chart 
df_radar_chart = pd.merge(df_player_passing[['player', 'passes', 'passes_completed']],
                          df_player_possession[['player', 'passes_received', 'dribbles', 'touches']],
                          on='player', how='outer')

df_radar_chart = pd.merge(df_radar_chart,
                          df_player_shooting[['player', 'shots', 'average_shot_distance']],
                          on='player', how='outer')

df_radar_chart = pd.merge(df_radar_chart,
                          df_player_stats[['player', 'minutes', 'games']],
                          on='player', how='outer')

df_radar_chart.columns = [col.capitalize().replace('_', ' ') for col in df_radar_chart.columns]

# List of columns to calculate per-game metrics
per_minute_columns = ['Passes', 'Passes completed', 'Passes received', 'Dribbles', 'Touches', 'Shots', 'Minutes']

# Calculate per-game metrics
for column in per_minute_columns:
    per_minute_column_name = f'{column} per game'
    df_radar_chart[per_minute_column_name] = df_radar_chart[column] / df_radar_chart['Games']

# Drop original non-per-minute columns
df_radar_chart = df_radar_chart.drop(per_minute_columns, axis=1)

# Reorder the columns
new_column_order = ['Player', 'Games', 'Minutes per game', 'Passes per game', 'Passes completed per game', 
                    'Passes received per game', 'Dribbles per game', 'Touches per game', 
                    'Shots per game', 'Average shot distance']

df_radar_chart = df_radar_chart[new_column_order]

df_radar_chart_columns = ['Minutes per game', 'Passes per game', 'Passes completed per game', 'Passes received per game',
                           'Dribbles per game', 'Touches per game', 'Shots per game', 
                           'Average shot distance']

#preliminary code match plot function

#Get a lists of all captains in home and away matches
home_captains = df_match_data['home_captain'].tolist()
away_captains = df_match_data['away_captain'].tolist()

#concatenate these lists and remove duplicates
list_all_captains = list(set(home_captains + away_captains))

# Function to generate pitch visualization
from PIL import ImageDraw, ImageFont

def add_player_names_to_image(image_path, player_name, text_position):
    # Open the image
    img = Image.open(image_path).convert('RGB')

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Load a bold font with a larger size for better readability
    font_size = 16
    bold_font = ImageFont.load_default().font_variant(size=font_size)

    # Draw the player name on the image using a bold font and bright color
    draw.text(text_position, player_name, fill=(255, 255, 0), font=bold_font)

    # Save the modified image
    img.save(image_path)



def generate_pitch_visualization(home_team, away_team):
    home_formation = df_match_data[(df_match_data['home_team'] == home_team) & (df_match_data['away_team'] == away_team)]['home_formation'].iloc[0]
    away_formation = df_match_data[(df_match_data['home_team'] == home_team) & (df_match_data['away_team'] == away_team)]['away_formation'].iloc[0]

    # Create a single VerticalPitch object for both teams
    pitch = VerticalPitch('uefa', line_alpha=0.5, pitch_color='#53ac5c', line_color='white')

    # Draw the pitch
    fig, ax = plt.subplots(figsize=(18, 6))
    pitch.draw(ax=ax)

    # Plot formations for both teams on the same pitch
    pitch.formation(home_formation, ax=ax, linewidth=2, color='blue',flip = True, half = True, label=home_team)
    pitch.formation(away_formation, ax=ax, linewidth=2, color='red', half = True, label=away_team)

    # Add legend for clarity
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Convert the plot to a base64 encoded image
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # Close the plot
    plt.close()

    return f'data:image/png;base64,{img_base64}'

#list of all player dataframes
player_dataframes = [df_player_defense, df_player_gca, df_player_keepers, df_player_keepersadv, df_player_misc, df_player_passing, df_player_passing_types, df_player_playingtime, df_player_possession, df_player_shooting, df_player_stats]

# Define the colors
background_color = '#272b30'

# Global list to keep track of selected player indices
selected_indices = []

# Global dataframe to keep track of the selected position
df_scatter_stats = df_attack

# Initialize the app
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])

# Define the layout
app.layout = html.Div(children=[
    html.H1('World Cup 2022 Visualization', style={'textAlign': 'center', 'color': 'white'}), # Title
    dbc.Row([
        dbc.Col([
            dcc.Graph(   # Scatter plot
            id='scatter-plot'
            ),
            dcc.Dropdown(   # Dropdown for player search (scatter plot)
            id='player-dropdown',
            options=[{'label': i, 'value': i} for i in df_player_stats['player'].unique()],
            #value='Lionel Messi',   # set default player
            searchable=True,
            style={'width': '60%', 'marginLeft': '19px', 'color':'black'},
            multi=True   # Multiple search options
            ),
            dcc.Dropdown(  # Dropdown for xaxis
                id='xaxis-column',
                value=sorted(df_scatter_stats)[0],
                style={'width': '60%', 'marginLeft': '19px', 'color':'black'}
            ),
            dcc.Dropdown(   # Dropdown for yaxis
                id='yaxis-column',
                value=sorted(df_scatter_stats)[1],
                style={'width': '60%', 'marginLeft': '19px', 'color':'black'}
                #style={'textAlign': 'left', 'color': '#7FDBFF'}
            ),
            dcc.Dropdown(   # Dropdown for player positions
                id='player-positions',
                options=['goalkeepers', 'defenders', 'midfielders', 'attackers'],
                value='attackers',
                style={'width': '60%', 'marginLeft': '19px', 'color':'black'}
                #style={'textAlign': 'left', 'color': '#7FDBFF'}
            ),
        ], width=6),
        dbc.Col([
            html.Div([   # Div for click data
                    dcc.Markdown("""
                        **Scatter plot**

                        Click on points in the graph. Select multiple points by holding down the shift button. 
                                Deselect points by clicking on them again.
                    """,  style={'color': 'white', 'font-family': 'Helvetica', 'font-size': '20px'}),
                    html.Pre(id='click-data'),
                ], className='three columns'
                ),
            dcc.Graph(   # radar chart
                id='radar-chart',
                style={'marginLeft': '19px', 'marginBottom': '10px'}
            ),
        ], width=6),
    ]),
    html.Div(style={'margin-bottom': 20}), # add a small space between the two rows
    dbc.Row([
        dbc.Col([
            dcc.Graph(   # Bar plot
                id='bar-plot',
                style={'marginLeft': '19px', 'marginBottom': '10px'}
                ),
                dcc.Dropdown(  # Dropdown for bar chart
                id='player-stats-bar',
                options=[{'label': i, 'value': i} for i in df_player_stats_cols],
                style={'width': '60%', 'marginLeft': '9px', 'color':'black'},
                multi=True
                ),
        ], width=6),
        dbc.Col([
            dcc.Graph(   # Histogram
            id='histogram',
            style={'marginBottom': '10px'}
            ),
            dcc.Dropdown(   # Dropdown for player search (histogram)
                id='player-dropdown2',
                options=[{'label': i, 'value': i} for i in df_player_stats['player'].unique()],
                value='Lionel Messi',   # set default player
                searchable=True,
                style={'width': '60%', 'color':'black'}
            ),
            dcc.Dropdown(   # Dropdown for histogram
                id='dropdown',
                options=[{'label': i, 'value': i} for i in df_player_stats.columns[[3]].append(df_player_stats.columns[6:])],   # drop categorical attributes
                value='xg',   # set default value
                style={'width': '60%', 'color':'black', 'marginBottom': '20px'}
            ),
            html.Div(   # Text for the histogram
            id='text',
            style = {'marginRight':'19px', 'color':'white'})
        ], width=6),
    ]),
    html.Div(style={'margin-bottom': 80}), # add a small space between the two rows
    dbc.Row([   # Player pitch row
        html.H1("Match data", style={'textAlign': 'center', 'color': 'white'}   # Row title
                ),
        html.H2("Select home and away players and then press 'Generate Pitch'", style={'textAlign': 'center', 'color': 'white'}   # Row substitle
                ),
        dcc.Dropdown(   # Home team dropdown
            id='home-player-dropdown',
            options=[{'label': player, 'value': player} for df in player_dataframes for player in df['player'].unique()],
            value=player_dataframes[0]['player'].iloc[0],
            style={'width': '50%', 'margin-top': '10px', 'margin-bottom': '10px', 'color':'black'}
        ),
        dcc.Dropdown(   # Away team dropdown
            id='away-player-dropdown',
            style={'width': '50%', 'margin-top': '10px', 'margin-bottom': '10px', 'color': 'black'}
        ),
        html.Button('Generate Pitch', id='generate-pitch-button', n_clicks=0   # Generarte button
                    ), 
        html.Div([   # Football field
            html.Img(id='pitch-image', style={'width': '60%', 'display': 'inline-block', 'margin-bottom': '10px'}
                     ),
            html.Div(id='match-info-table')   # Match data table
        ], style={'text-align': 'center', 'margin': 'auto', 'margin-top': '10px'}),
        html.Div(id='selected-players-output', style={'textAlign': 'center', 'color': 'white'})
    ]),
    html.Div(style={'margin-bottom': 50}), # add a small space between the two rows
    html.Footer([
        html.P("© 2023 Visualization Group 25 Technische Universiteit Eindhoven",
               style = {'marginLeft':'19px'}),
        html.P("The data sets can be downloaded at: https://www.kaggle.com/datasets",
               style = {'marginLeft':'19px'}),
    ])  
])

# Callback to update x-axis and y-axis options based on player-positions dropdown
@app.callback(
    [Output('xaxis-column', 'options'),
     Output('yaxis-column', 'options'),
     Output('xaxis-column', 'value'),  # Reset the x-axis dropdown value
     Output('yaxis-column', 'value')   # Reset the y-axis dropdown value
    ],
    [Input('player-positions', 'value')]
)
def update_dropdown_options(player_position):
    # Assuming df_scatter_stats is a global variable
    global df_scatter_stats
    
    # Update df_scatter_stats based on player-positions
    if player_position == 'goalkeepers':
        df_scatter_stats = df_gk
    elif player_position == 'defenders':
        df_scatter_stats = df_defence
    elif player_position == 'midfielders':
        df_scatter_stats = df_midfield
    elif player_position == 'attackers':
        df_scatter_stats = df_attack
    
    # Update x-axis and y-axis options
    options = [{'label': i, 'value': i} for i in sorted(df_scatter_stats)]
    
    # Set the initial values for x-axis and y-axis dropdowns
    initial_x_value = sorted(df_scatter_stats)[0]
    initial_y_value = sorted(df_scatter_stats)[1]
    
    return options, options, initial_x_value, initial_y_value

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('player-dropdown', 'value'),
     Input('scatter-plot', 'selectedData'),
     Input('player-positions', 'value')]
)
def update_graph(xaxis_column_name, yaxis_column_name, selected_players, selectedData, player_position):
    global selected_indices

    # Update selected_indices based on selectedData from scatter plot
    if selectedData:
        selected_indices = [point['pointIndex'] for point in selectedData['points']]

    global df_scatter_stats
    # Update the scatter plot on position
    if player_position == 'goalkeepers':
        df_scatter_stats = df_gk
    elif player_position == 'defenders':
        df_scatter_stats = df_defence
    elif player_position == 'midfielders':
        df_scatter_stats = df_midfield
    elif player_position == 'attackers':
        df_scatter_stats = df_attack

    # Update selected_indices based on selected_players from dropdown
    if selected_players:
        for player in selected_players:
            player_indices =  df_scatter_stats[df_scatter_stats['Player'] == player].index.tolist()
            for index in player_indices:
                if index not in selected_indices:
                    selected_indices.append(index)


    return {
        'data': [dict(   # create scatter plot
            x= df_scatter_stats[xaxis_column_name],
            y= df_scatter_stats[yaxis_column_name],
            text= df_scatter_stats['Player'],
            mode='markers',   
            marker={
                'size': 10,
                'opacity': 0.4,
                'line': {'width': 0.5, 'color': 'white'}
            },
            selected={   # make selected points red
                'marker': {
                    'color': 'red'
                }
            },
            unselected={   # make unselected points lighter
                'marker': {
                    'opacity': 0.6
                }
            },
            selectedpoints=selected_indices   # set selected points
        )],
        'layout': dict(   # define layout of the scatter plot
            xaxis={'title': xaxis_column_name},
            yaxis={'title': yaxis_column_name},
            #title = 'Scatter plot',
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},   
            hovermode='closest',
            clickmode='event+select',
            #plot_bgcolor='white',
            paper_bgcolor=background_color,
            font={'color': 'white'}
        )
    }

# callback for table
@app.callback(
    Output('click-data', 'children'),
    [Input('scatter-plot', 'selectedData'),
    Input('player-dropdown', 'value')]
    )
def display_click_data(selectedData, selected_players):
    # Combine selected players from graph and dropdown
    selected_names = set()

    if selectedData:
        for point in selectedData['points']:
            selected_names.add(point['text'])

    if selected_players:
        selected_names.update(selected_players)

    # If no players are selected
    if not selected_names:
        return 'Select a player by clicking on a point on the graph or use the search bar'

    # Initialize an empty list to store the table rows
    table_rows = []

    # Loop over all names
    for name in selected_names:
        # Look up age, team, and group in df_player_stats and df_prediction_groups
        age = df_player_stats.loc[df_player_stats['player'] == name, 'age'].iloc[0]
        #age = age.split('-')[0]
        team = df_player_stats.loc[df_player_stats['player'] == name, 'team'].iloc[0]
        group = df_prediction_groups.loc[df_prediction_groups['Team'] == team, 'Group'].iloc[0]
        club = df_player_stats.loc[df_player_stats['player'] == name, 'club'].iloc[0]


        # Create a table row for the player
        table_row = html.Tr([
            html.Td(name),
            html.Td(age),
            html.Td(team),
            html.Td(group),
            html.Td(club)
        ])

        # Add the table row to the list
        table_rows.append(table_row)

    # Create a table with the table rows
    table_header =[html.Thead([
            html.Tr([
            html.Th('Name'),
            html.Th('Age'),
            html.Th('Country'),
            html.Th('Group'),
            html.Th('Club')
        ])
    ])]
    table_body = [html.Tbody(table_rows)]

    table = dbc.Table(table_header + table_body, bordered=True, striped=True)

    return table

# Callback to update bar plot
@app.callback(   
    Output('bar-plot', 'figure'),
    [Input('scatter-plot', 'selectedData'),
     Input('player-dropdown', 'value'),
     Input('player-stats-bar', 'value')]
     )
def update_bar_plot(selectedData, selected_players, selectedStats):
    selected_names = set()

    if selectedData:
        for point in selectedData['points']:
            selected_names.add(point['text'])

    if selected_players:
        selected_names.update(selected_players)

    if selected_names is None:   # if no points are selected
        return dash.no_update
    
    if selectedStats is None:   # if no stats are selected
        return dash.no_update

    # Initialize an empty list to store the data for the plot
    data = []

    # Get player stats and add to the data list
    for name in selected_names:
        player_stats = (
            df_player_stats.loc[df_player_stats['player'] == name, selectedStats] - df_player_stats[selectedStats].min()
            ) / (df_player_stats[selectedStats].max() - df_player_stats[selectedStats].min())

        hover_text = [f"{stat}: {real_value:.2f}" for stat, real_value in zip(selectedStats, df_player_stats.loc[df_player_stats['player'] == name, selectedStats].values[0])]

        data.append({
            'x': player_stats.columns,
            'y': player_stats.iloc[0],
            'type': 'bar',
            'name': name,
            'hovertemplate': f"{name}<br>" + "<br>".join(hover_text)
        })
    # Create a bar plot of player stats
    figure = {
        'data': data,
        'layout': dict(
            title ='Normalized Player Comparison',
            annotations=[{
                'text': 'Hover over the bars to see the real values of the chosen statistic(s)',
                'showarrow': False,
                'x': 0.5,
                'y': 1.05,
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'center',
                'yanchor': 'bottom',
                'font': {'size': 12},
            }],
            #plot_bgcolor='#E0E0E0',
            #paper_bgcolor='#E0E0E0'
        )
    }

    return figure

# Define callback to update radar chart
@app.callback(
    Output('radar-chart', 'figure'),   
    [Input('scatter-plot', 'selectedData'),
     Input('player-dropdown', 'value')]
     )
def update_radar_chart(selectedData, selected_players):
    selected_names = set()
    if selectedData:
        for point in selectedData['points']:
            selected_names.add(point['text'])

    if selected_players:
        selected_names.update(selected_players)

    if not selected_names:   # if no points are selected
        # Return default plot where all attributes are set to 1
        default_fig = go.Figure(data=go.Scatterpolar(
        r=[1] * len(df_radar_chart_columns),
        theta=df_radar_chart_columns,
        fill='toself'
        ))

        default_fig.update_layout(
            title='Select a player to show radar chart',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    color = 'black'
                ),
            ),
            showlegend=False,
            font=dict(
                color='white'
            ),
            paper_bgcolor=background_color
        )

        return default_fig
        
    # Initialize the list for 'r' values with default values (1)
    r_values = [1] * len(df_radar_chart_columns)

    fig = go.Figure()

    for name in selected_names:
        # Initialize the list for 'r' values with default values (1)
        r_values = [1] * len(df_radar_chart_columns)

        player_stats = df_radar_chart.loc[df_radar_chart['Player'] == name, df_radar_chart_columns].values.tolist()
        if player_stats:
            # Normalize the player stats using min-max scaling
            min_values = df_radar_chart[df_radar_chart_columns].min().values
            max_values = df_radar_chart[df_radar_chart_columns].max().values

            normalized_stats = [
                (x - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
                for x, min_val, max_val in zip(player_stats[0], min_values, max_values)
            ]

            r_values = normalized_stats

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=df_radar_chart_columns,
            fill='toself',
            name=name  # Assign a name to each trace (player)
        ))

    fig.update_layout(
        title='Normalized Radar Chart',
        polar=dict(
            radialaxis=dict(
                visible=True,
                color = 'black'
            ),
        ),
        showlegend=True,  # Show legend with player names
        font=dict(
            color='white'  # Set font color to white
        ),
        paper_bgcolor=background_color  # Set paper background color
    )

    return fig

# Define callback to update histogram and text
@app.callback(
    Output('histogram', 'figure'),   
    Output('text', 'children'),
    [Input('dropdown', 'value'), Input('player-dropdown2', 'value')]
)
def update_graph_and_text(selected_value, selected_player):   # selected_value is the value of the dropdown, selected_player is the value of the player dropdown
    fig = px.histogram(df_player_stats, x=selected_value, template='none')   # create histogram
    player_value = df_player_stats.loc[df_player_stats['player'] == selected_player, selected_value].values[0]   # get value of selected player
    fig.add_shape(   # add line to show where the player is on the histogram
        type='line',
        x0=player_value, x1=player_value,   # x0 and x1 are the same to create a vertical line
        y0=0, y1=1, yref='paper',   # yref='paper' ensures the y-coordinate is in the range of 0 to 1
        line=dict(color='Red', dash='dot')
    )
    fig.update_layout(title_text='Histogram of ' + selected_value)   # set title of histogram

    percentile = np.sum(df_player_stats[selected_value] < player_value) / len(df_player_stats)* 100   # calculate percentile of player
    text = f"{selected_player} is outperforming {percentile:.1f}%  of the players on {selected_value}"   # create text, rounded to 1 decimal

    return fig, text

# Callback to change away player options
@app.callback(
        Output('away-player-dropdown', 'options'),
        [Input('home-player-dropdown', 'value')]
    )
def update_away_player_options(selected_home_player):
    selected_team = player_dataframes[0].loc[player_dataframes[0]['player'] == selected_home_player, 'team'].iloc[0]
    teams = df_match_data.loc[df_match_data['home_team'] == selected_team, 'away_team'].tolist()

    player_options = [{'label': player, 'value': player} for df in player_dataframes for player in df[df['team'].isin(teams)]['player'].unique()]

    return player_options

# Callback to set initial away player
@app.callback(
    Output('away-player-dropdown', 'value'),
    [Input('away-player-dropdown', 'options')]
)
def set_initial_value(available_options):
    if available_options:
        return available_options[0]['value']
    return None

# Callback to make selected players text
@app.callback(
    Output('selected-players-output', 'children'),
    [Input('home-player-dropdown', 'value'),
    Input('away-player-dropdown', 'value')]
)  
def display_selected_players(selected_home_player, selected_away_player):
    return f"Selected players: {selected_home_player} (Home) and {selected_away_player} (Away)"

initial_home_player = player_dataframes[0]['player'].iloc[0]

# Calback to greate the match pitch and table
@app.callback(
    [Output('pitch-image', 'src'), Output('match-info-table', 'children')],
    [Input('generate-pitch-button', 'n_clicks')],
    [State('home-player-dropdown', 'value'),
     State('away-player-dropdown', 'value')]
)
def generate_pitch(n_clicks, selected_home_player, selected_away_player):
    if n_clicks > 0:
        home_team = player_dataframes[0].loc[player_dataframes[0]['player'] == selected_home_player, 'team'].iloc[0]
        away_team = player_dataframes[1].loc[player_dataframes[1]['player'] == selected_away_player, 'team'].iloc[0]

        # Fetch positions of selected players
        home_position = player_dataframes[0].loc[player_dataframes[0]['player'] == selected_home_player, 'position'].iloc[0]
        away_position = player_dataframes[1].loc[player_dataframes[1]['player'] == selected_away_player, 'position'].iloc[0]

        image_source = generate_pitch_visualization(home_team, away_team)

        match_info = df_match_data[
            (df_match_data['home_team'] == home_team) & (df_match_data['away_team'] == away_team)
        ][['score', 'attendance', 'venue', 'home_possession', 'away_possession', 'home_total_shots', 'away_total_shots']]
        
        # Add positions to the match_info DataFrame
        match_info['home_position'] = home_position
        match_info['away_position'] = away_position

        #match_info_table = match_info.to_dict('records')


        score = match_info['score']
        attendance = match_info['attendance']
        venue = match_info['venue']
        home_possession = match_info['home_possession']
        away_possession = match_info['away_possession']
        home_total_shots = match_info['home_total_shots']
        away_total_shots = match_info['away_total_shots']
        home_player_position = match_info['home_position']
        away_player_position = match_info['away_position']


        # Create a table row for the player
        table_rows = html.Tr([
            html.Td(score),
            html.Td(attendance),
            html.Td(venue),
            html.Td(home_possession),
            html.Td(away_possession),
            html.Td(home_total_shots),
            html.Td(away_total_shots),
            html.Td(home_player_position),
            html.Td(away_player_position),
        ])

        # Create a table with the table rows
        table_header =[html.Thead([
                html.Tr([
                html.Th('Score'),
                html.Th('Attendance'),
                html.Th('Venue'),
                html.Th('Home Possesion'),
                html.Th('Away Possesion'),
                html.Th('Home total shots'),
                html.Th('Away total shots'),
                html.Th('Home player position'),
                html.Th('Away player position'),
                ])
            ])]
        table_body = [html.Tbody(table_rows)]

        table = dbc.Table(table_header + table_body, bordered=True, striped=True)

        return image_source, table
    return '', []


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
