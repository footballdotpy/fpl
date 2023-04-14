import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# the fpl api url.
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

# make the request
res = requests.get(url)

# check the status code and print a message
if res.status_code == 200:
    print('The API request was successful.')
else:
    print(f'Error: The API request failed with status code {res.status_code}.')

# store the output.
json = res.json()

# the keys reveal what information is stored in the api.
json.keys()

# dict_keys(['events', 'game_settings', 'phases', 'teams', 'total_players', 'elements', 'element_stats', 'element_types'])

elements_df = pd.DataFrame(json['elements'])
elements_types_df = pd.DataFrame(json['element_types'])
teams_df = pd.DataFrame(json['teams'])

# subset the information.

players_df = elements_df[
    ['first_name', 'second_name', 'photo', 'team', 'element_type', 'selected_by_percent', 'now_cost', 'minutes',
     'transfers_in', 'value_season', 'points_per_game', 'total_points', 'form', 'value_form', 'value_season',
     'goals_scored', 'assists', 'clean_sheets', 'yellow_cards', 'bonus', 'bps', 'influence', 'creativity', 'threat',
     'starts', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_per_90',
     'expected_assists_per_90', 'saves', 'penalties_saved', 'goals_conceded', 'expected_goals_conceded', 'saves_per_90',
     'expected_goals_conceded_per_90', 'expected_goal_involvements_per_90', 'goals_conceded_per_90',
     'clean_sheets_per_90', 'starts_per_90',
     'chance_of_playing_next_round', 'chance_of_playing_this_round', 'news', 'penalties_order',
     'corners_and_indirect_freekicks_order']]

# convert string columns to float.

cols_to_convert = ['form', 'value_form', 'points_per_game', 'influence', 'creativity', 'threat', 'starts',
                   'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_per_90',
                   'expected_assists_per_90', 'expected_goals_conceded']

players_df[cols_to_convert] = players_df[cols_to_convert].astype(float)

# concat the first and second name variables.

players_df['Player'] = players_df['first_name'] + ' ' + players_df['second_name']

# map the team names from the teams_df to the players_df, same for position.

players_df['team'] = players_df.team.map(teams_df.set_index('id').name)
players_df['position'] = players_df.element_type.map(elements_types_df.set_index('id').singular_name)

# filter out players who have no minutes.

players_df = players_df[players_df['minutes'] != 0]

# reformat the now_cost variable to sync in with the site values.

players_df['now_cost'] = (players_df['now_cost'] / 10).apply(lambda x: '{:.1f}'.format(x))


# aggregate the team totals to get %of team total player variable later

team_stats = players_df.groupby('team').agg({'total_points': 'sum', 'bonus': 'sum', 'goals_scored': 'sum','assists':'sum','expected_goals': 'sum','expected_assists':'sum'}).sort_values('total_points',ascending=False).reset_index()

team_stats['performance_xG'] = team_stats['goals_scored'] - team_stats['expected_goals']
team_stats['performance_xGa'] = team_stats['assists'] - team_stats['expected_assists']

columns = ['team_name','team_total_points','team_bonus','team_goals_scored','team_assists','team_expected_goals','team_expected_assists','team_performance_xG','team_performance_xGa']

# rename these columns to avoid clashing with merge.
team_stats.columns = columns

# apply minmaxscaler to scale the columns.

scaler = MinMaxScaler()

players_df[['influence', 'creativity', 'threat']] = scaler.fit_transform(players_df[['influence', 'creativity', 'threat']])

# merge player and team dfs to get the %of team total points.
players_df = pd.merge(players_df, team_stats, left_on='team',right_on='team_name')

players_df['%of_team_points'] = (players_df['total_points'] / players_df['team_total_points']) * 100

# drop these team df columns.
players_df = players_df.drop(['team_name','team_total_points','team_bonus','team_goals_scored','team_assists','team_expected_goals','team_expected_assists','team_performance_xG','team_performance_xGa'],axis=1)

# subset based on position.

defenders_df = players_df[players_df['position'] == 'Defender']
midfielders_df = players_df[players_df['position'] == 'Midfielder']
forwards_df = players_df[players_df['position'] == 'Forward']
goalkeepers_df = players_df[players_df['position'] == 'Goalkeeper']

# subset individual dfs with metrics relating to their position.

goalkeepers_df = goalkeepers_df[['Player','team','position','selected_by_percent','starts_per_90','total_points','bonus','points_per_game','form','saves','saves_per_90','expected_goals_conceded','goals_conceded','clean_sheets_per_90','value_season','value_form','now_cost','clean_sheets','penalties_saved','yellow_cards','%of_team_points']]
goalkeepers_df['performance_xG_def'] = goalkeepers_df['goals_conceded'] - goalkeepers_df['expected_goals_conceded']

defenders_df = defenders_df[['Player','team','position','selected_by_percent','starts_per_90','total_points','bonus','points_per_game','expected_goals_conceded','goals_conceded','clean_sheets_per_90','expected_goals','goals_scored','expected_goal_involvements_per_90','value_season','value_form','now_cost','clean_sheets','yellow_cards','%of_team_points','influence', 'creativity', 'threat','penalties_order','corners_and_indirect_freekicks_order']]
defenders_df['performance_xG_def'] = defenders_df['goals_conceded'] - defenders_df['expected_goals_conceded']

midfielders_df = midfielders_df[['Player','team','position','selected_by_percent','starts_per_90','total_points','bonus','points_per_game','expected_goals_conceded','goals_conceded','clean_sheets_per_90','expected_goals','goals_scored','expected_goal_involvements_per_90','value_season','value_form','now_cost','clean_sheets','yellow_cards','%of_team_points','influence', 'creativity', 'threat','penalties_order','corners_and_indirect_freekicks_order']]
midfielders_df['performance_xG_off'] = midfielders_df['goals_scored'] - midfielders_df['expected_goals']

forwards_df = forwards_df[['Player','team','position','selected_by_percent','starts_per_90','total_points','bonus','points_per_game','expected_goals','goals_scored','expected_goal_involvements_per_90','value_season','value_form','now_cost','clean_sheets','yellow_cards','%of_team_points','influence', 'creativity', 'threat','penalties_order','corners_and_indirect_freekicks_order']]
forwards_df['performance_xG_off'] = forwards_df['goals_scored'] - forwards_df['expected_goals']
# create a df based on injury news.

injuries_df = players_df[players_df['chance_of_playing_this_round'] < 0.66]
injuries_df = injuries_df[~injuries_df['news'].str.contains('Season-long loan|Recalled')]

injuries_df = injuries_df[['Player','team','chance_of_playing_this_round','news']].reset_index(drop=True)

# create a df based on penalty taker

penalty_df = players_df[players_df['penalties_order'] < 3]
penalty_df = penalty_df[['Player','team','penalties_order']].reset_index(drop=True)

# create a df based on set piece duties

setpiece_df = players_df[players_df['corners_and_indirect_freekicks_order'] < 3]
setpiece_df = setpiece_df[['Player','team','corners_and_indirect_freekicks_order']].reset_index(drop=True)


# save the csvs.

team_stats.to_csv('teams.csv',index=False)
goalkeepers_df.to_csv('goalkeepers.csv',index=False)
defenders_df.to_csv('defenders.csv',index=False)
midfielders_df.to_csv('midfielders.csv',index=False)
forwards_df.to_csv('forwards.csv',index=False)
injuries_df.to_csv('injuries.csv',index=False)
penalty_df.to_csv('penalty_taker.csv', index=False)
setpiece_df.to_csv('setpiece.csv',index=False)