import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import warnings
import streamlit as st

warnings.filterwarnings('ignore')


# Function to fetch data from the FPL API
def fetch_fpl_data():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    res = requests.get(url)
    if res.status_code == 200:
        print('The API request was successful.')
        return res.json()
    else:
        print(f'Error: The API request failed with status code {res.status_code}.')
        return None


# Function to preprocess and filter the FPL data
def preprocess_fpl_data(json_data):

    elements_df = pd.DataFrame(json_data['elements'])
    elements_types_df = pd.DataFrame(json_data['element_types'])
    teams_df = pd.DataFrame(json_data['teams'])

    # subset the information.

    players_df = elements_df[
        ['first_name', 'second_name', 'photo', 'team', 'element_type', 'selected_by_percent', 'now_cost', 'minutes',
         'transfers_in', 'value_season', 'points_per_game', 'total_points', 'form', 'value_form', 'value_season',
         'goals_scored', 'assists', 'clean_sheets', 'yellow_cards', 'bonus', 'bps', 'influence', 'creativity', 'threat',
         'starts', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_per_90',
         'expected_assists_per_90', 'saves', 'penalties_saved', 'goals_conceded', 'expected_goals_conceded',
         'saves_per_90',
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

    team_stats = players_df.groupby('team').agg(
        {'total_points': 'sum', 'bonus': 'sum', 'goals_scored': 'sum', 'assists': 'sum', 'expected_goals': 'sum',
         'expected_assists': 'sum'}).sort_values('total_points', ascending=False).reset_index()

    team_stats['performance_xG'] = team_stats['goals_scored'] - team_stats['expected_goals']
    team_stats['performance_xGa'] = team_stats['assists'] - team_stats['expected_assists']

    columns = ['team_name', 'team_total_points', 'team_bonus', 'team_goals_scored', 'team_assists',
               'team_expected_goals', 'team_expected_assists', 'team_performance_xG', 'team_performance_xGa']

    # rename these columns to avoid clashing with merge.
    team_stats.columns = columns

    # apply minmaxscaler to scale the columns.

    scaler = MinMaxScaler()

    players_df[['influence', 'creativity', 'threat']] = scaler.fit_transform(
        players_df[['influence', 'creativity', 'threat']])

    # merge player and team dfs to get the %of team total points.
    players_df = pd.merge(players_df, team_stats, left_on='team', right_on='team_name', suffixes=('_player', '_team'))

    players_df['%of_team_points'] = (players_df['total_points'] / players_df['team_total_points']) * 100

    # drop these team df columns.
    players_df = players_df.drop(
        ['team_name', 'team_total_points', 'team_bonus', 'team_goals_scored', 'team_assists', 'team_expected_goals',
         'team_expected_assists', 'team_performance_xG', 'team_performance_xGa'], axis=1)

    # subset based on position.

    defenders_df = players_df[players_df['position'] == 'Defender']
    midfielders_df = players_df[players_df['position'] == 'Midfielder']
    forwards_df = players_df[players_df['position'] == 'Forward']
    goalkeepers_df = players_df[players_df['position'] == 'Goalkeeper']

    # subset individual dfs with metrics relating to their position.

    goalkeepers_df_sorted = goalkeepers_df[
        ['Player', 'team', 'position', 'selected_by_percent', 'starts_per_90', 'total_points', 'bonus',
         'points_per_game', 'form', 'saves', 'saves_per_90', 'expected_goals_conceded', 'goals_conceded',
         'clean_sheets_per_90', 'value_season', 'value_form', 'now_cost', 'clean_sheets', 'penalties_saved',
         'yellow_cards', '%of_team_points']]
    goalkeepers_df_sorted['performance_xG_def'] = goalkeepers_df['goals_conceded'] - goalkeepers_df['expected_goals_conceded']

    defenders_df_sorted  = defenders_df[
        ['Player', 'team', 'position', 'selected_by_percent', 'starts_per_90', 'total_points', 'bonus',
         'points_per_game', 'expected_goals_conceded', 'goals_conceded', 'clean_sheets_per_90', 'expected_goals',
         'goals_scored', 'expected_goal_involvements_per_90', 'value_season', 'value_form', 'now_cost', 'clean_sheets',
         'yellow_cards', '%of_team_points', 'influence', 'creativity', 'threat', 'penalties_order',
         'corners_and_indirect_freekicks_order']]
    defenders_df_sorted['performance_xG_def'] = defenders_df['goals_conceded'] - defenders_df['expected_goals_conceded']

    midfielders_df_sorted = midfielders_df[
        ['Player', 'team', 'position', 'selected_by_percent', 'starts_per_90', 'total_points', 'bonus',
         'points_per_game', 'expected_goals_conceded', 'goals_conceded', 'clean_sheets_per_90', 'expected_goals',
         'goals_scored', 'expected_goal_involvements_per_90', 'value_season', 'value_form', 'now_cost', 'clean_sheets',
         'yellow_cards', '%of_team_points', 'influence', 'creativity', 'threat', 'penalties_order',
         'corners_and_indirect_freekicks_order']]
    midfielders_df_sorted['performance_xG_off'] = midfielders_df['goals_scored'] - midfielders_df['expected_goals']

    forwards_df_sorted = forwards_df[
        ['Player', 'team', 'position', 'selected_by_percent', 'starts_per_90', 'total_points', 'bonus',
         'points_per_game', 'expected_goals', 'goals_scored', 'expected_goal_involvements_per_90', 'value_season',
         'value_form', 'now_cost', 'clean_sheets', 'yellow_cards', '%of_team_points', 'influence', 'creativity',
         'threat', 'penalties_order', 'corners_and_indirect_freekicks_order']]
    forwards_df_sorted['performance_xG_off'] = forwards_df['goals_scored'] - forwards_df['expected_goals']
    # create a df based on injury news.

    injuries_df = players_df[players_df['chance_of_playing_this_round'] < 0.66]
    injuries_df = injuries_df[~injuries_df['news'].str.contains('Season-long loan|Recalled')]

    injuries_df = injuries_df[['Player', 'team', 'chance_of_playing_this_round', 'news']].reset_index(drop=True)

    # create a df based on penalty taker

    penalty_df = players_df[players_df['penalties_order'] < 3]
    penalty_df = penalty_df[['Player', 'team', 'penalties_order']].reset_index(drop=True)

    # create a df based on set piece duties

    setpiece_df = players_df[players_df['corners_and_indirect_freekicks_order'] < 3]
    setpiece_df = setpiece_df[['Player', 'team', 'corners_and_indirect_freekicks_order']].reset_index(drop=True)

    return goalkeepers_df_sorted, defenders_df_sorted, midfielders_df_sorted, forwards_df_sorted

def app():
    st.set_page_config(page_title='FPL app', layout='wide')
    st.title('Fantasy Premier League App')

    # Assuming you have the dataframes goalkeepers_df_sorted, defenders_df_sorted, midfielders_df_sorted, and forwards_df_sorted
    goalkeepers_df_sorted, defenders_df_sorted, midfielders_df_sorted, forwards_df_sorted = \
        preprocess_fpl_data(json_data=fetch_fpl_data())

    # Define tab names
    tab_names = ['Goalkeepers', 'Defenders', 'Midfielders', 'Forwards']

    # Select the active tab using a selectbox
    selected_tab = st.selectbox('Select a tab', tab_names)

    # Map the selected tab to the corresponding dataframe
    tab_data = {
        'Goalkeepers': goalkeepers_df_sorted,
        'Defenders': defenders_df_sorted,
        'Midfielders': midfielders_df_sorted,
        'Forwards': forwards_df_sorted
    }

    # Display the selected dataframe
    st.title(selected_tab)
    st.write(tab_data[selected_tab].to_html(), unsafe_allow_html=True)

if __name__ == '__main__':
    app()