import streamlit as st
import pandas as pd


def read_csv(filename):
    df = pd.read_csv(filename)
    if 'total_points' in df.columns:
        df = df.sort_values(by='total_points', ascending=False)
    return df


# Define the file names
file_names = {
    'Teams': 'teams.csv',
    'Goalkeepers': 'goalkeepers.csv',
    'Defenders': 'defenders.csv',
    'Midfielders': 'midfielders.csv',
    'Forwards': 'forwards.csv',
    'Injuries': 'injuries.csv',
    'Penalty Takers': 'penalty_taker.csv',
    'Set Piece Takers': 'setpiece.csv'
}


def app():
    st.set_page_config(page_title='FPL app', layout='wide')
    st.title('Fantasy Premier League App')

    # Add image to the top right of the app
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(" ")  # Add empty space to align the image
    with col2:
        st.image('football.jpg', use_column_width=False)

    # Create tabs for each file
    tabs = st.tabs(list(file_names.keys()))
    for i, tab_name in enumerate(file_names.keys()):
        with tabs[i]:
            st.title(tab_name)
            df = read_csv(file_names[tab_name])

            # Get the list of columns to display
            columns = st.multiselect('Select columns to display:', list(df.columns), default=list(df.columns))

            # Filter the dataframe to display only the selected columns
            filtered_df = df[columns]

            if tab_name != 'Teams':
                # Add a text input widget to filter by player name
                player_name = st.text_input('Enter player name:', key=f"{tab_name}_player_name")
                filtered_df1 = filtered_df[filtered_df['Player'].str.contains(player_name, case=False)]
            else:
                filtered_df1 = filtered_df

            if tab_name != 'Teams':
                # Add a text input widget to filter by team name
                team_name = st.text_input('Enter team name:', key=f"{tab_name}_team_name")
                filtered_df2 = filtered_df1[filtered_df1['team'].str.contains(team_name, case=False)]
            else:
                filtered_df2 = filtered_df1

            # Display the filtered dataframe
            st.dataframe(filtered_df2)


if __name__ == '__main__':
    app()
