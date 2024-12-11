# import required modules
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud # for unstructured text visualization
import plotly.express as px
import streamlit as st # for streamlit commands
import pydeck as pdk # interactive map
import pandas as pd
import calendar # for date analysis
import ast # for type conversions

st.set_page_config(layout='wide', page_title='Toronto Airbnb Choice Analysis', page_icon='🏡')

st.title('Toronto Airbnb Analysis Overview')
st.subheader('Using a sample of 5,000 listings')

mapstyle = 'mapbox://styles/mapbox/streets-v12'

@st.cache_data(ttl=2*60*60)
def load_toronto_airbnb_data():
    listings_df_clean = pd.read_csv('listings_sample.csv')
    listings_df_clean['review_scores_rating'] = listings_df_clean['review_scores_rating'].fillna(0)
    reviews_df_clean = pd.read_csv('reviews_sample.csv')
    reviews_df_clean['commentsWC'] = reviews_df_clean['commentsWC'].apply(ast.literal_eval)
    events_df_clean = pd.read_csv('cleaned_events.csv', parse_dates=['date_started', 'date_ended'])
    calendar_df_clean = pd.read_csv('calendar_sample.csv', parse_dates=['date'])
    return listings_df_clean, reviews_df_clean, events_df_clean, calendar_df_clean

@st.cache_data(ttl=2*60*60)
def price_categorization(listings_df_clean:pd.DataFrame):
    listings_df_clean['price_label'] = pd.Categorical(listings_df_clean['price_label'], ordered=True,
                                                      categories=['Cheap', 'Average', 'Expensive'])
    price_labels = listings_df_clean.groupby('price_label', observed=False)['price'].agg(['min', 'max'])
    
    price_legend = []
    for label in price_labels.index:
        legend_str = f"{label} price: {price_labels.loc[label, 'min']} to {price_labels.loc[label, 'max']} dollars"
        price_legend.append(legend_str)
    
    neighborhood_count = listings_df_clean['neighbourhood_cleansed'].value_counts()
    top10_neighbourhoods = neighborhood_count.index[:10]
    bottom10_neighbourhoods = neighborhood_count.index[-10:]

    # count the price labels per neighbourhood
    price_count_per_neighborhood = listings_df_clean.groupby(['neighbourhood_cleansed', 'price_label'], observed=False).size()\
        .reset_index(name='count')
    # calculate the total count per neighborhood
    total_per_neighbourhood = price_count_per_neighborhood.groupby('neighbourhood_cleansed')['count'].transform('sum')
    # calculate the proportion of each price label in each neighborhoood
    price_count_per_neighborhood['proportion'] = price_count_per_neighborhood['count'] / total_per_neighbourhood
    
    # check the price labels in neighbourhoods with the highest and lowest number of listings
    price_cat_most_listing_neighborhooods = price_count_per_neighborhood[price_count_per_neighborhood['neighbourhood_cleansed'].isin(top10_neighbourhoods)]
    price_cat_least_listing_neighborhooods = price_count_per_neighborhood[price_count_per_neighborhood['neighbourhood_cleansed'].isin(bottom10_neighbourhoods)]
    
    return price_legend, top10_neighbourhoods, bottom10_neighbourhoods, price_cat_most_listing_neighborhooods, price_cat_least_listing_neighborhooods

@st.cache_data(ttl=2*60*60)
def summary_stats(listings_df_clean):
    median_review_score = listings_df_clean['review_scores_rating'].median()
    min_price = listings_df_clean['price'].min()
    median_price = listings_df_clean['price'].median()
    max_price = listings_df_clean['price'].max()
    return median_review_score, min_price, median_price, max_price

@st.cache_data(ttl=2*60*60)
def availability_and_events(calendar_df_clean, events_df_clean):
    # Calculate the mean availability per day across all listings
    availability_trend = calendar_df_clean.groupby('date')['available_numeric'].mean()
    # apply a rolling window for smoothing (7-day rolling average)
    availability_trend_rolling = availability_trend.rolling(window=7).mean().dropna().reset_index()
    events_df_clean['month_name'] = events_df_clean['date_started'].dt.month_name()
    # count month occurences
    events_per_month = events_df_clean['month_name'].value_counts()
    # create a list of months ordered from Dec to Jan
    month_list = [i for i in calendar.month_name if i]
    # reindex the month occurence
    events_per_month = events_per_month.reindex(month_list)

    return availability_trend_rolling, events_df_clean, month_list

@st.cache_data(ttl=2*60*60)
def month_proportion(calendar_df_clean, month_name='June'):
    month_availability = calendar_df_clean.loc[(calendar_df_clean['date'].dt.month_name() == month_name)]
    month_counts = month_availability.groupby(['available'])['listing_id'].nunique()
    total_month_counts = month_counts.sum()
    month_prop = month_counts/total_month_counts
    return month_prop

def render_map(mapstyle:str, layers:list, view_state:object, tooltip:dict):
    pdk_map = pdk.Deck(map_style=mapstyle, layers=layers, initial_view_state=view_state, tooltip=tooltip)
    return pdk_map

filter_col, blank_col1, blank_col2, blank_col3 = st.columns(4, gap='medium')
date_selector = blank_col1.date_input(label='Select a date')

def map_rendering(dataframe:pd.DataFrame):
    available_listings = calendar_df_clean.loc[(calendar_df_clean['date'] == str(date_selector)) &
    (calendar_df_clean['available'] == True), 'listing_id']
    available_df = listings_df_clean[listings_df_clean['id'].isin(available_listings)]
    unavailable_df = listings_df_clean[~(listings_df_clean['id'].isin(available_listings))]
    av_map_layer = pdk.Layer(type='ScatterplotLayer', data=available_df, get_position=['longitude','latitude'], pickable=True,
                          get_color=[3, 127, 81], get_radius=50, auto_highlight=True)
    unav_map_layer = pdk.Layer(type='ScatterplotLayer', data=unavailable_df, get_position=['longitude','latitude'], pickable=True,
                          get_color=[255, 0, 0], get_radius=50, auto_highlight=True)
    map_view_state = pdk.ViewState(longitude=-79.3883, latitude=43.6548,
                                   zoom=10, min_zoom=8, max_zoom=24, pitch=0)
    map = render_map(mapstyle=mapstyle, layers=[av_map_layer, unav_map_layer],  view_state=map_view_state,
                     tooltip={"text": """ID: {id}\nNeighbourhood: {neighbourhood_cleansed}\nPrice/night: ${price}\nMinimum nights: {minimum_nights}\nRating: {review_scores_rating}"""})
    return map

listings_df_clean, reviews_df_clean, events_df_clean, calendar_df_clean = load_toronto_airbnb_data()

price_legend, top10_neighbourhoods, bottom10_neighbourhoods, price_cat_most_listing_neighborhooods, price_cat_least_listing_neighborhooods \
    = price_categorization(listings_df_clean)

with filter_col:
    options = ['All'] + listings_df_clean['neighbourhood_cleansed'].unique().tolist()
    selection = st.selectbox('Select a neighbourhood', options=options)

    if selection != 'All':
        listings_df_clean = listings_df_clean[listings_df_clean['neighbourhood_cleansed'] == selection]
        ids = listings_df_clean['id'].unique()
        reviews_df_clean = reviews_df_clean[reviews_df_clean['listing_id'].isin(ids)]
        calendar_df_clean = calendar_df_clean[calendar_df_clean['listing_id'].isin(ids)]
    else:
        pass

median_review_score, min_price, median_price, max_price = summary_stats(listings_df_clean)

availability_trend_rolling, events_df_clean, month_list = availability_and_events(calendar_df_clean, events_df_clean)

june_prop = month_proportion(calendar_df_clean).reset_index(name='proportion')
jan_prop = month_proportion(calendar_df_clean, month_name='January').reset_index(name='proportion')
selected_month_prop = month_proportion(calendar_df_clean, month_name=date_selector.strftime("%B")).reset_index(name='proportion')

price_color_scale = {'Cheap': '#1f77b4', 'Average': '#A9A9A9', 'Expensive': '#D3D3D3'}
availability_color_scale = {True: '#008000', False: '#FF0000'}

map_column, eda_column = st.columns(2, gap='medium')

with eda_column:
    min_col, median_col, max_col = st.columns(3, gap='small')
    min_col.metric(value=f'${min_price:,.2f}', label='Minimum Price')
    median_col.metric(value=f'${median_price:,.2f}', label='Median Price')
    max_col.metric(value=f'${max_price:,.2f}', label='Maximum Price')
    with st.expander('**Price categories in neighborhoods with *many* listings**', expanded=True):
        fig1 = px.bar(price_cat_most_listing_neighborhooods, 
              x='proportion', y='neighbourhood_cleansed', color='price_label',
              labels={'neighbourhood_cleansed': 'Neighbourhood', 'proportion': 'Percentage', 'price_label': 'Price Label'},
              hover_data={'proportion': ':.2%'},
              color_discrete_map=price_color_scale)

        st.plotly_chart(fig1, use_container_width=True)
        st.write('; '.join(price_legend))

    with st.expander('**Price categories in neighborhoods with *few* listings**', expanded=True):
        fig2 = px.bar(price_cat_least_listing_neighborhooods, 
              x='proportion', y='neighbourhood_cleansed', color='price_label',
              color_discrete_map=price_color_scale,
              labels={'neighbourhood_cleansed': 'Neighbourhood', 'proportion': 'Percentage', 'price_label': 'Price Label'},
              hover_data={'proportion': ':.2%'})

        st.plotly_chart(fig2, use_container_width=True)
        st.write('; '.join(price_legend))
    
    col1, col2 = st.columns(2)
    with col1.expander('**Proportion of available listings in June (more events)**', expanded=True):
        fig3 = px.pie(june_prop, color_discrete_map=availability_color_scale,
              values='proportion', names='available', 
              hole=0.5, color='available')

        st.plotly_chart(fig3, use_container_width=True)

    with col2.expander('**Proportion of available listings in January (less events)**', expanded=True):
        fig4 = px.pie(jan_prop, color_discrete_map=availability_color_scale,
              values='proportion', names='available', 
              hole=0.5, color='available')

        st.plotly_chart(fig4, use_container_width=True)
    
    with col1.expander(f'**Proportion of available listings in {date_selector.strftime('%B')}**', expanded=True):
        fig5 = px.pie(selected_month_prop, color_discrete_map=availability_color_scale,
              values='proportion', names='available', 
              hole=0.5, color='available')
        
        st.plotly_chart(fig5, use_container_width=True)        

with map_column:
    st.metric(value=f'⭐ {median_review_score:.2f}', label='Median Rating')
    
    with st.expander('**Geographical distribution of Airbnb listings in Toronto**', expanded=True):
        map = map_rendering(listings_df_clean)
        st.pydeck_chart(map, use_container_width=True)
        st.markdown("""Listings are shown as points on the map. :red[Red] points are unavailable while :green[green]
                    points are available for the selected date""")
    
    col1, col2 = st.columns(2)
    with col1.expander('**Frequency of the different room types available**', expanded=True):
        room_type_counts = listings_df_clean['room_type'].value_counts().sort_values(ascending=True).to_frame().reset_index()
        fig6 = px.bar(room_type_counts, x='count', y='room_type', color_discrete_sequence=['black'])

        st.plotly_chart(fig6, use_container_width=True)

    with col2.expander('**What customers are saying**', expanded=True):
        comment_frequencies = Counter()
        for comments in reviews_df_clean['commentsWC']:
            comment_frequencies.update(comments)

        comment_frequencies.__delitem__('br')
        wordcloud_fig = plt.figure()
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10, colormap='Dark2')\
                    .generate_from_frequencies(comment_frequencies)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

        st.pyplot(wordcloud_fig)
    
    with col1.expander('**7-day rolling average of listings availability**', expanded=True):
        fig7 = px.line(availability_trend_rolling, 
               x='date', y='available_numeric',  color_discrete_sequence=['black'],
               labels={'date': 'Date', 'available_numeric': 'Average Availability (7-day rolling)'})

        st.plotly_chart(fig7, use_container_width=True)

    with col2.expander('**Number of events starting in different months**', expanded=True):
        events_count = events_df_clean['month_name'].value_counts().sort_values(ascending=True).to_frame().reset_index()
        fig8 = px.bar(events_count, 
              x='count', y='month_name', color_discrete_sequence=['black'],
              labels={'count': 'Count', 'month_name': 'Month Name'})

        st.plotly_chart(fig8, use_container_width=True)