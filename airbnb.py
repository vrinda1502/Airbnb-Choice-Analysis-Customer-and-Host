# import required modules
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud # for unstructured text visualization
import streamlit as st # for streamlit commands
import altair as alt # interactive viz on streamlit
import pydeck as pdk # interactive map
import pandas as pd
import calendar # for date analysis
import ast # for type conversions
from datetime import date

st.set_page_config(layout='wide', page_title='Toronto Airbnb Choice Analysis', page_icon='🏡')

st.title('Toronto Airbnb Analysis Overview')

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
    listings_df_clean['price_label'] = listings_df_clean['price_label'].cat.rename_categories({'Cheap': '1. Cheap', 'Average': '2. Average', 'Expensive':'3. Expensive'})
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
date_selector = blank_col1.date_input(label='Date')

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
    selection = st.selectbox('Neighbourhood', options=options)

    if selection != 'All':
        listings_df_clean = listings_df_clean[listings_df_clean['neighbourhood_cleansed'] == selection]
        ids = listings_df_clean['id'].unique()
        reviews_df_clean = reviews_df_clean[reviews_df_clean['listing_id'].isin(ids)]
        calendar_df_clean = calendar_df_clean[calendar_df_clean['listing_id'].isin(ids)]
    else:
        pass

median_review_score, min_price, median_price, max_price = summary_stats(listings_df_clean)

availability_trend_rolling, events_df_clean, month_list = availability_and_events(calendar_df_clean, events_df_clean)

june_prop = month_proportion(calendar_df_clean)
jan_prop = month_proportion(calendar_df_clean, month_name='January')
selected_month_prop = month_proportion(calendar_df_clean, month_name=date_selector.strftime("%B"))

price_color_scale = alt.Scale(domain=['1. Cheap', '2. Average', '3. Expensive'], range=['#1f77b4', '#A9A9A9', '#D3D3D3'])
availability_color_scale = alt.Scale(domain=[True, False], range=['green', 'red'])

map_column, eda_column = st.columns(2, gap='medium')

with eda_column:
    min_col, median_col, max_col = st.columns(3, gap='small')
    min_col.metric(value=f'${min_price:,.2f}', label='Minimum Price')
    median_col.metric(value=f'${median_price:,.2f}', label='Median Price')
    max_col.metric(value=f'${max_price:,.2f}', label='Maximum Price')
    with st.expander('**Price categories in neighborhoods with *many* listings**', expanded=True):
        chart = alt.Chart(price_cat_most_listing_neighborhooods).mark_bar().encode(
        x=alt.X('proportion:Q').title(None).axis(format="%"),
        y=alt.Y('neighbourhood_cleansed:O').title(None).axis(labelLimit=240),
        color=alt.Color('price_label:O', scale=price_color_scale).title("Price Label"),#legend(orient="bottom", titleOrient="left"),
        tooltip=[
            alt.Tooltip('neighbourhood_cleansed:N', title='Neighbourhood'),
            alt.Tooltip('price_label:N', title='Price Label'),
            alt.Tooltip('proportion:Q', title='Percentage', format='.2%')  # Format proportion as percentage with 2 decimals
        ]
    )
        st.altair_chart(chart, use_container_width=True)
        st.write('; '.join(price_legend))

    with st.expander('**Price categories in neighborhoods with *few* listings**', expanded=True):
        chart = alt.Chart(price_cat_least_listing_neighborhooods).mark_bar().encode(
        x=alt.X('proportion:Q').title(None).axis(format="%"),
        y=alt.Y('neighbourhood_cleansed:O').title(None).axis(labelLimit=240),
        color=alt.Color('price_label:O', scale=price_color_scale).title("Price Label"),#legend(orient="bottom", titleOrient="left"),
        tooltip=[
            alt.Tooltip('neighbourhood_cleansed:N', title='Neighbourhood'),
            alt.Tooltip('price_label:N', title='Price Label'),
            alt.Tooltip('proportion:Q', title='Percentage', format='.2%')  # Format proportion as percentage with 2 decimals
        ]
    )
        st.altair_chart(chart, use_container_width=True)
        st.write('; '.join(price_legend))
    
    col1, col2 = st.columns(2)
    with col1.expander('**Proportion of available listings in June (more events)**', expanded=True):
        chart = alt.Chart(june_prop.reset_index(name='proportion')).mark_arc(innerRadius=50).encode(
    theta="proportion",
    color=alt.Color("available:N", scale=availability_color_scale).title('Availability'),
    tooltip=[
            alt.Tooltip('available:N', title='Availability'),
            alt.Tooltip('proportion:Q', title='Percentage', format='.2%')  # Format proportion as percentage with 2 decimals
        ]
)
        st.altair_chart(chart, use_container_width=True)

    with col2.expander('**Proportion of available listings in January (less events)**', expanded=True):
        chart = alt.Chart(jan_prop.reset_index(name='proportion')).mark_arc(innerRadius=50).encode(
            theta="proportion",
            color=alt.Color("available:N", scale=availability_color_scale).title('Availability'),
            tooltip=[
            alt.Tooltip('available:N', title='Availability'),
            alt.Tooltip('proportion:Q', title='Percentage', format='.2%')  # Format proportion as percentage with 2 decimals
        ]
        )
        st.altair_chart(chart, use_container_width=True)
    
    with col1.expander(f'**Proportion of available listings in {date_selector.strftime('%B')}**', expanded=True):
        chart = alt.Chart(selected_month_prop.reset_index(name='proportion')).mark_arc(innerRadius=50).encode(
    theta="proportion",
    color=alt.Color("available:N", scale=availability_color_scale).title('Availability'),
    tooltip=[
            alt.Tooltip('available:N', title='Availability'),
            alt.Tooltip('proportion:Q', title='Percentage', format='.2%')  # Format proportion as percentage with 2 decimals
        ]
)
        st.altair_chart(chart, use_container_width=True)

with map_column:
    st.metric(value=f'⭐ {median_review_score:.2f}', label='Median Rating')
    
    with st.expander('**Geographical distribution of Airbnb listings in Toronto**', expanded=True):
        map = map_rendering(listings_df_clean)
        st.pydeck_chart(map, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1.expander('**Frequency of the different room types available**', expanded=True):
        chart = alt.Chart(listings_df_clean[['room_type']]).mark_bar(color='black').encode(
            y=alt.X('room_type:N', title=None),
            x=alt.Y('count():Q', title=None, axis=None),
            tooltip=[
                alt.Tooltip('room_type:N', title='Room Type'),
                alt.Tooltip('count():Q', title='Count'),
            ]
        )
        text = chart.mark_text(align='left', baseline='middle', dx=3).encode(text='count()')
        chart = chart + text
        st.altair_chart(chart, use_container_width=True)

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
        chart = alt.Chart(availability_trend_rolling).mark_line(color='black').encode(
        x=alt.X('date:T', title='Date'),  # Date on x-axis
        y=alt.Y('available_numeric:Q', title='Average Availability (7-day rolling)'),  # Rolling avg on y-axis
        tooltip=['date:T', 'availability_trend_rolling:Q']  # Tooltip for interactivity
    )
        st.altair_chart(chart, use_container_width=True)

    with col2.expander('**Number of events starting in different months**', expanded=True):
        chart = alt.Chart(events_df_clean).mark_bar(color='black').encode(
            y=alt.X('month_name:N', title=None, sort=month_list),
            x=alt.Y('count():Q', title=None, axis=None),
            tooltip=[
                alt.Tooltip('month_name:N', title='Room Type'),
                alt.Tooltip('count():Q', title='Count'),
            ]
        )
        text = chart.mark_text(align='left', baseline='middle', dx=3).encode(text='count()')
        chart = chart + text
        st.altair_chart(chart, use_container_width=True)