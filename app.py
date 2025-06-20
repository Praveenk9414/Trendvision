import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from datetime import datetime, timedelta
import pytz

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Set page config
st.set_page_config(
    page_title="TrendVision",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar Navigation (always visible) ---
st.sidebar.title("Main Menu")

# --- Update your sidebar navigation ---
if "page" not in st.session_state:
    st.session_state.page = "Home"
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# Add "Predictions" to the selectbox
selected = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Explore", "Trending Now", "Predictions"],
    index=["Home", "Explore", "Trending Now", "Predictions"].index(st.session_state.page)
)

if selected != st.session_state.page:
    st.session_state.page = selected
    st.rerun()

page = st.session_state.page


# Load data
@st.cache_data
def load_data():
    return pd.read_csv('dataset.csv', parse_dates=['Started', 'Ended'])


df = load_data()
df['Started_date'] = df['Started'].dt.date
df['duration'] = df['Ended'] - df['Started']
df['duration_days'] = df['duration'].dt.days
df['hour'] = df['Started'].dt.hour
df['day_name'] = df['Started'].dt.day_name()


# --- Helper Functions ---
def plot_trend_over_time(data, title):
    fig = px.line(data.sort_values('Started'),
                  x='Started', y='search_vol',
                  title=title,
                  labels={'search_vol': 'Search Volume', 'Started': 'Date'})
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    peak = data.loc[data['search_vol'].idxmax()]
    avg_vol = data['search_vol'].mean()
    st.caption(f"""
    📌 **Insights**: 
    - Peak search volume of {peak['search_vol']} occurred on {peak['Started'].strftime('%b %d, %Y')}
    - Average daily search volume: {avg_vol:.0f}
    - Trend {'increased' if data['search_vol'].iloc[-1] > data['search_vol'].iloc[0] else 'decreased'} over time
    """)


def plot_top_trends(data, title, n=10):
    top_trends = data.sort_values('search_vol', ascending=False).head(n)
    fig = px.bar(top_trends,
                 x='search_vol', y='Trends',
                 orientation='h',
                 title=title,
                 labels={'search_vol': 'Search Volume', 'Trends': 'Trend'})
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.caption(f"""
    📌 **Top {n} Trends**: 
    1. **{top_trends.iloc[0]['Trends']}** (Volume: {top_trends.iloc[0]['search_vol']})
    2. **{top_trends.iloc[1]['Trends']}** (Volume: {top_trends.iloc[1]['search_vol']})
    3. **{top_trends.iloc[2]['Trends']}** (Volume: {top_trends.iloc[2]['search_vol']})
    - Total search volume for top {n}: {top_trends['search_vol'].sum():,}
    - Average search volume: {top_trends['search_vol'].mean():.0f}
    """)


def plot_category_distribution(data):
    category_counts = data['categories'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    fig = px.pie(category_counts,
                 values='Count', names='Category',
                 title='Trend Distribution by Category',
                 hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    top_category = category_counts.iloc[0]
    st.caption(f"""
    📌 **Category Insights**: 
    - Most popular category: **{top_category['Category']}** ({top_category['Count']} trends)
    - Top 3 categories account for {category_counts.head(3)['Count'].sum() / category_counts['Count'].sum():.0%} of all trends
    - {category_counts['Category'].nunique()} unique categories in dataset
    """)


def plot_duration_vs_volume(data):
    fig = px.scatter(data,
                     x='duration_days', y='search_vol',
                     color='categories',
                     title='Duration vs Search Volume',
                     labels={'duration_days': 'Trend Duration (Days)', 'search_vol': 'Search Volume'})
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    corr = data['duration_days'].corr(data['search_vol'])
    st.caption(f"""
    📌 **Duration Insights**: 
    - Correlation between duration and search volume: {corr:.2f}
    - Average trend duration: {data['duration_days'].mean():.1f} days
    - Longest trend lasted {data['duration_days'].max()} days
    - Shortest trend lasted {data['duration_days'].min()} days
    """)


def plot_trends_by_hour(data):
    hourly_counts = data.groupby('hour').size().reset_index(name='count')
    fig = px.bar(hourly_counts,
                 x='hour', y='count',
                 title='Trends by Hour of Day',
                 labels={'hour': 'Hour of Day', 'count': 'Number of Trends'})
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    peak_hour = hourly_counts.loc[hourly_counts['count'].idxmax()]
    st.caption(f"""
    📌 **Hourly Insights**: 
    - Peak activity hour: **{peak_hour['hour']}:00** ({peak_hour['count']} trends)
    - Quietest hour: {hourly_counts.loc[hourly_counts['count'].idxmin()]['hour']}:00
    - {hourly_counts['count'].sum()} total trends recorded
    """)


# --- Home Page ---
if page == "Home":
    st.title("Welcome to TrendVision")
    st.markdown("Explore trending topics and gain insights from real Google Trends data.")

    # Main search bar that redirects to Explore page
    search_query = st.text_input(
        "🔍 Search for trends...",
        placeholder="Enter a topic (e.g., 'IPL')",
        key="home_search"
    )

    if search_query:
        st.session_state.search_query = search_query
        st.session_state.page = "Explore"
        st.rerun()

    st.subheader("🔥 Today's Top Trending Topics")
    top_trends = df.sort_values('search_vol', ascending=False).head(6)['Trends'].tolist()
    cols = st.columns(3)
    for i, topic in enumerate(top_trends[:6]):
        if cols[i % 3].button(topic, use_container_width=True):
            st.session_state.search_query = topic
            st.session_state.page = "Explore"
            st.rerun()

    st.subheader("📊 Quick Insights")
    key_feature = st.selectbox("Choose a Feature to View", [
        "Top Trends by Search Volume",
        "Trend Distribution by Category",
        "Interest Over Time (All Data)",
        "Duration vs Search Volume",
        "Trends by Hour"
    ])

    if key_feature == "Top Trends by Search Volume":
        st.subheader("Top Trends by Search Volume")
        plot_top_trends(df, "Top 10 Trends", 10)
    elif key_feature == "Trend Distribution by Category":
        st.subheader("Trend Distribution by Category")
        plot_category_distribution(df)
    elif key_feature == "Interest Over Time (All Data)":
        st.subheader("Search Interest Over Time")
        plot_trend_over_time(df, "Overall Interest Over Time")
    elif key_feature == "Duration vs Search Volume":
        st.subheader("Duration vs Search Volume")
        plot_duration_vs_volume(df)
    elif key_feature == "Trends by Hour":
        st.subheader("Trends by Hour of Day")
        plot_trends_by_hour(df)

# --- Explore Page ---
elif page == "Explore":
    st.title("Explore Trends")


    def show_trend_visualizations(data):
        """Show specialized visualizations for a specific trend search"""
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Trend Timeline", "Hourly Patterns", "Duration Analysis", "Raw Data"])

        with tab1:
            st.subheader("Trend Timeline Analysis")

            # Main trend line with smoothing
            fig = px.line(
                data.sort_values('Started'),
                x='Started',
                y='search_vol',
                title=f'Search Volume Over Time (Smoothed)',
                line_shape='spline'
            )
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Search Volume',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            peak = data.loc[data['search_vol'].idxmax()]
            st.caption(f"""
            📌 **Timeline Insights**: 
            - Peak interest: **{peak['search_vol']}** searches on {peak['Started'].strftime('%b %d')}
            - Active for {data['duration_days'].mean():.1f} days on average
            - {'Rapid growth' if data['search_vol'].diff().mean() > 0 else 'Gradual decline'} pattern observed
            """)

            # Timeline visualization
            fig = px.timeline(
                data,
                x_start='Started',
                x_end='Ended',
                y='Trends',
                title='Active Periods for This Trend'
            )
            fig.update_yaxes(autorange='reversed')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Hourly and Daily Patterns")
            col1, col2 = st.columns(2)

            with col1:
                # Search volume by hour (spline)
                hourly_data = data.groupby('hour')['search_vol'].mean().reset_index()
                fig = px.line(
                    hourly_data,
                    x='hour',
                    y='search_vol',
                    title='Search Volume by Hour of Day',
                    line_shape='spline'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                peak_hour = hourly_data.loc[hourly_data['search_vol'].idxmax()]
                st.caption(f"""
                📌 **Hourly Pattern**: 
                - Peak search time: **{peak_hour['hour']}:00** ({peak_hour['search_vol']:.0f} avg volume)
                - Lowest activity at {(24 - data['hour'].mode()[0]) % 24}:00
                """)

            with col2:
                # Search volume by day of week
                fig = px.bar(
                    data,
                    x='day_name',
                    y='search_vol',
                    title='Search Volume by Day of Week',
                    category_orders={
                        'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                peak_day = data.groupby('day_name')['search_vol'].mean().idxmax()
                st.caption(f"""
                📌 **Weekly Pattern**: 
                - Most active day: **{peak_day}**
                - Weekend vs weekday difference: {data[data['day_name'].isin(['Saturday', 'Sunday'])]['search_vol'].mean() / data[~data['day_name'].isin(['Saturday', 'Sunday'])]['search_vol'].mean():.0%}
                """)

            # Heatmap of hour vs day
            try:
                pivot = data.pivot_table(
                    index='day_name',
                    columns='hour',
                    values='search_vol',
                    aggfunc='mean'
                )
                fig = px.imshow(
                    pivot,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Search Volume"),
                    title="Search Volume Heatmap (Hour vs Day)",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                max_val = pivot.max().max()
                max_idx = pivot.stack().idxmax()
                st.caption(f"""
                📌 **Heatmap Insights**: 
                - Highest activity: **{max_idx[1]}:00 on {max_idx[0]}** ({max_val:.0f} avg volume)
                - Clear daily pattern visible with peak at {pivot.mean(axis=0).idxmax()}:00
                """)
            except:
                st.warning("Insufficient data to generate heatmap")

        with tab3:
            st.subheader("Duration Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Duration distribution
                fig = px.histogram(
                    data,
                    x='duration_days',
                    nbins=20,
                    title='Duration Distribution (Days)',
                    labels={'duration_days': 'Duration in Days'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.caption(f"""
                📌 **Duration Distribution**: 
                - Most common duration: {data['duration_days'].mode()[0]} days
                - {data[data['duration_days'] > 7]['duration_days'].count()} trends lasted over 1 week
                """)

            with col2:
                # Duration vs Search Volume
                fig = px.scatter(
                    data,
                    x='duration_days',
                    y='search_vol',
                    title='Duration vs Search Volume',
                    trendline="lowess"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                corr = data['duration_days'].corr(data['search_vol'])
                st.caption(f"""
                📌 **Duration-Volume Relationship**: 
                - Correlation coefficient: {corr:.2f}
                - {'Longer' if corr > 0 else 'Shorter'} trends tend to have higher search volume
                """)

        with tab4:
            st.subheader("Raw Data")
            st.dataframe(data.sort_values('search_vol', ascending=False))
            st.caption(f"Showing {len(data)} records matching your search")


    def show_category_visualizations(data):
        """Show category visualizations with insights"""
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Main Visualizations", "Temporal Patterns", "Duration Analysis", "Raw Data"])

        with tab1:
            # Add two columns - one for timeline, one for sunburst
            col1, col2 = st.columns(2)

            with col1:
                # Timeline visualization
                fig = px.timeline(
                    data,
                    x_start='Started',
                    x_end='Ended',
                    y='Trends',
                    title='Trend Timelines'
                )
                fig.update_yaxes(autorange='reversed')
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.caption(f"""
                        📌 **Category Overview**: 
                        - Contains {len(data)} total trends
                        - Time range: {data['Started'].min().strftime('%b %Y')} to {data['Started'].max().strftime('%b %Y')}
                        - Average search volume: {data['search_vol'].mean():.0f}
                        """)

            with col2:
                # Sunburst chart
                st.subheader("Category Breakdown")
                tempdf = data.sort_values(by='search_vol', ascending=False)
                fig = px.sunburst(
                    tempdf,
                    path=['categories', 'Trends'],
                    values='search_vol',
                    color='search_vol',
                    color_continuous_scale='Blues',
                    title='Trends by Category and Volume'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Calculate insights for the sunburst
                category_vol = tempdf.groupby('categories')['search_vol'].sum().sort_values(ascending=False)
                top_category = category_vol.idxmax()
                top_category_pct = (category_vol.max() / category_vol.sum()) * 100
                top_trend = tempdf.iloc[0]
                diversity_score = len(tempdf['categories'].unique()) / len(tempdf)  # Categories per trend

                st.caption(f"""
                📌 **Sunburst Insights**: 
                - Dominant Category: **{top_category}** accounts for {top_category_pct:.1f}% of total search volume
                - Top Trend: **{top_trend['Trends']}** (Volume: {top_trend['search_vol']:,}) in {top_trend['categories']}
                - Category Diversity: {len(tempdf['categories'].unique())} categories ({diversity_score:.1%} per trend)
                - Volume Distribution: Top 3 categories make up {category_vol.head(3).sum() / category_vol.sum():.0%} of searches
                - Color Intensity: Darker shades indicate higher search volumes
                """)


        with tab2:
            st.subheader("Temporal Patterns")
            col1, col2 = st.columns(2)

            with col1:
                # Heatmap
                try:
                    pivot = data.pivot_table(
                        index='day_name',
                        columns='hour',
                        values='search_vol',
                        aggfunc='mean'
                    )
                    fig = px.imshow(
                        pivot,
                        labels=dict(x="Hour", y="Day", color="Search Volume"),
                        title="Search Volume by Day and Hour",
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Insights
                    max_val = pivot.max().max()
                    max_idx = pivot.stack().idxmax()
                    st.caption(f"""
                    📌 **Peak Activity**: 
                    - Best time: **{max_idx[1]}:00 on {max_idx[0]}**
                    - {max_val:.0f} average search volume
                    """)
                except:
                    st.warning("Could not generate heatmap")

            with col2:
                # Hourly pattern
                hourly_data = data.groupby('hour')['search_vol'].mean().reset_index()
                trace = go.Scatter(
                    x=hourly_data['hour'],
                    y=hourly_data['search_vol'],
                    mode='lines',
                    line_shape='spline'
                )
                fig = go.Figure([trace])
                fig.update_layout(
                    title='Search Volume by Hour of Day',
                    xaxis_title='Hour',
                    yaxis_title='Search Volume'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.caption(f"""
                📌 **Hourly Trend**: 
                - {hourly_data['search_vol'].max() / hourly_data['search_vol'].min():.1f}x difference between peak and low
                - Morning ({hourly_data.loc[hourly_data['hour'].between(6, 12), 'search_vol'].mean():.0f}) vs Evening ({hourly_data.loc[hourly_data['hour'].between(18, 23), 'search_vol'].mean():.0f}) comparison
                """)

            # Day of week visualization
            fig = px.bar(
                data,
                x='day_name',
                y='search_vol',
                title='Trend Frequency by Day of Week'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.caption(f"""
            📌 **Weekly Pattern**: 
            - {data.groupby('day_name')['search_vol'].mean().idxmax()} has highest activity
            - Weekend drop of {1 - data[data['day_name'].isin(['Saturday', 'Sunday'])]['search_vol'].mean() / data[~data['day_name'].isin(['Saturday', 'Sunday'])]['search_vol'].mean():.0%}
            """)


        with tab3:
            st.subheader("Duration Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Box plot
                fig = px.box(
                    data,
                    y='duration_days',
                    title='Duration Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.caption(f"""
                📌 **Duration Stats**: 
                - Median: {data['duration_days'].median()} days
                - IQR: {data['duration_days'].quantile(0.25)}-{data['duration_days'].quantile(0.75)} days
                - Longest: {data['duration_days'].max()} days
                """)

            with col2:
                # Duration histogram
                fig = px.histogram(
                    data,
                    x='duration_days',
                    nbins=10,
                    title='Trend Duration (in Days)',
                    labels={'duration_days': 'Duration (Days)'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.caption(f"""
                📌 **Duration Distribution**: 
                - {data[data['duration_days'] <= 1]['duration_days'].count()} one-day trends
                - {data[data['duration_days'] > 7]['duration_days'].count()} week+ trends
                """)

            st.subheader("Top Trends in This Category")
            # Top trends bar chart
            top_n = data.sort_values('search_vol', ascending=False).head(10)
            fig = px.bar(
                top_n,
                x='Trends',
                y='search_vol',
                title='Top 10 Most Searched Trends'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.caption(f"""
            📌 **Top Trends**: 
            1. **{top_n.iloc[0]['Trends']}** ({top_n.iloc[0]['search_vol']})
            2. **{top_n.iloc[1]['Trends']}** ({top_n.iloc[1]['search_vol']})
            3. **{top_n.iloc[2]['Trends']}** ({top_n.iloc[2]['search_vol']})
            """)

        with tab4:
            st.subheader("Raw Data")
            st.dataframe(data.sort_values('search_vol', ascending=False))
            st.caption(f"Showing {len(data)} trends in this category")


    def show_trend_comparison(data):
        """Show comparison visualizations for selected trends"""
        st.subheader("Trend Comparison Analysis")

        # Create tabs for different comparison views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overview", "Volume Analysis", "Temporal Patterns", "Duration Analysis", "Raw Data"])

        with tab1:
            st.subheader("Overview Comparison")
            col1, col2 = st.columns(2)

            with col1:
                # Treemap visualization
                fig = px.treemap(
                    data,
                    path=['Trends'],
                    values='search_vol',
                    title='Search Volume Distribution (Treemap)'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                if len(data) > 0:
                    st.caption(f"""
                    📌 **Market Share**: 
                    - {data.iloc[0]['Trends']} accounts for {data.iloc[0]['search_vol'] / data['search_vol'].sum():.0%} of total volume
                    - Top 2 trends make up {(data.iloc[0]['search_vol'] + data.iloc[1]['search_vol']) / data['search_vol'].sum():.0%}
                    """)

            with col2:
                # Pie/donut chart
                fig = px.pie(
                    data,
                    names='Trends',
                    values='search_vol',
                    title='Search Volume Share',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                if len(data) > 0:
                    st.caption(f"""
                    📌 **Volume Distribution**: 
                    - Average search volume: {data['search_vol'].mean():.0f}
                    - Range: {data['search_vol'].min()} to {data['search_vol'].max()}
                    """)

        with tab2:
            st.subheader("Search Volume Comparison")
            # Bar chart comparison
            fig = px.bar(
                data,
                x='Trends',
                y='search_vol',
                title='Search Volume Comparison',
                color='Trends'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            if len(data) > 0:
                st.caption(f"""
                📌 **Volume Comparison**: 
                - {data.iloc[0]['Trends']} is {data.iloc[0]['search_vol'] / data.iloc[1]['search_vol']:.1f}x more popular than {data.iloc[1]['Trends']}
                - Total volume across all: {data['search_vol'].sum():,}
                """)

            # Line chart over time
            fig = px.line(
                data.sort_values('Started'),
                x='Started',
                y='search_vol',
                color='Trends',
                title='Search Volume Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights - Modified this section to handle empty data
            if len(data) > 0:
                try:
                    peak_trend = data.loc[data.groupby('Trends')['search_vol'].idxmax()].sort_values('Started',
                                                                                                     ascending=False).iloc[
                        0]['Trends']
                    st.caption(f"""
                    📌 **Temporal Patterns**: 
                    - {peak_trend} had the most recent peak
                    - Seasonal patterns visible in some trends
                    """)
                except:
                    st.caption("📌 **Temporal Patterns**: Unable to determine peak trends")

        with tab3:
            st.subheader("Temporal Patterns")
            # Heatmap
            try:
                heat_df = data.pivot_table(
                    index='Trends',
                    columns='hour',
                    values='search_vol',
                    aggfunc='mean'
                )
                fig = px.imshow(
                    heat_df,
                    labels=dict(x="Hour", y="Trend", color="Search Volume"),
                    title="Hourly Search Volume by Trend",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights
                st.caption(f"""
                📌 **Hourly Differences**: 
                - Different trends show distinct daily patterns
                - Some peak in morning, others in evening
                """)
            except:
                st.warning("Could not generate heatmap - insufficient data")

            # Day of week comparison
            fig = px.box(
                data,
                x='Trends',
                y='search_vol',
                color='day_name',
                title='Search Volume by Day of Week'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            if len(data) > 0:
                try:
                    strongest_day = \
                    data.groupby(['Trends', 'day_name'])['search_vol'].mean().groupby('Trends').idxmax()[0][1]
                    st.caption(f"""
                    📌 **Weekly Patterns**: 
                    - Weekday vs weekend differences vary by trend
                    - {strongest_day} is consistently the strongest day
                    """)
                except:
                    st.caption("📌 **Weekly Patterns**: Unable to determine strongest day")

        with tab4:
            st.subheader("Duration Analysis")
            # Box plot of durations
            fig = px.box(
                data,
                x='Trends',
                y='duration_days',
                title='Duration Distribution by Trend',
                labels={'duration_days': 'Duration (Days)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            if len(data) > 0:
                try:
                    longest_trend = data.groupby('Trends')['duration_days'].mean().idxmax()
                    shortest_trend = data.groupby('Trends')['duration_days'].mean().idxmin()
                    st.caption(f"""
                    📌 **Duration Comparison**: 
                    - {longest_trend} lasts longest on average
                    - {shortest_trend} is most short-lived
                    """)
                except:
                    st.caption("📌 **Duration Comparison**: Unable to compare durations")

            # Scatter plot of duration vs volume
            fig = px.scatter(
                data,
                x='duration_days',
                y='search_vol',
                color='Trends',
                title='Duration vs Search Volume',
                trendline="lowess"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.caption(f"""
            📌 **Duration-Volume Relationship**: 
            - Correlation varies by trend
            - Some show strong relationship, others don't
            """)

        with tab5:
            st.subheader("Raw Data")
            st.dataframe(data.sort_values('search_vol', ascending=False))
            st.caption(f"Comparing {len(data['Trends'].unique())} trends")


    # Main page content
    option = st.radio(
        "Choose how you want to explore trends:",
        ["Search a specific trend", "Browse by category", "Compare multiple trends"],
        horizontal=True
    )

    if option == "Search a specific trend":
        search_query = st.text_input(
            "🔍 Enter a trend to search:",
            value=st.session_state.search_query,
            key="explore_search"
        )

        if search_query and search_query != st.session_state.search_query:
            st.session_state.search_query = search_query
            st.rerun()

        if st.session_state.search_query:
            st.subheader(f"Results for: {st.session_state.search_query}")
            filtered_df = df[df['Trends'].str.contains(st.session_state.search_query, case=False, na=False)]

            if not filtered_df.empty:
                show_trend_visualizations(filtered_df)
            else:
                st.warning("No results found for this search term")

    elif option == "Browse by category":
        all_categories = sorted(df['categories'].unique().tolist())
        selected_category = st.selectbox(
            "Select a category to explore:",
            all_categories
        )

        if selected_category:
            st.subheader(f"Showing trends in category: {selected_category}")
            filtered_df = df[df['categories'] == selected_category]
            show_category_visualizations(filtered_df)

    else:  # Compare multiple trends
        st.subheader("Compare Multiple Trends")

        # Show top 10 trends as suggestions
        st.write("### Top 10 Trending Topics")
        top_trends = df.sort_values('search_vol', ascending=False).head(10)

        # Display top trends in columns with select buttons
        cols = st.columns(5)
        for i, trend in enumerate(top_trends['Trends']):
            if cols[i % 5].button(trend, key=f"trend_{i}"):
                if 'selected_trends' not in st.session_state:
                    st.session_state.selected_trends = []
                if trend not in st.session_state.selected_trends:
                    st.session_state.selected_trends.append(trend)

        # Multi-select for trends
        all_trends = sorted(df['Trends'].unique().tolist())
        selected_trends = st.multiselect(
            "Select trends to compare (2-5 recommended):",
            all_trends,
            default=st.session_state.get('selected_trends', [])
        )

        if len(selected_trends) > 0:
            # Prepare comparison data
            comparison_df = df[df['Trends'].isin(selected_trends)].copy()
            show_trend_comparison(comparison_df)
        else:
            st.warning("Please select at least 1 trend to compare")

# --- Trending Now Page ---
elif page == "Trending Now":
    st.title("🔥 Trending Now")
    # Full dataset view with filtering
    st.subheader("📊 Full Trend Dataset")

    # Add filters
    with st.expander("🔍 Filter Data"):
        col1, col2, col3 = st.columns(3)

        with col1:
            date_range = st.date_input(
                "Date Range",
                value=[df['Started'].min().date(), df['Started'].max().date()],
                min_value=df['Started'].min().date(),
                max_value=df['Started'].max().date()
            )

        with col2:
            min_volume = st.slider(
                "Minimum Search Volume",
                min_value=int(df['search_vol'].min()),
                max_value=int(df['search_vol'].max()),
                value=int(df['search_vol'].quantile(0.25))
            )

        with col3:
            categories = st.multiselect(
                "Categories",
                options=df['categories'].unique(),
                default=[]
            )

    # Apply filters
    filtered_data = df[
        (df['Started'].dt.date >= date_range[0]) &
        (df['Started'].dt.date <= date_range[1]) &
        (df['search_vol'] >= min_volume)
        ].copy()

    if categories:
        filtered_data = filtered_data[filtered_data['categories'].isin(categories)]


    # Calculate human-readable duration
    def format_duration(duration):
        total_seconds = duration.total_seconds()
        days = int(total_seconds // 86400)
        hours = int((total_seconds % 86400) // 3600)

        parts = []
        if days > 0:
            parts.append(f"{days} {'month' if days >= 30 else 'day'}{'s' if days != 1 else ''}")
            if days >= 30:
                months = days // 30
                days_remaining = days % 30
                parts = [f"{months} month{'s' if months != 1 else ''}"]
                if days_remaining > 0:
                    parts.append(f"{days_remaining} day{'s' if days_remaining != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")

        if not parts:  # if duration is less than 1 hour
            minutes = int((total_seconds % 3600) // 60)
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            else:
                parts.append("Less than 1 minute")

        return ' '.join(parts)


    filtered_data['duration_readable'] = filtered_data['duration'].apply(format_duration)

    # Select and rename columns for display
    display_columns = {
        'Trends': 'Trend',
        'Started': 'Start Time',
        'Ended': 'End Time',
        'search_vol': 'Search Volume',
        'duration_readable': 'Duration',
        'categories': 'Category'
    }

    # Create the filtered dataframe with only the columns we want
    display_data = filtered_data[list(display_columns.keys())].rename(columns=display_columns)

    # Sort by search volume descending
    display_data = display_data.sort_values('Search Volume', ascending=False)

    # Show the data in an interactive table
    st.dataframe(
        display_data,
        column_config={
            "Start Time": st.column_config.DatetimeColumn("Start Time"),
            "End Time": st.column_config.DatetimeColumn("End Time"),
            "Search Volume": st.column_config.NumberColumn("Search Volume", format="%d"),
            "Duration": "Duration"
        },
        hide_index=True,
        use_container_width=True
    )

    # Download button - include all original data in download
    st.download_button(
        label="Download Filtered Data",
        data=filtered_data.to_csv(index=False).encode('utf-8'),
        file_name='filtered_trends.csv',
        mime='text/csv'
    )

# --- Predictions ---
elif page == "Predictions":
    st.title("🔮 Trend Predictor")
    st.markdown("Predict future popularity of trends and discover similar trends")

    # Feature Engineering
    df['duration_hrs'] = df['duration'].dt.total_seconds() / 3600
    category_map = {cat: i for i, cat in enumerate(df['categories'].unique())}
    df['category_seq'] = df['categories'].map(category_map)
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['day_num'] = df['day_name'].map(day_map)

    # Model Training
    features = ['duration_hrs', 'category_seq', 'day_num']
    X = df[features]
    y = df['search_vol']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction Interface
    with st.form("prediction_form"):
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Input Parameters")
            duration_hrs = st.slider("Duration (hours)", 1, 168, 24)
            category = st.selectbox("Category", sorted(df['categories'].unique()))
            day = st.selectbox("Day of Week", list(day_map.keys()))

            if st.form_submit_button("Predict Popularity"):
                # Make prediction
                category_num = category_map[category]
                day_num = day_map[day]
                prediction = model.predict([[duration_hrs, category_num, day_num]])[0]

                # Store in session state
                st.session_state['prediction_result'] = {
                    'predicted_volume': prediction,
                    'category': category,
                    'duration': duration_hrs,
                    'day': day
                }

        with col2:
            st.subheader("Prediction Insights")

            if 'prediction_result' in st.session_state:
                res = st.session_state['prediction_result']

                # Prediction Card
                st.metric("Predicted Search Volume", f"{res['predicted_volume']:,.0f}")

                # Find similar trends
                similar_trends = df[
                    (df['categories'] == res['category']) &
                    (df['duration'].dt.total_seconds() / 3600).between(
                        res['duration'] * 0.8, res['duration'] * 1.2)
                    ].sort_values('search_vol', ascending=False)

                # Category Analysis
                st.write(f"**🔍 Trends in {res['category']}**")
                if not similar_trends.empty:
                    top_trend = similar_trends.iloc[0]
                    avg_vol = similar_trends['search_vol'].mean()

                    st.caption(f"""
                    - Top trend: **{top_trend['Trends']}** ({top_trend['search_vol']:,.0f} searches)
                    - Average volume: {avg_vol:,.0f}
                    - Your prediction is **{'above' if res['predicted_volume'] > avg_vol else 'below'}** average
                    """)

                    # Show sample trends
                    with st.expander("View Similar Trends"):
                        st.dataframe(
                            similar_trends[['Trends', 'search_vol', 'duration']].head(10),
                            column_config={
                                "Trends": "Trend Name",
                                "search_vol": st.column_config.NumberColumn("Searches", format="%,d"),
                                "duration": "Duration"
                            }
                        )
                else:
                    st.warning("No similar trends found in this category")

                # Duration Analysis
                duration_days = res['duration'] / 24
                st.write(f"**⏳ Duration Analysis**")
                st.caption(f"""
                - {duration_days:.1f} days predicted
                - Most trends in this category last {df[df['categories'] == res['category']]['duration'].mean().total_seconds() / 86400:.1f} days
                """)

    # Model Performance Section
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.metric("RMSE", f"{rmse:.2f}", help="Lower is better")

    # Visualize predictions vs actual
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title="Actual vs Predicted Search Volumes"
    )
    st.plotly_chart(fig, use_container_width=True)