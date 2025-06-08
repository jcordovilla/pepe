"""
Query Analytics Dashboard for PEPE Discord Bot

Streamlit dashboard for viewing and analyzing user queries and agent responses.
Provides insights into query patterns, performance metrics, and system usage.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List

# Set up page configuration
st.set_page_config(
    page_title="PEPE Query Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import query logging functions
try:
    from db.query_logs import get_recent_queries, get_query_analytics, query_log_manager
    from db.db import get_db_session
    from db.query_logs import QueryLog
except ImportError as e:
    st.error(f"Failed to import query logging modules: {e}")
    st.stop()

def load_query_data(days: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
    """Load recent query data."""
    try:
        return get_recent_queries(hours=days*24, limit=limit)
    except Exception as e:
        st.error(f"Failed to load query data: {e}")
        return []

def display_overview_metrics(analytics: Dict[str, Any]):
    """Display overview metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Queries",
            value=analytics.get('total_queries', 0)
        )
    
    with col2:
        success_rate = analytics.get('success_rate', 0)
        st.metric(
            label="Success Rate",
            value=f"{success_rate:.1%}"
        )
    
    with col3:
        avg_time = analytics.get('average_processing_time_ms', 0)
        st.metric(
            label="Avg Response Time",
            value=f"{avg_time:.0f}ms"
        )
    
    with col4:
        strategies = analytics.get('strategy_distribution', {})
        most_used = max(strategies.keys(), key=lambda k: strategies[k]) if strategies else "N/A"
        st.metric(
            label="Most Used Strategy",
            value=most_used.replace('_', ' ').title() if most_used != "N/A" else "N/A"
        )

def plot_query_timeline(queries: List[Dict[str, Any]]):
    """Plot query timeline."""
    if not queries:
        st.info("No query data available for timeline.")
        return
    
    df = pd.DataFrame(queries)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.floor('H')
    
    # Group by hour and count queries
    hourly_counts = df.groupby('hour').size().reset_index(name='count')
    
    fig = px.line(
        hourly_counts, 
        x='hour', 
        y='count',
        title='Query Volume Over Time',
        labels={'hour': 'Time', 'count': 'Number of Queries'}
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Queries",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_strategy_distribution(analytics: Dict[str, Any]):
    """Plot strategy distribution pie chart."""
    strategies = analytics.get('strategy_distribution', {})
    
    if not strategies:
        st.info("No strategy data available.")
        return
    
    # Clean up strategy names
    cleaned_strategies = {
        strategy.replace('_', ' ').title(): count 
        for strategy, count in strategies.items()
    }
    
    fig = px.pie(
        values=list(cleaned_strategies.values()),
        names=list(cleaned_strategies.keys()),
        title='Query Strategy Distribution'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_response_times(queries: List[Dict[str, Any]]):
    """Plot response time distribution."""
    if not queries:
        st.info("No response time data available.")
        return
    
    df = pd.DataFrame(queries)
    
    # Filter out None values and outliers
    response_times = [
        q.get('processing_time_ms') for q in queries 
        if q.get('processing_time_ms') is not None and q.get('processing_time_ms') < 30000  # Less than 30 seconds
    ]
    
    if not response_times:
        st.info("No valid response time data available.")
        return
    
    fig = px.histogram(
        x=response_times,
        nbins=30,
        title='Response Time Distribution',
        labels={'x': 'Response Time (ms)', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Response Time (ms)",
        yaxis_title="Number of Queries",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_recent_queries_table(queries: List[Dict[str, Any]], limit: int = 20):
    """Display recent queries in a table."""
    if not queries:
        st.info("No recent queries to display.")
        return
    
    # Prepare data for table
    table_data = []
    for query in queries[-limit:]:  # Show most recent first
        table_data.append({
            'Timestamp': query.get('timestamp', ''),
            'User': query.get('username', 'Unknown'),
            'Query': query.get('query_text', '')[:100] + ('...' if len(query.get('query_text', '')) > 100 else ''),
            'Strategy': query.get('routing_strategy', 'Unknown').replace('_', ' ').title(),
            'Confidence': f"{query.get('confidence_score', 0):.2f}" if query.get('confidence_score') else 'N/A',
            'Status': query.get('response_status', 'Unknown'),
            'Time (ms)': query.get('processing_time_ms', 'N/A')
        })
    
    df = pd.DataFrame(table_data)
    
    # Reverse to show most recent first
    df = df.iloc[::-1].reset_index(drop=True)
    
    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )

def display_query_details(queries: List[Dict[str, Any]]):
    """Display detailed query information."""
    if not queries:
        st.info("No queries to display details for.")
        return
    
    st.subheader("Query Details")
    
    # Select query to view
    query_options = [
        f"{q.get('timestamp', 'Unknown')[:19]} - {q.get('username', 'Unknown')}: {q.get('query_text', '')[:50]}..."
        for q in queries[-50:]  # Last 50 queries
    ]
    
    if not query_options:
        st.info("No queries available.")
        return
    
    selected_idx = st.selectbox(
        "Select a query to view details:",
        range(len(query_options)),
        format_func=lambda i: query_options[i]
    )
    
    if selected_idx is not None:
        selected_query = queries[-(50-selected_idx)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Query Information:**")
            st.write(f"**User:** {selected_query.get('username', 'Unknown')}")
            st.write(f"**Timestamp:** {selected_query.get('timestamp', 'Unknown')}")
            st.write(f"**Query Type:** {selected_query.get('query_type', 'Unknown')}")
            st.write(f"**Strategy:** {selected_query.get('routing_strategy', 'Unknown').replace('_', ' ').title()}")
            st.write(f"**Confidence:** {selected_query.get('confidence_score', 'N/A')}")
            st.write(f"**Status:** {selected_query.get('response_status', 'Unknown')}")
            
        with col2:
            st.write("**Performance Metrics:**")
            st.write(f"**Processing Time:** {selected_query.get('processing_time_ms', 'N/A')} ms")
            st.write(f"**Query Length:** {selected_query.get('query_length', 'N/A')} chars")
            st.write(f"**Response Length:** {selected_query.get('response_length', 'N/A')} chars")
            st.write(f"**Successful:** {selected_query.get('is_successful', 'Unknown')}")
            
        st.write("**Full Query:**")
        st.text_area(
            "Query Text",
            value=selected_query.get('query_text', ''),
            height=100,
            disabled=True
        )

def main():
    st.title("📊 PEPE Query Analytics Dashboard")
    st.markdown("Monitor and analyze user queries and agent responses")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    days = st.sidebar.slider(
        "Days to analyze",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to include in the analysis"
    )
    
    max_queries = st.sidebar.slider(
        "Max queries to load",
        min_value=50,
        max_value=500,
        value=200,
        help="Maximum number of recent queries to load"
    )
    
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (30s)",
        value=False,
        help="Automatically refresh data every 30 seconds"
    )
    
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 30 seconds")
    
    # Load data
    with st.spinner("Loading query data..."):
        analytics = get_query_analytics(days=days)
        queries = load_query_data(days=days, limit=max_queries)
    
    if not analytics and not queries:
        st.error("Failed to load any data. Please check the database connection.")
        return
    
    # Display overview metrics
    st.header("📈 Overview")
    if analytics:
        display_overview_metrics(analytics)
    else:
        st.warning("Analytics data not available")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📋 Recent Queries", "🔍 Query Details", "🏥 System Health"])
    
    with tab1:
        st.header("📊 Analytics Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_query_timeline(queries)
            
        with col2:
            plot_strategy_distribution(analytics)
        
        plot_response_times(queries)
    
    with tab2:
        st.header("📋 Recent Queries")
        display_recent_queries_table(queries, limit=50)
    
    with tab3:
        display_query_details(queries)
    
    with tab4:
        st.header("🏥 System Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Database Status")
            try:
                with get_db_session() as session:
                    query_count = session.query(QueryLog).count()
                    st.success(f"✅ Database connected")
                    st.info(f"📊 Total queries in database: {query_count:,}")
            except Exception as e:
                st.error(f"❌ Database error: {e}")
        
        with col2:
            st.subheader("Recent Activity")
            if queries:
                recent_activity = len([
                    q for q in queries 
                    if datetime.fromisoformat(q['timestamp'].replace('Z', '+00:00')) > datetime.now().replace(tzinfo=None) - timedelta(hours=1)
                ])
                st.info(f"🕐 Queries in last hour: {recent_activity}")
                
                error_rate = len([q for q in queries if not q.get('is_successful', True)]) / len(queries) if queries else 0
                st.info(f"❌ Error rate: {error_rate:.1%}")
            else:
                st.warning("No recent activity data")
    
    # Auto-refresh
    if auto_refresh:
        st.rerun()

if __name__ == "__main__":
    main()
