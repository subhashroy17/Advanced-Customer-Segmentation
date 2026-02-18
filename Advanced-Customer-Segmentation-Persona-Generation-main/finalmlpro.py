import streamlit as st
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Page Config ---
st.set_page_config(
    page_title="Customer Persona Engine",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stApp header {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding='utf-8')

def calculate_rfm(df):
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    max_date = df['ORDERDATE'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('CUSTOMERNAME').agg({
        'ORDERDATE': lambda x: (max_date - x.max()).days,
        'ORDERNUMBER': 'nunique',
        'SALES': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerName', 'Recency', 'Frequency', 'Monetary']
    return rfm

def normalize_centroids(centroids):
    """Normalize cluster centers for the radar chart."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(centroids)
    # Min-Max scaling to 0-1 range for better radar visualization
    from sklearn.preprocessing import MinMaxScaler
    min_max = MinMaxScaler()
    return pd.DataFrame(min_max.fit_transform(centroids), columns=centroids.columns)

# --- Main App ---
def main():
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type=['csv'])
    
    st.sidebar.markdown("---")
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 4)
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        # Check columns
        req_cols = ['ORDERDATE', 'ORDERNUMBER', 'SALES', 'CUSTOMERNAME']
        if not all(col in df.columns for col in req_cols):
            st.error(f"Missing columns. Required: {req_cols}")
            return

        # Process Data
        rfm = calculate_rfm(df)
        
        # Clustering
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        rfm['Cluster'] = rfm['Cluster'].astype(str) # Convert to string for categorical coloring

        # --- Dashboard Layout ---
        st.title("👥 Customer Persona Engine")
        st.markdown("Analyze customer behavior using **RFM Clustering** to identify unique personas.")

        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Customers", len(rfm))
        kpi2.metric("Avg Recency", f"{rfm['Recency'].mean():.0f} days")
        kpi3.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f} orders")
        kpi4.metric("Avg Lifetime Value", f"${rfm['Monetary'].mean():,.0f}")

        st.markdown("---")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 3D Cluster View", "🕸️ Persona Radar", "📋 Data Details"])

        with tab1:
            st.subheader("Interactive 3D Customer Segments")
            st.info("Rotate, zoom, and hover over the plot to explore customer segments.")
            
            fig_3d = px.scatter_3d(
                rfm, 
                x='Recency', 
                y='Frequency', 
                z='Monetary',
                color='Cluster',
                hover_name='CustomerName',
                opacity=0.7,
                size_max=18,
                title="<b>Recency vs Frequency vs Monetary</b>",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=600)
            st.plotly_chart(fig_3d, use_container_width=True)

        with tab2:
            st.subheader("Persona Definitions (Cluster DNA)")
            
            # Calculate means
            cluster_centers = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            
            # Normalize for Radar Chart
            normalized_centers = normalize_centroids(cluster_centers)
            normalized_centers['Cluster'] = cluster_centers.index
            
            # Radar Chart Logic
            categories = ['Recency', 'Frequency', 'Monetary']
            fig_radar = go.Figure()

            for i, row in normalized_centers.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['Recency'], row['Frequency'], row['Monetary']],
                    theta=categories,
                    fill='toself',
                    name=f"Cluster {row['Cluster']}"
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Normalized Cluster Characteristics (0=Low, 1=High)",
                height=500
            )
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(fig_radar, use_container_width=True)
            with c2:
                st.markdown("### Cluster Summaries")
                st.dataframe(cluster_centers.style.format("{:,.0f}"))

        with tab3:
            st.subheader("Detailed Customer List")
            
            # Filter option
            selected_cluster = st.selectbox("Filter by Cluster", ["All"] + list(rfm['Cluster'].unique()))
            
            if selected_cluster != "All":
                filtered_df = rfm[rfm['Cluster'] == selected_cluster]
            else:
                filtered_df = rfm
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download
            csv = rfm.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", data=csv, file_name='customer_segmentation.csv', mime='text/csv')

    else:
        # Welcome Screen
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>👋 Welcome!</h2>
            <p>Please upload your <b>sales_data_sample.csv</b> file in the sidebar to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    