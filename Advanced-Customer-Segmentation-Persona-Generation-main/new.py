import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Page Config ---
st.set_page_config(
    page_title="ML-II Clustering Workbench",
    page_icon="🤖",
    layout="wide"
)

# --- Course Context (Unit I - III) ---
# This app implements K-Means (Unit I), PCA (Unit II), and Evaluation Metrics (Unit III)

@st.cache_data
def load_data(file):
    return pd.read_csv(file, encoding='ISO-8859-1')

def calculate_elbow(data, max_k=10):
    """Implementation of the Elbow Method (CO1/Unit I)"""
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def main():
    st.title("🧪 Machine Learning-II: Clustering & PCA Lab")
    st.markdown("""
    This workbench demonstrates core concepts from **INT423**:
    * **Unit I:** K-Means Clustering & Elbow Method [cite: 11]
    * **Unit II:** PCA for Feature Selection & Dimensionality Reduction 
    * **Unit III:** Clustering Evaluation Metrics (Silhouette Score) [cite: 15]
    """)

    # Sidebar - Configuration
    st.sidebar.header("Step 1: Data Setup")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.sidebar.success("Data Loaded!")
        
        # Feature Selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.sidebar.multiselect(
            "Select Features for Clustering", 
            options=numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )

        if len(selected_features) < 2:
            st.warning("Please select at least 2 numerical features.")
            return

        # Preprocessing
        data_subset = df[selected_features].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)

        # PCA - Unit II
        st.sidebar.markdown("---")
        st.sidebar.header("Step 2: PCA (Unit II)")
        use_pca = st.sidebar.checkbox("Apply PCA for Dimensionality Reduction")
        
        if use_pca:
            pca = PCA(n_components=2)
            processed_data = pca.fit_transform(scaled_data)
            st.sidebar.info(f"Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            processed_data = scaled_data

        # Elbow Method - Unit I
        st.sidebar.markdown("---")
        st.sidebar.header("Step 3: Optimization")
        if st.sidebar.button("Run Elbow Method"):
            st.subheader("Choosing Optimal K: The Elbow Method [cite: 11]")
            wcss = calculate_elbow(processed_data)
            fig_elbow = px.line(x=range(1, 11), y=wcss, markers=True, 
                                labels={'x': 'Number of Clusters (K)', 'y': 'WCSS'},
                                title="Elbow Plot to determine Optimal K")
            st.plotly_chart(fig_elbow)

        n_clusters = st.sidebar.slider("Select K (Number of Clusters)", 2, 8, 3)

        # Clustering Execution - Unit I
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(processed_data)
        data_subset['Cluster'] = clusters.astype(str)

        # Metrics - Unit III
        st.sidebar.markdown("---")
        st.sidebar.header("Step 4: Evaluation")
        sil_score = silhouette_score(processed_data, clusters)
        st.sidebar.metric("Silhouette Score", f"{sil_score:.3f}")

        # --- Visualizations ---
        tab1, tab2 = st.tabs(["📊 Cluster Visualization", "📋 Analyzed Data"])

        with tab1:
            st.subheader(f"K-Means Clustering Results (K={n_clusters})")
            
            # If PCA used, plot the 2 components
            if use_pca:
                plot_df = pd.DataFrame(processed_data, columns=['PC1', 'PC2'])
                plot_df['Cluster'] = clusters.astype(str)
                fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                                 title="Clusters in PCA-Reduced Space")
            else:
                # 3D Plot using first 3 selected features
                fig = px.scatter_3d(data_subset, 
                                   x=selected_features[0], 
                                   y=selected_features[1], 
                                   z=selected_features[2] if len(selected_features) > 2 else selected_features[0],
                                   color='Cluster',
                                   title="Cluster Distribution")
            
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Processed Cluster Data")
            st.dataframe(data_subset)
            
            # Download
            csv = data_subset.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", data=csv, file_name='ml_clustering_results.csv')

    else:
        st.info("Please upload a CSV file to begin the Machine Learning-II clustering workflow.")

if __name__ == "__main__":
    main()