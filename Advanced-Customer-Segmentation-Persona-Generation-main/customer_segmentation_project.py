import os
import json
import math
import sys
import re
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import streamlit as st
except Exception:
    st = None

try:
    import umap
except Exception:
    umap = None

try:
    import openai
except Exception:
    openai = None


# -----------------------------
# 1) Dataset creation / loader
# -----------------------------

def generate_synthetic_data_single_file(n_customers=2000, seed=42):
    """Generate a single merged dataset (Transactions + Customer Info)."""
    np.random.seed(seed)
    customer_ids = [f'C{100000+i}' for i in range(n_customers)]
    ages = np.random.randint(18, 70, size=n_customers)
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_customers, p=[0.48, 0.48, 0.04])
    incomes = np.random.normal(loc=50000, scale=30000, size=n_customers).clip(8000, 250000)
    cities = np.random.choice(['Mumbai','Delhi','Bengaluru','Chennai','Hyderabad','Kolkata','Pune','Jaipur'], size=n_customers)

    # Customer lookup
    cust_dict = {
        cid: {'age': a, 'gender': g, 'income': int(inc), 'city': c}
        for cid, a, g, inc, c in zip(customer_ids, ages, genders, incomes, cities)
    }

    categories = ['Electronics','Fashion','Grocery','Home','Sports','Books','Beauty']
    rows = []
    tx_id = 1
    today = datetime.now()
    
    for cid in customer_ids:
        num_tx = np.random.poisson(6) + 1 
        cust_info = cust_dict[cid]
        
        for _ in range(max(1, num_tx)):
            days_ago = np.random.exponential(scale=200)
            purchased_at = today - timedelta(days=int(days_ago))
            category = np.random.choice(categories, p=[0.12,0.2,0.25,0.15,0.08,0.1,0.1])
            amount = float(max(5, np.random.normal(100, 80))) * (1 + (categories.index(category)/7))
            quantity = np.random.randint(1,4)
            
            # Row structure: TransID, CustID, Date, Amount, Category, Qty, Age, Gender, Income, City
            rows.append([
                f'TX{tx_id}', cid, purchased_at.strftime('%Y-%m-%d'), round(amount,2), category, quantity,
                cust_info['age'], cust_info['gender'], cust_info['income'], cust_info['city']
            ])
            tx_id += 1

    columns = [
        'transaction_id', 'customer_id', 'purchased_at', 'amount', 'category', 'quantity',
        'age', 'gender', 'income', 'city'
    ]
    df = pd.DataFrame(rows, columns=columns)
    df['purchased_at'] = pd.to_datetime(df['purchased_at'])
    return df


def normalize_col_name(name):
    """Normalize column name: lowercase, remove spaces/underscores/special chars."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def process_single_file(df):
    """
    Splits the single DataFrame into customers_df and tx_df.
    Robustly maps column names.
    """
    # Create a mapping of normalized name -> original name
    norm_map = {normalize_col_name(c): c for c in df.columns}
    
    # Define aliases for required columns
    # Key = internal name, Value = list of possible external names (normalized)
    aliases = {
        'transaction_id': ['ordernumber', 'orderid', 'transactionid', 'transid', 'invoiceno', 'invoicenumber', 'invoice'],
        'customer_id': ['customername', 'customerid', 'clientid', 'clientname', 'id', 'email', 'customer'],
        'purchased_at': ['orderdate', 'date', 'purchasedat', 'transactiondate', 'timestamp', 'invoicedate'],
        'amount': ['sales', 'amount', 'total', 'revenue', 'totalprice', 'price'],
        'category': ['productline', 'category', 'productcategory', 'department', 'type'],
        'quantity': ['quantityordered', 'quantity', 'qty', 'count'],
        'age': ['age', 'customerage'],
        'gender': ['gender', 'sex'],
        'income': ['income', 'annualincome', 'salary'],
        'city': ['city', 'location']
    }
    
    # Identify renaming dictionary
    rename_dict = {}
    for internal, possible_names in aliases.items():
        for alias in possible_names:
            if alias in norm_map:
                rename_dict[norm_map[alias]] = internal
                break
    
    # Apply renaming
    df = df.rename(columns=rename_dict)
    
    # Check for essential columns and fill/warn
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = df.index
        
    if 'amount' not in df.columns:
        if 'quantity' in df.columns and 'priceeach' in [c.lower() for c in df.columns]:
            # Try to find price column specifically
            price_col = next((c for c in df.columns if 'price' in c.lower()), None)
            if price_col:
                df['amount'] = df['quantity'] * df[price_col]
        
    required = ['customer_id', 'purchased_at', 'amount']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # If still missing, try more generous fallback (e.g. use first string col as customer_id?)
        # For now, just raise error but with clearer message
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}. Please check your CSV headers.")

    # 2. Extract Transactions
    df['purchased_at'] = pd.to_datetime(df['purchased_at'], errors='coerce')
    
    # Filter for transaction columns that exist
    tx_cols = ['transaction_id', 'customer_id', 'purchased_at', 'amount']
    if 'category' in df.columns: tx_cols.append('category')
    if 'quantity' in df.columns: tx_cols.append('quantity')
    
    tx_df = df[tx_cols].copy()
    if 'category' not in tx_df.columns:
        tx_df['category'] = 'General' # Fallback category

    # 3. Extract Customers
    unique_customers = df['customer_id'].unique()
    n_cust = len(unique_customers)
    
    # Prepare customer dataframe
    # We aggregate to get the 'first' value of demographic columns for each customer
    grp = df.groupby('customer_id')
    
    cust_data = {'customer_id': unique_customers}
    
    # Handle Age
    if 'age' in df.columns:
        cust_data['age'] = grp['age'].first().values
    else:
        cust_data['age'] = np.random.randint(18, 70, size=n_cust)
        
    # Handle Gender
    if 'gender' in df.columns:
        cust_data['gender'] = grp['gender'].first().values
    else:
        cust_data['gender'] = np.random.choice(['Male', 'Female'], size=n_cust)
        
    # Handle Income
    if 'income' in df.columns:
        cust_data['income'] = grp['income'].first().values
    else:
        cust_data['income'] = np.random.normal(50000, 15000, size=n_cust).astype(int)
    
    # Handle City
    if 'city' in df.columns:
        cust_data['city'] = grp['city'].first().values
    
    customers_df = pd.DataFrame(cust_data)
    
    return customers_df, tx_df


# -----------------------------
# 2) Preprocessing & Features
# -----------------------------

def compute_rfm(customers_df, tx_df, snapshot_date=None):
    """Compute RFM features per customer."""
    if snapshot_date is None:
        snapshot_date = tx_df['purchased_at'].max() + pd.Timedelta(days=1)
    
    snapshot_date = pd.to_datetime(snapshot_date)

    agg = tx_df.groupby('customer_id').agg({
        'purchased_at': lambda x: (snapshot_date - x.max()).days,
        'transaction_id': 'count',
        'amount': 'sum'
    }).rename(columns={'purchased_at':'recency','transaction_id':'frequency','amount':'monetary'})

    # JOIN NOTE: This left join preserves all columns in customers_df (age, income, etc.)
    df = customers_df.set_index('customer_id').join(agg, how='left').fillna({'recency':999,'frequency':0,'monetary':0}).reset_index()
    df['monetary'] = df['monetary'].astype(float)
    return df


def add_behavioral_features(df, tx_df):
    """Add simple behavioral features: avg_order_value, avg_days_between, category_affinity."""
    df = df.copy()
    orders = tx_df.groupby('customer_id').agg({'amount':'mean'}).rename(columns={'amount':'aov'})
    df = df.set_index('customer_id').join(orders['aov']).reset_index()
    df['aov'] = df['aov'].fillna(0)

    # Avg days between
    tx_sorted = tx_df.sort_values(['customer_id', 'purchased_at'])
    tx_sorted['prev_date'] = tx_sorted.groupby('customer_id')['purchased_at'].shift(1)
    tx_sorted['days_diff'] = (tx_sorted['purchased_at'] - tx_sorted['prev_date']).dt.days
    
    avg_days = tx_sorted.groupby('customer_id')['days_diff'].mean().reset_index().rename(columns={'days_diff':'avg_days_between'})
    df = df.set_index('customer_id').join(avg_days.set_index('customer_id')).reset_index()
    df['avg_days_between'] = df['avg_days_between'].fillna(999)

    # Category Affinity
    if 'category' in tx_df.columns:
        cat_pref = tx_df.groupby(['customer_id','category']).size().unstack(fill_value=0)
        if not cat_pref.empty:
            cat_pref = cat_pref.div(cat_pref.sum(axis=1), axis=0).add_prefix('cat_')
            df = df.set_index('customer_id').join(cat_pref).reset_index().fillna(0)
    
    # Fallback if no category columns added
    if not any(c.startswith('cat_') for c in df.columns):
        df['cat_general'] = 0

    df['engagement_score'] = (1 / (1 + df['avg_days_between'])) * (1 + np.log1p(df['frequency']))
    return df


# -----------------------------
# 3) Modeling & Clustering
# -----------------------------

def prepare_features_for_clustering(df, feature_columns, scaler=None):
    # Ensure all selected features exist, fill missing with 0
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0
            
    X = df[feature_columns].fillna(0).copy()
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def run_kmeans(X, n_clusters=4, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def run_gmm(X, n_components=4, random_state=42):
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels


# -----------------------------
# 4) Evaluation & Persona Construction
# -----------------------------

def evaluate_clusters(X, labels):
    result = {}
    unique_labels = set(labels)
    if len(unique_labels) > 1 and (len(unique_labels - {-1}) > 0):
        try:
            result['silhouette'] = silhouette_score(X, labels)
        except Exception:
            result['silhouette'] = float('nan')
        try:
            result['davies_bouldin'] = davies_bouldin_score(X, labels)
        except Exception:
            result['davies_bouldin'] = float('nan')
    else:
        result['silhouette'] = float('nan')
        result['davies_bouldin'] = float('nan')
    result['n_clusters'] = len(unique_labels - {-1})
    return result


def summarize_cluster(df, labels, label):
    cluster_df = df.loc[labels == label]
    if len(cluster_df) == 0:
        return {}
        
    summary = {
        'cluster_label': int(label),
        'size': int(len(cluster_df)),
        'avg_recency': float(cluster_df['recency'].mean()),
        'avg_frequency': float(cluster_df['frequency'].mean()),
        'avg_monetary': float(cluster_df['monetary'].mean()),
        'avg_aov': float(cluster_df['aov'].mean()),
        'engagement_score': float(cluster_df['engagement_score'].mean())
    }
    
    # Demographics if available
    if 'age' in cluster_df.columns:
        summary['avg_age'] = float(cluster_df['age'].mean())
    if 'income' in cluster_df.columns:
        summary['avg_income'] = float(cluster_df['income'].mean())
    if 'gender' in cluster_df.columns:
        summary['gender_ratio'] = dict(cluster_df['gender'].value_counts(normalize=True).round(2))

    # Top categories
    cat_cols = [c for c in cluster_df.columns if c.startswith('cat_')]
    if cat_cols:
        summary['top_categories'] = cluster_df[cat_cols].mean().sort_values(ascending=False).head(3).to_dict()
    else:
        summary['top_categories'] = {}
        
    return summary


def generate_persona_from_summary(summary):
    if not summary:
        return ""
    name_candidates = [
        'Value-Seeker', 'Premium Enthusiast', 'Occasional Shopper', 'Loyal Regular', 'Bargain Hunter', 'Trend Follower'
    ]
    name = np.random.choice(name_candidates)
    persona = f"Persona: {name} (Cluster {summary['cluster_label']})\n"
    persona += f"Size: {summary['size']} customers\n"
    
    if 'avg_age' in summary:
        persona += f"Age ~ {int(summary['avg_age'])}"
    if 'avg_income' in summary:
        persona += f", Income ~ {int(summary['avg_income']):,}"
    persona += "\n"
        
    persona += f"Typical behavior: Avg order value {summary['avg_aov']:.2f}, Avg frequency {summary['avg_frequency']:.1f} purchases\n"
    
    top_cats = summary.get('top_categories', {})
    if top_cats:
        persona += "Top categories: " + ", ".join([k.replace('cat_','') for k in top_cats.keys()]) + "\n"
    
    # Engagement Logic
    recency = summary.get('avg_recency', 999)
    eng_score = summary.get('engagement_score', 0)
    
    if eng_score > 0.02 and recency < 90:
        persona += "Engagement: Highly engaged — target with loyalty programs.\n"
    elif recency > 180:
        persona += "Engagement: Dormant — target with reactivation campaigns.\n"
    else:
        persona += "Engagement: Medium — run personalized offers.\n"
    
    return persona


def call_openai_generate_persona(summary, max_tokens=180):
    if openai is None:
        return generate_persona_from_summary(summary)
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return generate_persona_from_summary(summary)
    
    openai.api_key = api_key
    prompt = f"Create a marketing persona description in ~120 words from this cluster summary:\n{json.dumps(summary, indent=2)}"
    try:
        resp = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens
        )
        text = resp['choices'][0]['message']['content'].strip()
        return text
    except Exception as e:
        print('OpenAI API failed:', e)
        return generate_persona_from_summary(summary)


# -----------------------------
# 5) Streamlit App
# -----------------------------

APP_MARKDOWN = """
# Customer Segmentation & Persona Generator

Upload a single CSV file containing transaction data. 
The app effectively separates customer and transaction information, filling in missing demographics with synthetic data if necessary.
"""

def streamlit_app():
    if st is None:
        raise RuntimeError('Streamlit is required. Install with: pip install streamlit')

    st.set_page_config(layout='wide', page_title='Customer Segmentation')
    st.markdown(APP_MARKDOWN)

    # Single File Upload
    uploaded_file = st.file_uploader('Upload Sales Data CSV', type=['csv'])
    use_synth = st.button('Generate synthetic dataset')

    customers_df = None
    tx_df = None
    
    if uploaded_file:
        try:
            # Try reading with different encodings
            try:
                raw_df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                raw_df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                
            customers_df, tx_df = process_single_file(raw_df)
            st.success(f"File processed! Found {len(customers_df)} customers and {len(tx_df)} transactions.")
            
            if 'age' not in raw_df.columns and 'age' not in tx_df.columns:
                st.info("Note: Demographic data (Age, Income) was not found in the file and has been synthesized for demonstration.")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return
            
    elif use_synth:
        with st.spinner('Generating synthetic data...'):
            raw_df = generate_synthetic_data_single_file(n_customers=2000)
            customers_df, tx_df = process_single_file(raw_df)
            st.success('Synthetic data generated')
    else:
        st.info('Please upload a file or generate synthetic data to start.')
        return

    if customers_df is None or tx_df is None:
        return

    # Show sample
    with st.expander('Data Preview'):
        st.subheader("Extracted Customer Profiles")
        st.dataframe(customers_df.head())
        st.subheader("Transaction Records")
        st.dataframe(tx_df.head())

    # Snapshot date
    max_tx = tx_df['purchased_at'].max()
    snapshot_date = st.date_input('Snapshot date for recency', value=max_tx.date())

    # Compute RFM
    # NOTE: This function joins customers_df (with age, income) + aggregated tx info
    # So 'df' already contains age and income!
    df = compute_rfm(customers_df, tx_df, snapshot_date=snapshot_date)
    df = add_behavioral_features(df, tx_df)

    # Feature Selection
    cat_cols = [c for c in df.columns if c.startswith('cat_')]
    default_features = ['recency','frequency','monetary','aov','engagement_score'] + cat_cols
    
    # Add demographics if they exist in the df (they are preserved from customers_df)
    if 'age' in df.columns:
        default_features.append('age')
    if 'income' in df.columns:
        default_features.append('income')

    features = st.multiselect('Select features for clustering', options=df.columns.tolist(), default=default_features)

    if not features:
        st.warning('Select at least one feature')
        return

    X_scaled, scaler = prepare_features_for_clustering(df, features)

    # Dimensionality Reduction
    dr_method = st.selectbox('Dimensionality reduction', options=['PCA','t-SNE','UMAP'], index=0)
    X_2d = None
    
    with st.spinner(f'Running {dr_method}...'):
        if dr_method == 'PCA':
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
        elif dr_method == 't-SNE':
            X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
        elif dr_method == 'UMAP':
            if umap:
                X_2d = umap.UMAP(n_components=2).fit_transform(X_scaled)
            else:
                st.warning("UMAP not installed, falling back to PCA")
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_scaled)

    # Clustering
    algo = st.selectbox('Clustering algorithm', options=['KMeans','GMM','DBSCAN'])
    
    if algo == 'KMeans':
        n_clusters = st.slider('Number of Clusters', 2, 10, 4)
        model, labels = run_kmeans(X_scaled, n_clusters=n_clusters)
    elif algo == 'GMM':
        n_clusters = st.slider('Number of Components', 2, 10, 4)
        model, labels = run_gmm(X_scaled, n_components=n_clusters)
    else:
        eps = st.number_input('EPS (DBSCAN)', 0.1, 10.0, 0.5, 0.1)
        min_samples = st.slider('Min Samples', 1, 20, 5)
        model, labels = run_dbscan(X_scaled, eps=eps, min_samples=min_samples)

    df['cluster'] = labels

    # Metrics
    eval_metrics = evaluate_clusters(X_scaled, labels)
    c1, c2, c3 = st.columns(3)
    c1.metric('Silhouette Score', f"{eval_metrics.get('silhouette', 0):.3f}")
    c2.metric('Davies-Bouldin', f"{eval_metrics.get('davies_bouldin', 0):.3f}")
    c3.metric('Clusters Found', eval_metrics.get('n_clusters', 0))

    # Plot
    if X_2d is not None:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=labels, palette='viridis', s=20, ax=ax)
        ax.set_title(f'Clusters ({dr_method})')
        st.pyplot(fig)

    # Summaries
    st.header('Cluster Personas')
    personas = []
    unique_labels = sorted(list(set(labels)))
    for lbl in unique_labels:
        if lbl == -1: continue
        summary = summarize_cluster(df, labels, lbl)
        personas.append((lbl, summary))

    cols = st.columns(2)
    for i, (lbl, summary) in enumerate(personas):
        with cols[i % 2]:
            st.subheader(f'Cluster {lbl} (n={summary.get("size")})')
            txt = generate_persona_from_summary(summary)
            st.text_area(f"Persona {lbl}", txt, height=150)
            
            # Optional OpenAI integration
            if st.checkbox(f'Enhance with AI (Cluster {lbl})', key=f'ai_{lbl}'):
                ai_txt = call_openai_generate_persona(summary)
                st.info(ai_txt)

    if st.button('Download Results JSON'):
        export = {lbl: s for lbl, s in personas}
        st.download_button('Download JSON', json.dumps(export, indent=2), file_name='personas.json')


# -----------------------------
# 6) CLI / Entry Point
# -----------------------------

def demo_run_and_save(output_prefix='output_demo'):
    # Generate single file style data
    raw_df = generate_synthetic_data_single_file(2000)
    customers, transactions = process_single_file(raw_df)
    
    df = compute_rfm(customers, transactions)
    df = add_behavioral_features(df, transactions)
    
    # Feature selection
    cat_cols = [c for c in df.columns if c.startswith('cat_')]
    features = ['recency','frequency','monetary','aov','engagement_score'] + cat_cols
    if 'age' in df.columns: features.append('age')
    if 'income' in df.columns: features.append('income')
    
    X_scaled, scaler = prepare_features_for_clustering(df, features)
    model, labels = run_kmeans(X_scaled, n_clusters=4)
    df['cluster'] = labels
    
    summaries = [summarize_cluster(df, labels, lbl) for lbl in sorted(set(labels)) if lbl!=-1]
    personas = {s['cluster_label']: generate_persona_from_summary(s) for s in summaries}
    
    os.makedirs(output_prefix, exist_ok=True)
    df.to_csv(os.path.join(output_prefix, 'clustered_customers.csv'), index=False)
    with open(os.path.join(output_prefix, 'personas.json'), 'w') as f:
        json.dump(personas, f, indent=2)
    print('Demo run saved to', output_prefix)


if __name__ == '__main__':
    if len(sys.argv) > 1 and '--demo' in sys.argv:
        demo_run_and_save()
    else:
        if st is not None:
            try:
                streamlit_app()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            print("Run with: streamlit run script.py")