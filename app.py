import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar

# Configuration de la page
st.set_page_config(page_title="Data Science Learning Dashboard", layout="wide")

# Chargement du CSS personnalisé
st.markdown("""
    <style>
    .learning-section {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #1f1f1f;
        border: 1px solid #c0d8ea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .learning-section h3 {
        color: #2E86C1;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .learning-section ul {
        color: #2c3e50;
        margin-left: 20px;
    }
    .learning-section li {
        margin: 8px 0;
        line-height: 1.5;
    }
    .code-explanation {
        background-color: #f5f5f5;
        padding: 15px;
        border-left: 3px solid #2E86C1;
        margin: 15px 0;
        color: #333;
    }
    .code-explanation code {
        background-color: #e8e8e8;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

def train_and_evaluate_model(X, y, model_type='linear'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'R2 Score': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    feature_importance = None
    if model_type == 'random_forest':
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    return metrics, feature_importance

def perform_clustering(df, n_clusters=3):
    features = ['Ventes', 'Coûts', 'Satisfaction_Client']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )
    return df, cluster_centers

def main():
    st.title("🎓 Dashboard d'Apprentissage Data Science")

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        learning_mode = st.checkbox("Mode Apprentissage", value=True)
        data_option = st.radio(
            "Source des données",
            ["Données d'exemple", "Mes données", "Générer des données personnalisées"]
        )

        if data_option == "Mes données":
            uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
            else:
                st.info("En attente du fichier CSV...")
                return
        elif data_option == "Générer des données personnalisées":
            n_samples = st.slider("Nombre d'échantillons", 100, 1000, 365)
            noise_level = st.slider("Niveau de bruit", 0.0, 1.0, 0.2)
            seasonality = st.checkbox("Ajouter de la saisonnalité", value=True)
            trend = st.checkbox("Ajouter une tendance", value=True)

            dates = pd.date_range(start='2023-01-01', periods=n_samples)
            x = np.linspace(0, n_samples, n_samples)
            y = np.zeros(n_samples)
            if trend:
                y += x * 0.1
            if seasonality:
                y += np.sin(np.linspace(0, 4*np.pi, n_samples)) * 100
            y += np.random.normal(0, noise_level * np.std(y), n_samples)

            df = pd.DataFrame({
                'Date': dates,
                'Ventes': y + 1000,
                'Coûts': y * 0.7 + 800,
                'Satisfaction_Client': np.random.normal(4.2, 0.5, n_samples).clip(1, 5)
            })
        else:
            df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=365),
                'Ventes': np.random.normal(1000, 100, 365),
                'Coûts': np.random.normal(800, 50, 365),
                'Satisfaction_Client': np.random.normal(4.2, 0.5, 365).clip(1, 5)
            })

    # Tabs for EDA, ML, Time Series, Documentation
    tabs = st.tabs(["📊 EDA", "🤖 Machine Learning", "📈 Séries Temporelles", "📚 Documentation"])

    with tabs[0]:  # EDA
        st.header("Analyse Exploratoire des Données")
        if learning_mode:
            st.markdown("""
            <div class="learning-section">
                <h3>🎯 Objectifs d'apprentissage - EDA</h3>
                <ul>
                    <li>Comprendre la distribution des variables</li>
                    <li>Détecter les anomalies et valeurs aberrantes</li>
                    <li>Analyser les corrélations entre variables</li>
                    <li>Visualiser les tendances temporelles</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution des variables")
            variable = st.selectbox("Choisir une variable", ['Ventes', 'Coûts', 'Satisfaction_Client'])
            fig = go.Figure(data=[go.Histogram(x=df[variable], nbinsx=30), go.Box(x=df[variable], name='Box Plot')])
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Analyse des corrélations")
            correlation = df[['Ventes', 'Coûts', 'Satisfaction_Client']].corr()
            fig = px.imshow(correlation, title='Matrice de corrélation', color_continuous_scale='RdBu')
            st.plotly_chart(fig)

    with tabs[1]:  # Machine Learning
        st.header("Expérimentation Machine Learning")
        if learning_mode:
            st.markdown("""
            <div class="learning-section">
                <h3>🤖 Apprentissage Machine Learning</h3>
                <ul>
                    <li>Comparaison de différents modèles</li>
                    <li>Analyse des métriques de performance</li>
                    <li>Importance des features</li>
                    <li>Clustering et segmentation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        features = ['Coûts', 'Satisfaction_Client', 'DayOfWeek', 'Month']
        X, y = df[features], df['Ventes']

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Choisir un modèle", ['linear', 'random_forest'])
            metrics, feature_importance = train_and_evaluate_model(X, y, model_type)
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.4f}")
            if feature_importance is not None:
                st.bar_chart(feature_importance.set_index('feature')['importance'])
        
        with col2:
            n_clusters = st.slider("Nombre de clusters", 2, 5, 3)
            clustered_df, centers = perform_clustering(df, n_clusters)
            fig = px.scatter_3d(clustered_df, x='Ventes', y='Coûts', z='Satisfaction_Client', color='Cluster')
            st.plotly_chart(fig)

    with tabs[2]:  # Séries Temporelles
        st.header("Analyse de Séries Temporelles")
        if learning_mode:
            st.markdown("""
            <div class="learning-section">
                <h3>📈 Analyse temporelle</h3>
                <ul>
                    <li>Décomposition de série temporelle</li>
                    <li>Analyse de saisonnalité</li>
                    <li>Prédiction de tendances</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        decomposition = seasonal_decompose(df.set_index('Date')['Ventes'], period=30, extrapolate_trend='freq')
        fig = go.Figure([
            go.Scatter(x=df['Date'], y=decomposition.trend, name='Tendance'),
            go.Scatter(x=df['Date'], y=decomposition.seasonal, name='Saisonnalité'),
            go.Scatter(x=df['Date'], y=decomposition.resid, name='Résidus')
        ])
        st.plotly_chart(fig)

    with tabs[3]:  # Documentation
        st.header("Documentation et Ressources")
        st.markdown("""
        ### 📚 Ressources d'apprentissage
        - **Pandas**: Manipulation et analyse de données
        - **Scikit-learn**: Modèles de Machine Learning
        - **Plotly**: Visualisation interactive
        - **Statsmodels**: Analyse de séries temporelles
        """)

if __name__ == "__main__":
    main()