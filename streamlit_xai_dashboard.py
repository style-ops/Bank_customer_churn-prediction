"""
Streamlit XAI Dashboard for Bank Churn Prediction
Features: SHAP, LIME, and NLP Explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# XAI Libraries
import shap
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer

# NLP Libraries
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Download NLTK data (with error handling)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Page Configuration
st.set_page_config(
    page_title="XAI Dashboard - Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try loading from data_splits directory first
        if os.path.exists("data_splits/train_set.csv"):
            train_df = pd.read_csv("data_splits/train_set.csv")
            test_df = pd.read_csv("data_splits/test_set.csv")
            return train_df, test_df, None
        else:
            # Try loading original dataset
            csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
            if csv_files:
                df = pd.read_csv(csv_files[0])
                return df, None, None
            else:
                st.error("No CSV files found. Please upload a dataset.")
                return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def load_or_train_model(train_df, target_col):
    """Load saved model or train a new one"""
    model_path = "models/xgb_model.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            return model, preprocessor, "Loaded saved model"
        except:
            pass
    
    # Train new model
    return train_model(train_df, target_col)

def train_model(train_df, target_col):
    """Train XGBoost model with preprocessing"""
    from sklearn.model_selection import train_test_split
    
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Ensure binary target
    y = (y != 0).astype(int)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Remove customer_id if present
    if 'customer_id' in numeric_features:
        numeric_features.remove('customer_id')
    
    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Transform data
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_val_prepared = preprocessor.transform(X_val)
    
    # Handle infinite values
    X_train_prepared = pd.DataFrame(X_train_prepared).replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_val_prepared = pd.DataFrame(X_val_prepared).replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1]) if 1 in y_train.value_counts() else 1
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    model.fit(X_train_prepared, y_train)
    
    # Save model and preprocessor
    os.makedirs("models", exist_ok=True)
    with open("models/xgb_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open("models/preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    
    return model, preprocessor, "Trained new model"

def prepare_data_for_explanation(df, target_col, preprocessor):
    """Prepare data for SHAP and LIME explanations"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    y = (y != 0).astype(int)
    
    X_prepared = preprocessor.transform(X)
    X_prepared = pd.DataFrame(X_prepared).replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Get feature names
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if 'customer_id' in numeric_features:
        numeric_features.remove('customer_id')
    
    if categorical_features:
        encoded_cat_cols = list(preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features))
        feature_names = numeric_features + encoded_cat_cols
    else:
        feature_names = numeric_features
    
    return X_prepared, y, X, feature_names

def main():
    st.markdown('<h1 class="main-header">🏦 XAI Dashboard: Bank Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Overview", "🔍 SHAP Explanations", "🍋 LIME Explanations", "📝 NLP Analysis", "🔮 Single Prediction"]
    )
    
    # Load data
    train_df, test_df, raw_df = load_data()
    
    if train_df is None:
        st.error("Please ensure your data files are in the correct location.")
        return
    
    # Determine target column
    target_col = "aml_flag"
    
    # Check if target column exists
    if target_col not in train_df.columns:
        # Try to find a binary target column
        binary_cols = [col for col in train_df.columns if train_df[col].nunique() == 2]
        if binary_cols:
            target_col = binary_cols[0]
            st.sidebar.warning(f"Using '{target_col}' as target column instead of 'aml_flag'")
        else:
            st.error(f"Target column '{target_col}' not found in dataset. Available columns: {list(train_df.columns)}")
            return
    
    # Load or train model
    with st.spinner("Loading model..."):
        try:
            model, preprocessor, model_status = load_or_train_model(train_df, target_col)
        except Exception as e:
            st.error(f"Error loading/training model: {str(e)}")
            st.info("Please ensure your data is properly formatted and the target column exists.")
            return
    
    st.sidebar.success(f"Model: {model_status}")
    
    # Page routing
    if page == "📊 Overview":
        show_overview(train_df, test_df, model, preprocessor, target_col)
    elif page == "🔍 SHAP Explanations":
        show_shap_explanations(train_df, test_df, model, preprocessor, target_col)
    elif page == "🍋 LIME Explanations":
        show_lime_explanations(train_df, test_df, model, preprocessor, target_col)
    elif page == "📝 NLP Analysis":
        show_nlp_analysis(train_df, test_df)
    elif page == "🔮 Single Prediction":
        show_single_prediction(train_df, model, preprocessor, target_col)

def show_overview(train_df, test_df, model, preprocessor, target_col):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", len(train_df))
    with col2:
        st.metric("Test Samples", len(test_df) if test_df is not None else "N/A")
    with col3:
        st.metric("Features", len(train_df.columns) - 1)
    with col4:
        churn_rate = (train_df[target_col] != 0).mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    st.subheader("Target Distribution")
    target_dist = (train_df[target_col] != 0).value_counts()
    
    fig = px.pie(
        values=target_dist.values,
        names=["No Churn", "Churn"],
        title="Class Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Data Preview")
    st.dataframe(train_df.head(10), use_container_width=True)
    
    # Model Performance
    if test_df is not None:
        st.subheader("Model Performance")
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        y_test = (y_test != 0).astype(int)
        
        X_test_prepared = preprocessor.transform(X_test)
        X_test_prepared = pd.DataFrame(X_test_prepared).replace([np.inf, -np.inf], np.nan).fillna(0).values
        
        y_pred = model.predict(X_test_prepared)
        y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with col2:
            st.write("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_score:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title=f'ROC Curve (AUC = {auc_score:.3f})'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.write("**Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

def show_shap_explanations(train_df, test_df, model, preprocessor, target_col):
    st.header("🔍 SHAP Explanations")
    st.write("SHAP (SHapley Additive exPlanations) values explain the output of machine learning models.")
    
    if test_df is None:
        st.warning("Test set not found. Using sample from training set.")
        explanation_df = train_df.sample(min(100, len(train_df)))
    else:
        explanation_df = test_df.sample(min(100, len(test_df)))
    
    X_explanation, y_explanation, X_original, feature_names = prepare_data_for_explanation(
        explanation_df, target_col, preprocessor
    )
    
    explanation_type = st.radio(
        "Select SHAP Explanation Type",
        ["Summary Plot", "Waterfall Plot (Single Instance)", "Feature Importance", "Dependence Plot"]
    )
    
    with st.spinner("Computing SHAP values... This may take a moment."):
        # Use TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(model)
        
        # Limit instances for performance
        max_instances = min(50, len(X_explanation))
        X_shap = X_explanation[:max_instances]
        
        if explanation_type == "Summary Plot":
            shap_values = explainer.shap_values(X_shap)
            
            # Handle binary classification - get positive class values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class (churn) values
            
            st.subheader("SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(10, 8))
            feature_display = min(len(feature_names), shap_values.shape[1])
            shap.summary_plot(
                shap_values, 
                X_shap[:, :feature_display], 
                feature_names=feature_names[:feature_display], 
                show=False, 
                max_display=20
            )
            st.pyplot(fig)
            plt.close()
            
        elif explanation_type == "Waterfall Plot (Single Instance)":
            instance_idx = st.slider("Select instance index", 0, min(len(X_explanation) - 1, 20), 0)
            
            shap_values = explainer.shap_values(X_explanation[instance_idx:instance_idx+1])
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values_instance = shap_values[1][0]  # Positive class
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            else:
                shap_values_instance = shap_values[0]
                base_value = explainer.expected_value
            
            st.subheader(f"Waterfall Plot - Instance {instance_idx}")
            st.write(f"**Actual Label:** {'Churn' if y_explanation.iloc[instance_idx] == 1 else 'No Churn'}")
            
            # Create waterfall plot
            feature_display = min(len(feature_names), len(shap_values_instance))
            fig = plt.figure(figsize=(10, 8))
            shap.plots.waterfall(shap.Explanation(
                values=shap_values_instance[:feature_display],
                base_values=base_value,
                data=X_explanation[instance_idx, :feature_display],
                feature_names=feature_names[:feature_display]
            ), show=False, max_display=15)
            st.pyplot(fig)
            plt.close()
            
        elif explanation_type == "Feature Importance":
            shap_values = explainer.shap_values(X_shap)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class values
            
            st.subheader("SHAP Feature Importance")
            feature_display = min(len(feature_names), shap_values.shape[1])
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.bar(
                shap_values, 
                feature_names=feature_names[:feature_display], 
                show=False, 
                max_display=20
            )
            st.pyplot(fig)
            plt.close()
            
        elif explanation_type == "Dependence Plot":
            shap_values = explainer.shap_values(X_explanation[:100])
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class values
            
            # Select feature for dependence plot
            numeric_features = X_original.select_dtypes(include=[np.number]).columns.tolist()
            if 'customer_id' in numeric_features:
                numeric_features.remove('customer_id')
            
            if numeric_features:
                selected_feature = st.selectbox("Select feature for dependence plot", numeric_features[:10])
                
                if selected_feature in numeric_features:
                    feature_idx = numeric_features.index(selected_feature)
                    if feature_idx < shap_values.shape[1] and feature_idx < X_explanation.shape[1]:
                        st.subheader(f"SHAP Dependence Plot: {selected_feature}")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.scatter(shap.Explanation(
                            values=shap_values[:, feature_idx],
                            data=X_explanation[:100, feature_idx],
                            feature_names=[selected_feature]
                        ), show=False)
                        st.pyplot(fig)
                        plt.close()

def show_lime_explanations(train_df, test_df, model, preprocessor, target_col):
    st.header("🍋 LIME Explanations")
    st.write("LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions.")
    
    if test_df is None:
        explanation_df = train_df.sample(min(100, len(train_df)))
    else:
        explanation_df = test_df.sample(min(100, len(test_df)))
    
    X_explanation, y_explanation, X_original, feature_names = prepare_data_for_explanation(
        explanation_df, target_col, preprocessor
    )
    
    instance_idx = st.slider("Select instance to explain", 0, min(len(X_explanation) - 1, 50), 0)
    
    st.write(f"**Explaining instance {instance_idx}**")
    
    # Display instance details
    st.subheader("Instance Details")
    instance_data = X_original.iloc[instance_idx]
    
    # Show top features
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numeric Features:**")
        numeric_features = X_original.select_dtypes(include=[np.number]).columns.tolist()
        if 'customer_id' in numeric_features:
            numeric_features.remove('customer_id')
        st.dataframe(instance_data[numeric_features].head(10))
    
    with col2:
        st.write("**Categorical Features:**")
        categorical_features = X_original.select_dtypes(exclude=[np.number]).columns.tolist()
        st.dataframe(instance_data[categorical_features].head(10))
    
    st.write(f"**Actual Label:** {'Churn' if y_explanation.iloc[instance_idx] == 1 else 'No Churn'}")
    
    with st.spinner("Computing LIME explanation..."):
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_explanation,
            feature_names=feature_names[:X_explanation.shape[1]],
            class_names=['No Churn', 'Churn'],
            mode='classification'
        )
        
        # Get explanation
        explanation = explainer.explain_instance(
            X_explanation[instance_idx],
            model.predict_proba,
            num_features=15,
            top_labels=1
        )
        
        # Display explanation
        st.subheader("LIME Explanation")
        
        # Get feature importance
        exp_list = explanation.as_list()
        exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Importance'])
        exp_df = exp_df.sort_values('Importance', key=abs, ascending=False)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['green' if x > 0 else 'red' for x in exp_df['Importance']]
        ax.barh(range(len(exp_df)), exp_df['Importance'], color=colors)
        ax.set_yticks(range(len(exp_df)))
        ax.set_yticklabels(exp_df['Feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('LIME Feature Importance')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show prediction probabilities
        st.subheader("Prediction Probabilities")
        proba = model.predict_proba(X_explanation[instance_idx:instance_idx+1])[0]
        prob_df = pd.DataFrame({
            'Class': ['No Churn', 'Churn'],
            'Probability': proba
        })
        st.dataframe(prob_df, use_container_width=True)
        
        # Show top contributing features
        st.subheader("Top Contributing Features")
        st.dataframe(exp_df, use_container_width=True)

def show_nlp_analysis(train_df, test_df):
    st.header("📝 NLP Analysis")
    st.write("Natural Language Processing analysis on text columns in the dataset.")
    
    # Identify text columns
    text_columns = ['transaction_description', 'merchant_category', 'feedback_type', 'resolution_status']
    available_text_cols = [col for col in text_columns if col in train_df.columns]
    
    if not available_text_cols:
        st.warning("No text columns found for NLP analysis.")
        return
    
    selected_column = st.selectbox("Select text column to analyze", available_text_cols)
    
    # Combine train and test if available
    if test_df is not None:
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df.copy()
    
    # Filter out missing values
    df_text = df[[selected_column]].dropna()
    df_text[selected_column] = df_text[selected_column].astype(str)
    
    st.metric("Total Text Entries", len(df_text))
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Word Cloud", "Sentiment Analysis", "Top Words", "Text Statistics"]
    )
    
    if analysis_type == "Word Cloud":
        st.subheader("Word Cloud")
        
        # Combine all text
        all_text = ' '.join(df_text[selected_column].values)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
    elif analysis_type == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Analyze sentiment for each text
        sentiments = []
        for text in df_text[selected_column].head(500):  # Limit for performance
            scores = sia.polarity_scores(text)
            sentiments.append({
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            })
        
        sentiment_df = pd.DataFrame(sentiments)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sentiment Distribution**")
            fig = px.histogram(
                sentiment_df,
                x='compound',
                nbins=30,
                title='Sentiment Score Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Sentiment Statistics**")
            st.dataframe(sentiment_df.describe(), use_container_width=True)
        
        # Sentiment by category
        st.write("**Average Sentiment Scores**")
        avg_sentiment = sentiment_df.mean()
        fig = px.bar(
            x=avg_sentiment.index,
            y=avg_sentiment.values,
            title='Average Sentiment Scores'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Top Words":
        st.subheader("Top Words Analysis")
        
        # Tokenize and count words
        stop_words = set(stopwords.words('english'))
        
        all_words = []
        for text in df_text[selected_column].head(1000):  # Limit for performance
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in stop_words]
            all_words.extend(words)
        
        word_freq = pd.Series(all_words).value_counts().head(30)
        
        fig = px.bar(
            x=word_freq.values,
            y=word_freq.index,
            orientation='h',
            title='Top 30 Most Frequent Words'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Word Frequency Table**")
        st.dataframe(word_freq.reset_index().rename(columns={'index': 'Word', 0: 'Frequency'}), use_container_width=True)
        
    elif analysis_type == "Text Statistics":
        st.subheader("Text Statistics")
        
        # Calculate statistics
        text_lengths = df_text[selected_column].str.len()
        word_counts = df_text[selected_column].str.split().str.len()
        
        stats_df = pd.DataFrame({
            'Text Length': text_lengths,
            'Word Count': word_counts
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Text Length Distribution**")
            fig = px.histogram(
                stats_df,
                x='Text Length',
                nbins=30,
                title='Distribution of Text Lengths'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Word Count Distribution**")
            fig = px.histogram(
                stats_df,
                x='Word Count',
                nbins=30,
                title='Distribution of Word Counts'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Statistics Summary**")
        st.dataframe(stats_df.describe(), use_container_width=True)

def show_single_prediction(train_df, model, preprocessor, target_col):
    st.header("🔮 Single Instance Prediction")
    st.write("Make predictions on individual instances and see explanations.")
    
    # Get sample instance
    sample_idx = st.slider("Select instance index", 0, len(train_df) - 1, 0)
    sample_instance = train_df.iloc[sample_idx:sample_idx+1].copy()
    
    st.subheader("Instance Features")
    st.dataframe(sample_instance, use_container_width=True)
    
    # Make prediction
    X_instance = sample_instance.drop(columns=[target_col])
    X_instance_prepared = preprocessor.transform(X_instance)
    X_instance_prepared = pd.DataFrame(X_instance_prepared).replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    prediction = model.predict(X_instance_prepared)[0]
    prediction_proba = model.predict_proba(X_instance_prepared)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction", "Churn" if prediction == 1 else "No Churn")
    
    with col2:
        st.metric("Confidence", f"{max(prediction_proba) * 100:.2f}%")
    
    # Show probabilities
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Class': ['No Churn', 'Churn'],
        'Probability': prediction_proba
    })
    
    fig = px.bar(
        prob_df,
        x='Class',
        y='Probability',
        title='Prediction Probabilities',
        color='Class',
        color_discrete_map={'No Churn': 'green', 'Churn': 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Actual label
    actual_label = "Churn" if (train_df.iloc[sample_idx][target_col] != 0) else "No Churn"
    st.write(f"**Actual Label:** {actual_label}")
    
    if prediction == (train_df.iloc[sample_idx][target_col] != 0).astype(int):
        st.success("✅ Prediction is correct!")
    else:
        st.error("❌ Prediction is incorrect!")

if __name__ == "__main__":
    main()

