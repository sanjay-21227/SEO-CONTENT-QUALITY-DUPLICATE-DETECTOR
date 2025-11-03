"""
SEO Content Quality Analyzer - Streamlit App (TensorFlow-Free Version)
Bonus Feature: Interactive web interface for real-time URL analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import textstat
import nltk
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import re

# Page configuration
st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6366f1;
    }
    .quality-high {
        color: #10b981;
        font-weight: bold;
    }
    .quality-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .quality-low {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained quality model"""
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Try to load existing model first
    model_paths = [
        '../models/quality_model.pkl',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'quality_model.pkl')
    ]
    
    for model_path in model_paths:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            continue
        except Exception as e:
            # If we get a compatibility error, try to retrain
            if "incompatible dtype" in str(e) or "node array" in str(e):
                st.warning("⚠️ Model compatibility issue detected. Retraining model...")
                return retrain_model_inline()
            continue
    
    # If no model found, try to retrain
    st.warning("⚠️ Model not found. Attempting to train new model...")
    return retrain_model_inline()

def retrain_model_inline():
    """Retrain the model inline if needed"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import os
        
        # Load feature data
        df_features, _ = load_existing_data()
        if df_features is None:
            st.error("❌ No training data available. Please run the data collection notebook first.")
            return None
        
        # Create quality labels
        def create_quality_labels(row):
            word_count = row['word_count']
            readability = row['flesch_reading_ease']
            
            if word_count >= 1500 and readability >= 30:
                return 'High'
            elif word_count >= 800 or (word_count >= 500 and readability >= 40):
                return 'Medium'
            else:
                return 'Low'
        
        df_features['quality_label'] = df_features.apply(create_quality_labels, axis=1)
        
        # Prepare features
        feature_columns = ['word_count', 'sentence_count', 'flesch_reading_ease']
        X = df_features[feature_columns].fillna(0)
        y = df_features['quality_label']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        
        # Save the new model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, 'quality_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        
        accuracy = rf_model.score(X_test, y_test)
        st.success(f"✅ Model retrained successfully! Accuracy: {accuracy:.3f}")
        
        return rf_model
        
    except Exception as e:
        st.error(f"❌ Failed to retrain model: {str(e)}")
        return None

@st.cache_resource
def initialize_vectorizer():
    """Initialize TF-IDF vectorizer for text similarity"""
    return TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True
    )

@st.cache_data
def load_existing_data():
    """Load existing analyzed content"""
    try:
        # Try relative path from streamlit_app directory
        df = pd.read_csv('../data/features.csv')
        df_content = pd.read_csv('../data/extracted_content.csv')
        return df, df_content
    except:
        try:
            # Try absolute path
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            features_path = os.path.join(base_dir, 'data', 'features.csv')
            content_path = os.path.join(base_dir, 'data', 'extracted_content.csv')
            df = pd.read_csv(features_path)
            df_content = pd.read_csv(content_path)
            return df, df_content
        except:
            return None, None

# Utility functions
def clean_text(text):
    """Clean extracted text"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def extract_content(html):
    """Extract title and body text from HTML"""
    try:
        soup = BeautifulSoup(html, 'lxml')
        
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        title = soup.find('title')
        title = title.get_text() if title else ""
        
        body_text = ""
        for tag in ['article', 'main']:
            content = soup.find(tag)
            if content:
                body_text = content.get_text()
                break
        
        if not body_text:
            paragraphs = soup.find_all('p')
            body_text = ' '.join([p.get_text() for p in paragraphs])
        
        title = clean_text(title)
        body_text = clean_text(body_text)
        word_count = len(body_text.split()) if body_text else 0
        
        return title, body_text, word_count
    except Exception as e:
        st.error(f"Error parsing HTML: {str(e)}")
        return "", "", 0

def scrape_url(url):
    """Scrape content from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error scraping URL: {str(e)}")
        return None

def calculate_features(text):
    """Calculate text features"""
    if not text:
        return 0, 0
    
    try:
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        if len(text.split()) >= 100:
            readability = textstat.flesch_reading_ease(text)
        else:
            readability = 0
        
        return sentence_count, round(readability, 2)
    except:
        return 0, 0

def find_similar_content_tfidf(new_text, existing_texts, threshold=0.3):
    """Find similar content using TF-IDF vectorization"""
    if not existing_texts or not new_text:
        return []
    
    try:
        vectorizer = initialize_vectorizer()
        
        # Combine all texts
        all_texts = existing_texts + [new_text]
        
        # Fit vectorizer and transform
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities with the new text (last item)
        new_text_vector = tfidf_matrix[-1]
        existing_vectors = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(new_text_vector, existing_vectors).flatten()
        
        # Find similar documents
        similar_indices = np.where(similarities > threshold)[0]
        
        similar_content = []
        for idx in similar_indices:
            if similarities[idx] < 0.99:  # Exclude exact matches
                similar_content.append({
                    'index': idx,
                    'similarity': round(float(similarities[idx]), 3)
                })
        
        return sorted(similar_content, key=lambda x: x['similarity'], reverse=True)[:3]
    
    except Exception as e:
        st.warning(f"Could not calculate similarity: {str(e)}")
        return []

def analyze_url(url, model):
    """Analyze a URL and return results"""
    with st.spinner("🔍 Analyzing URL..."):
        # Scrape
        html = scrape_url(url)
        if not html:
            return {"error": "Failed to scrape URL"}
        
        # Extract content
        title, body_text, word_count = extract_content(html)
        if word_count == 0:
            return {"error": "No content extracted from URL"}
        
        # Calculate features
        sentence_count, readability = calculate_features(body_text)
        
        # Predict quality
        features = np.array([[word_count, sentence_count, readability]])
        quality_label = model.predict(features)[0]
        quality_proba = model.predict_proba(features)[0]
        
        # Find similar content using existing data
        similar_to = []
        df_features, df_content = load_existing_data()
        
        if df_content is not None and 'body_text' in df_content.columns:
            existing_texts = df_content['body_text'].fillna('').tolist()
            similar_content = find_similar_content_tfidf(body_text, existing_texts)
            
            for sim in similar_content:
                idx = sim['index']
                if idx < len(df_content):
                    similar_to.append({
                        'url': df_content.iloc[idx].get('url', 'Unknown URL'),
                        'similarity': sim['similarity']
                    })
        
        result = {
            'url': url,
            'title': title,
            'word_count': int(word_count),
            'sentence_count': int(sentence_count),
            'readability': float(readability),
            'quality_label': quality_label,
            'quality_confidence': {
                'Low': round(float(quality_proba[0]), 3),
                'Medium': round(float(quality_proba[1]), 3),
                'High': round(float(quality_proba[2]), 3)
            },
            'is_thin': word_count < 500,
            'similar_to': similar_to,
            'body_preview': body_text[:500] + "..." if len(body_text) > 500 else body_text
        }
        
        return result

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🔍 SEO Content Quality Analyzer</p>', unsafe_allow_html=True)
    st.markdown("Analyze web content for quality, readability, and duplicate detection using machine learning.")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        This tool analyzes web content using:
        - **NLP features** (word count, readability)
        - **Machine Learning** (Random Forest classifier)
        - **TF-IDF similarity** (text similarity detection)
        
        **Quality Levels:**
        - 🟢 High: 1500+ words, good readability
        - 🟡 Medium: Standard content
        - 🔴 Low: <500 words or poor readability
        """)
        
        st.header("📈 Dataset Stats")
        df_features, df_content = load_existing_data()
        if df_features is not None:
            st.metric("Total URLs Analyzed", len(df_features))
            st.metric("Avg Word Count", f"{df_features['word_count'].mean():.0f}")
            st.metric("Avg Readability", f"{df_features['flesch_reading_ease'].mean():.1f}")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Please train the model first by running the Jupyter notebook.")
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze URL", "📊 Dataset Overview", "📈 Insights"])
    
    with tab1:
        st.header("Analyze a URL")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url_input = st.text_input(
                "Enter URL to analyze:",
                placeholder="https://example.com/article",
                key="url_input"
            )
        with col2:
            analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
        
        if analyze_button and url_input:
            # Analyze
            result = analyze_url(url_input, model)
            
            if 'error' in result:
                st.error(f"❌ {result['error']}")
            else:
                st.session_state.result = result
                st.session_state.analyzed = True
        
        # Display results
        if st.session_state.analyzed and st.session_state.result:
            result = st.session_state.result
            
            st.success("✅ Analysis Complete!")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Word Count", result['word_count'])
            with col2:
                st.metric("Sentences", result['sentence_count'])
            with col3:
                st.metric("Readability", f"{result['readability']:.1f}")
            with col4:
                quality_color = "quality-high" if result['quality_label'] == 'High' else \
                               "quality-medium" if result['quality_label'] == 'Medium' else "quality-low"
                st.markdown(f"<div class='metric-card'><div class='{quality_color}' style='font-size: 1.5rem;'>{result['quality_label']} Quality</div></div>", 
                           unsafe_allow_html=True)
            
            # Details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📄 Content Details")
                st.markdown(f"**Title:** {result['title']}")
                st.markdown(f"**URL:** {result['url']}")
                st.markdown(f"**Thin Content:** {'⚠️ Yes' if result['is_thin'] else '✅ No'}")
                
                with st.expander("📝 Content Preview"):
                    st.text(result['body_preview'])
            
            with col2:
                st.subheader("📊 Quality Confidence")
                
                # Confidence chart
                conf_df = pd.DataFrame({
                    'Label': list(result['quality_confidence'].keys()),
                    'Confidence': list(result['quality_confidence'].values())
                })
                
                fig = px.bar(
                    conf_df,
                    x='Confidence',
                    y='Label',
                    orientation='h',
                    color='Label',
                    color_discrete_map={'Low': '#ef4444', 'Medium': '#f59e0b', 'High': '#10b981'}
                )
                fig.update_layout(showlegend=False, height=200)
                st.plotly_chart(fig, use_container_width=True)
            
            # Similar content
            if result['similar_to']:
                st.subheader("🔗 Similar Content Found (TF-IDF)")
                for sim in result['similar_to']:
                    st.markdown(f"- [{sim['url']}]({sim['url']}) (Similarity: {sim['similarity']:.2%})")
            else:
                st.info("No similar content found in the dataset.")
    
    with tab2:
        st.header("Dataset Overview")
        
        df_features, df_content = load_existing_data()
        
        if df_features is not None:
            # Display data
            st.subheader("📋 Analyzed URLs")
            if df_content is not None and 'title' in df_content.columns:
                display_df = df_content[['url', 'title', 'word_count']].head(20)
            else:
                display_df = df_features[['url', 'word_count']].head(20)
            st.dataframe(display_df, use_container_width=True)
            
            # Statistics
            st.subheader("📊 Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Pages", len(df_features))
            with col2:
                st.metric("Avg Words", f"{df_features['word_count'].mean():.0f}")
            with col3:
                thin = (df_features['word_count'] < 500).sum()
                st.metric("Thin Content", f"{thin} ({thin/len(df_features)*100:.1f}%)")
        else:
            st.warning("No data available. Run the Jupyter notebook first.")
    
    with tab3:
        st.header("Dataset Insights")
        
        df_features, _ = load_existing_data()
        
        if df_features is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Word count distribution
                fig = px.histogram(
                    df_features,
                    x='word_count',
                    nbins=30,
                    title='Word Count Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Readability distribution
                fig = px.histogram(
                    df_features,
                    x='flesch_reading_ease',
                    nbins=30,
                    title='Readability Score Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            fig = px.scatter(
                df_features,
                x='word_count',
                y='flesch_reading_ease',
                title='Word Count vs Readability',
                labels={'word_count': 'Word Count', 'flesch_reading_ease': 'Readability Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available. Run the Jupyter notebook first.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | TensorFlow-Free Version")

if __name__ == "__main__":
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    main()
