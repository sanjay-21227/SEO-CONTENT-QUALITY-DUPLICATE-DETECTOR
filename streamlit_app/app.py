"""
SEO Content Quality Analyzer - Streamlit App
Bonus Feature: Interactive web interface for real-time URL analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import textstat
import nltk
from sentence_transformers import SentenceTransformer
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

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

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained quality model"""
    try:
        with open('models/quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.error("⚠️ Model file not found. Please train the model first.")
        return None

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_existing_data():
    """Load existing analyzed content"""
    try:
        df = pd.read_csv('data/features.csv')
        df_content = pd.read_csv('data/extracted_content.csv')
        return df, df_content
    except:
        return None, None

# Utility functions
def clean_text(text):
    """Clean extracted text"""
    if not text:
        return ""
    import re
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

def analyze_url(url, model, transformer, existing_embeddings=None):
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
        
        # Generate embedding
        embedding = transformer.encode([body_text])[0]
        
        # Predict quality
        features = np.array([[word_count, sentence_count, readability]])
        quality_label = model.predict(features)[0]
        quality_proba = model.predict_proba(features)[0]
        
        # Find similar content
        similar_to = []
        if existing_embeddings is not None:
            similarities = cosine_similarity([embedding], existing_embeddings)[0]
            similar_indices = np.where(similarities > 0.70)[0]
            
            df_features, _ = load_existing_data()
            for idx in similar_indices:
                if similarities[idx] < 0.99:
                    similar_to.append({
                        'url': df_features.iloc[idx]['url'],
                        'similarity': round(float(similarities[idx]), 3)
                    })
            
            similar_to = sorted(similar_to, key=lambda x: x['similarity'], reverse=True)[:3]
        
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
        - **Semantic similarity** (sentence embeddings)
        
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
    transformer = load_sentence_transformer()
    
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
            # Load existing embeddings for similarity
            df_features, _ = load_existing_data()
            existing_embeddings = None
            if df_features is not None:
                existing_embeddings = np.array([eval(e) for e in df_features['embedding']])
            
            # Analyze
            result = analyze_url(url_input, model, transformer, existing_embeddings)
            
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
                st.subheader("🔗 Similar Content Found")
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
            display_df = df_content[['url', 'title', 'word_count']].head(20)
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
    st.markdown("Built with ❤️ using Streamlit | [GitHub Repository](#)")

if __name__ == "__main__":
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    main()