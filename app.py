from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

BASE_DIR = '/home/arunachal/Programming/apps/recipe_recommender/'
DATA_DIR = os.path.join(BASE_DIR, 'data/')
DATA_PATH = os.path.join(DATA_DIR, 'processed_recipes_enhanced.csv')
GLOVE_PATH = os.getenv('GLOVE_PATH', os.path.join(DATA_DIR, 'glove.6B.100d.txt'))
COUNTVEC_PATH = os.path.join(BASE_DIR, 'count_vectorizer.pkl')
GLOVE_VECTORS_PATH = os.path.join(BASE_DIR, 'glove_vectors.npy')

ps = PorterStemmer()
stop_words = set(stopwords.words('english') + [
    'oil', 'salt', 'pepper', 'powder', 'leaves', 'seeds', 'water', 'paste',
    'sauce', 'spices', 'spice', 'seasoning', 'taste', 'required', 'chopped',
    'sliced', 'minced', 'diced', 'crushed', 'ground', 'finely', 'thin',
    'bc', 'delmonte', 'bhat', 'as', 'per', 'needed'
])

try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load dataset: {str(e)}")

try:
    glove_model = {}
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_model[word] = vector
    print("GloVe embeddings loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load GloVe embeddings: {str(e)}")

try:
    glove_vectors = np.load(GLOVE_VECTORS_PATH)
    print(f"Loaded GloVe vectors from: {GLOVE_VECTORS_PATH}")
except Exception as e:
    raise Exception(f"Failed to load GloVe vectors: {str(e)}")

try:
    with open(COUNTVEC_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    count_matrix = vectorizer.transform(df['ProcessedIngredients'])
    print(f"Loaded CountVectorizer from: {COUNTVEC_PATH}")
except Exception as e:
    raise Exception(f"Failed to load CountVectorizer: {str(e)}")

def clean_ingredients(text):
    noise = r'\b(\d+[-/\d]*\s*(grams?|cups?|tablespoons?|teaspoons?|ml|litres?|kg|pinch|to taste|as needed|as required|finely|sliced|chopped|minced|diced|crushed|ground))\b'
    text = re.sub(noise, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*-\s*.*?(,|$)', ',', text)
    text = re.sub(r'[^\w\s,]', '', text)
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [ps.stem(t) for t in tokens if t not in stop_words and not t.isdigit()]
    return ' '.join(tokens) if tokens else 'empty'

def get_glove_vector(text):
    tokens = text.split()
    vectors = [glove_model.get(ps.stem(t), np.zeros(100)) for t in tokens if ps.stem(t) in glove_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

def get_recommendations(ingredients, cuisine=None, top_n=3):
    if not ingredients or not isinstance(ingredients, str) or len(ingredients.strip()) < 3:
        return [], []
    
    processed_input = clean_ingredients(ingredients)
    if processed_input == 'empty':
        return [], []
    input_tokens = set(processed_input.split())
    input_token_count = len(input_tokens)
    
    filtered_df = df
    if cuisine and cuisine in df['Cuisine'].values:
        filtered_df = df[df['Cuisine'] == cuisine].copy().reset_index(drop=True)
    
    filtered_df['MatchCount'] = filtered_df['ProcessedIngredients'].apply(
        lambda x: sum(token in x.split() for token in input_tokens)
    )
    min_matches = 2 if input_token_count >= 2 else 1
    filtered_df = filtered_df[filtered_df['MatchCount'] >= min_matches]
    if filtered_df.empty:
        return [], []
    
    filtered_indices = filtered_df.index
    filtered_count_matrix = count_matrix[filtered_indices]
    filtered_glove_vectors = glove_vectors[filtered_indices]
    
    try:
        query_count_vec = vectorizer.transform([processed_input])
        count_similarities = cosine_similarity(query_count_vec, filtered_count_matrix).flatten()
        
        query_glove_vec = get_glove_vector(processed_input).reshape(1, -1)
        glove_similarities = cosine_similarity(query_glove_vec, filtered_glove_vectors).flatten()
        
        overlap_scores = []
        for ing in filtered_df['MainIngredients']:
            recipe_tokens = set(ing.lower().split(', '))
            overlap = len(input_tokens.intersection(recipe_tokens)) / max(len(input_tokens), 1)
            overlap_scores.append(overlap)
        overlap_scores = np.array(overlap_scores)
        
        cuisine_scores = np.array([1.0 if cuisine and row['Cuisine'] == cuisine else 0.0 for _, row in filtered_df.iterrows()])
        
        final_scores = (0.4 * count_similarities + 0.3 * glove_similarities + 0.2 * overlap_scores + 0.1 * cuisine_scores)
        top_indices = final_scores.argsort()[-top_n:][::-1]
        results = filtered_df.iloc[top_indices][['TranslatedRecipeName', 'TranslatedIngredients', 'TranslatedInstructions', 'Cuisine']].to_dict('records')
        
        return results, final_scores[top_indices]
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return [], []

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    scores = []
    error = None
    ingredients = None
    cuisine = None
    if request.method == 'POST':
        ingredients = request.form.get('ingredients')
        cuisine = request.form.get('cuisine')
        if ingredients:
            recommendations, scores = get_recommendations(ingredients, cuisine)
            if not recommendations:
                error = "No recipes found. Try different ingredients or cuisine."
        else:
            error = "Please enter ingredients."
    cuisines = sorted(df['Cuisine'].unique().tolist())
    rec_with_scores = list(zip(recommendations, scores)) if recommendations else []
    return render_template('index.html', recommendations=rec_with_scores, cuisines=cuisines, error=error, ingredients=ingredients, cuisine=cuisine)

@app.template_filter('format_score')
def format_score(score):
    return f"{score:.4f}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)