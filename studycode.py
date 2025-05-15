import pandas as pd
import jieba
import re
from collections import Counter
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import json
import os

# Set up Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters display
plt.rcParams['axes.unicode_minus'] = False  # For minus sign display

print("Starting product review analysis...")

# 1. Import data
print("Importing data...")
excel_file = 'all.xlsx'# Enter file name
df = pd.read_excel(excel_file)

# Check column names and print
print(f"Excel file column names: {df.columns.tolist()}")

# Assume review content column is 'Review Content', replace with actual column name if different
review_column = '评价内容'
if review_column not in df.columns:
    possible_names = [col for col in df.columns if '评' in col or '内容' in col or '评论' in col]
    if possible_names:
        review_column = possible_names[0]
        print(f"Using column '{review_column}' as review content")
    else:
        raise KeyError(f"Review content column not found, please check Excel file and specify correct column name")

# 2. Data cleaning
print("Cleaning data...")
df = df.drop_duplicates(subset=[review_column])
df = df.dropna(subset=[review_column])

# 3. Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Word segmentation
    words = jieba.lcut(text)
    # Remove stopwords (simplified version)
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '也', '都', '很', '这个', '这', '那', '就是', '还'}
    words = [w for w in words if w not in stopwords and len(w) > 1]
    return words

# Apply preprocessing
print("Performing text preprocessing...")
df['processed_words'] = df[review_column].apply(preprocess_text)

# 4. Feature extraction
# Predefined product feature categories
feature_keywords = {
    'Performance': ['运行', '速度', '快', '配置', '性能', '流畅', '卡顿', '处理器'],
    'Appearance': ['外观', '设计', '漂亮', '颜值', '好看', '美观', '时尚'],
    'Screen': ['屏幕', '显示', '分辨率', '色彩', '清晰', '亮度'],
    'Battery': ['电池', '续航', '充电', '耗电', '电量'],
    'Value': ['价格', '性价比', '值', '实惠', '划算', '便宜']
}

# Extract features from each review
def extract_features(words):
    found_features = {}
    for feature, keywords in feature_keywords.items():
        for word in words:
            if word in keywords:
                if feature in found_features:
                    found_features[feature].append(word)
                else:
                    found_features[feature] = [word]
    return found_features

print("Extracting product features...")
df['features'] = df['processed_words'].apply(extract_features)

# 5. Sentiment analysis
def analyze_sentiment(text):
    if not isinstance(text, str):
        return 0.5
    try:
        s = SnowNLP(text)
        return s.sentiments  # Returns score between 0-1, closer to 1 is more positive
    except:
        return 0.5  # Return neutral sentiment in case of exception

print("Performing sentiment analysis...")
df['sentiment_score'] = df[review_column].apply(analyze_sentiment)

# 6. Feature-sentiment mapping
def feature_sentiment_mapping(row):
    features = row['features']
    sentiment = row['sentiment_score']
    result = {}
    for feature in features:
        result[feature] = sentiment
    return result

df['feature_sentiment'] = df.apply(feature_sentiment_mapping, axis=1)

# 7. Results analysis
# Count feature mentions
feature_mentions = Counter()
for features in df['features']:
    for feature in features:
        feature_mentions[feature] += 1

# Calculate average sentiment scores for each feature
feature_sentiments = {}
feature_counts = {}
for _, row in df.iterrows():
    for feature, sentiment in row['feature_sentiment'].items():
        if feature in feature_sentiments:
            feature_sentiments[feature] += sentiment
            feature_counts[feature] += 1
        else:
            feature_sentiments[feature] = sentiment
            feature_counts[feature] = 1

avg_sentiments = {f: s/feature_counts[f] for f, s in feature_sentiments.items() if f in feature_counts}

# 8. Visualization
print("Generating visualization charts...")
# Feature mention frequency bar chart
plt.figure(figsize=(10, 6))
plt.bar(feature_mentions.keys(), feature_mentions.values())
plt.title('Product Feature Mention Frequency')
plt.xlabel('Features')
plt.ylabel('Mention Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_frequency.png')
print("Feature frequency chart generated: feature_frequency.png")

# Feature sentiment analysis bar chart
plt.figure(figsize=(10, 6))
plt.bar(avg_sentiments.keys(), avg_sentiments.values(), color=['green' if v > 0.5 else 'red' for v in avg_sentiments.values()])
plt.title('Product Feature Sentiment Analysis')
plt.xlabel('Features')
plt.ylabel('Sentiment Score (Higher is more positive)')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('feature_sentiment.png')
print("Sentiment analysis chart generated: feature_sentiment.png")

# 9. Weka data preparation
print("\n=== Weka Analysis Data Preparation ===")

def prepare_weka_data():
    try:
        # Create feature vectors
        feature_vectors = []
        headers = []
        
        # Add feature names
        for feature in feature_keywords.keys():
            headers.append(f'has_{feature}')
        
        headers.append('sentiment_score')
        headers.append('class')
        
        for _, row in df.iterrows():
            # Initialize feature vector
            features = []
            for feature in feature_keywords.keys():
                features.append(1 if feature in row['features'] else 0)
            
            # Add sentiment score
            features.append(row['sentiment_score'])
            
            # Classification (e.g., positive/negative)
            rating_class = "positive" if row['sentiment_score'] > 0.6 else "negative"
            
            feature_vectors.append(features + [rating_class])
        
        # Create DataFrame for export
        weka_df = pd.DataFrame(feature_vectors, columns=headers)
        
        # Export to CSV (can be imported into Weka)
        weka_df.to_csv('reviews_for_weka.csv', index=False)
        
        # Also export in ARFF format
        with open('reviews_for_weka.arff', 'w', encoding='utf-8') as f:
            f.write('@RELATION reviews\n\n')
            
            # Write attributes
            for feature in feature_keywords.keys():
                f.write(f'@ATTRIBUTE has_{feature} {{0,1}}\n')
            
            f.write('@ATTRIBUTE sentiment_score NUMERIC\n')
            f.write('@ATTRIBUTE class {positive,negative}\n\n')
            
            # Write data
            f.write('@DATA\n')
            for vector in feature_vectors:
                f.write(','.join(map(str, vector)) + '\n')
        
        print("Weka data preparation completed:")
        print("- CSV format: reviews_for_weka.csv")
        print("- ARFF format: reviews_for_weka.arff")
        print("You can now import these files into Weka for classification and clustering analysis")
        
    except Exception as e:
        print(f"Error preparing Weka data: {e}")

prepare_weka_data()

# 10. SketchEngine data preparation
print("\n=== SketchEngine Analysis Data Preparation ===")

def export_for_sketchengine():
    try:
        # Create a text file containing all reviews
        with open('reviews_for_sketchengine.txt', 'w', encoding='utf-8') as f:
            for review in df[review_column]:
                if isinstance(review, str):
                    f.write(review + '\n\n')
        
        print("SketchEngine data preparation completed:")
        print("- Text file: reviews_for_sketchengine.txt")
        print("Usage instructions:")
        print("1. Log in to SketchEngine website (https://www.sketchengine.eu/)")
        print("2. Create a new corpus")
        print("3. Upload the reviews_for_sketchengine.txt file")
        print("4. Perform word frequency, collocation, and keyword analysis")
        
    except Exception as e:
        print(f"Error preparing SketchEngine data: {e}")

export_for_sketchengine()

# 11. Generate comprehensive report
print("\n=== Generating Comprehensive Analysis Report ===")

def create_comprehensive_report():
    try:
        report = {
            "Basic Statistics": {
                "Total Reviews": len(df),
                "Average Sentiment Score": float(df['sentiment_score'].mean()),
                "Feature Mention Frequency": dict(feature_mentions),
                "Feature Average Sentiment": {k: float(v) for k, v in avg_sentiments.items()}
            },
            "Tool Usage Instructions": {
                "Basic Analysis": "Text processing and sentiment analysis using Python, Jieba, and SnowNLP",
                "Weka Usage": "Import reviews_for_weka.arff file and use classifiers (e.g., J48 decision tree, Naive Bayes) for classification",
                "SketchEngine Usage": "Upload reviews_for_sketchengine.txt to create a corpus and perform word frequency and collocation analysis"
            },
            "Analysis Result Interpretation": {
                "Basic Statistics Explanation": "This analysis shows the product features that users focus on most and their overall sentiment towards these features",
                "Feature Frequency Interpretation": "Features with higher frequency are the aspects users care most about and should be prioritized",
                "Sentiment Analysis Interpretation": "Features with sentiment scores below 0.5 indicate user dissatisfaction and need improvement"
            }
        }
        
        # Output to JSON
        with open('comprehensive_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        print("Comprehensive analysis report generated: comprehensive_analysis.json")
        print("Next steps and recommendations:")
        print("1. View feature_frequency.png and feature_sentiment.png to understand product feature distribution and sentiment")
        print("2. Analyze reviews_for_weka.arff in Weka to identify feature relationships")
        print("3. Discover language patterns in reviews using SketchEngine")
        
    except Exception as e:
        print(f"Error generating comprehensive report: {e}")

create_comprehensive_report()

print("\nAnalysis complete! All results and data have been generated for use in research report writing.")
