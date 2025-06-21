import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -------------------------------
# 🔹 Load the Dataset
# -------------------------------
df = pd.read_csv("twitter_training.csv", header=None)
df.columns = ['id', 'entity', 'sentiment', 'content']

# -------------------------------
# 🔹 Basic Info
# -------------------------------
print("🔹 Dataset Preview:")
print(df.head())

print("\n🔹 Class Distribution:")
print(df['sentiment'].value_counts())

# -------------------------------
# 🔹 Sentiment Distribution Plot
# -------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='sentiment', order=df['sentiment'].value_counts().index, palette='pastel')
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output1.png")
plt.show()

# -------------------------------
# 🔹 Most Frequent Entities Plot
# -------------------------------
top_entities = df['entity'].value_counts().head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_entities.values, y=top_entities.index, palette='viridis')
plt.title("Top 10 Most Mentioned Entities", fontsize=14)
plt.xlabel("Frequency")
plt.ylabel("Entity")
plt.tight_layout()
plt.savefig("output2.png")
plt.show()

# -------------------------------
# 🔹 WordClouds for Each Sentiment
# -------------------------------
sentiments = df['sentiment'].unique()

for sentiment in sentiments:
    text = " ".join(df[df['sentiment'] == sentiment]['content'].astype(str))

    if text.strip():  # Skip empty cases
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='coolwarm'
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud for '{sentiment}' Sentiment", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"wordcloud_{sentiment}.png")
        plt.show()
