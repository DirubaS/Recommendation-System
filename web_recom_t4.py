import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data and preprocess
df = pd.read_csv("movies_metadata.csv", low_memory=False)
df.drop_duplicates(inplace=True)
links_small = pd.read_csv("links_small.csv")
links_small = links_small[links_small["tmdbId"].notnull()]["tmdbId"].astype("int")
df = df.drop([19730, 29503, 35587])
df["id"] = df["id"].astype("int")
smd = df[df["id"].isin(links_small)].copy()
smd.loc[:, "tagline"] = smd["tagline"].fillna("")
smd.loc[:, "description"] = (smd["overview"] + smd["tagline"]).fillna("")

# Compute TF-IDF matrix and cosine similarity
tf = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 3), min_df=0.0, stop_words="english"
)
tfidf_matrix = tf.fit_transform(smd["description"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
smd = smd.reset_index()
titles = smd["title"]
indices = pd.Series(smd.index, index=smd["title"])


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]  # 0 would be the same movie itself.
    movie_indices = [i[0] for i in sim_scores]
    rec = titles.iloc[movie_indices]
    return rec


def on_button_click():
    user_input = entry.get()
    if user_input:
        # Display user input
        output.config(state=tk.NORMAL)
        output.delete(1.0, tk.END)
        output.insert(tk.END, f"User Input: {user_input}\n\n")

        # Get movie recommendations
        recommendations = get_recommendations(user_input).head(10)

        # Display recommendations
        output.insert(tk.END, "Top 10 Movie Recommendations:\n")
        output.insert(tk.END, recommendations.to_string(index=False))
        output.config(state=tk.DISABLED)


# Tkinter app
root = tk.Tk()
root.title("Diruba's Movie Recommendation App")

# Create and place widgets
label = tk.Label(root, text="Enter a movie name:")
label.pack(pady=10)

entry = tk.Entry(root)
entry.pack(pady=10)

button = tk.Button(root, text="Get Recommendations", command=on_button_click)
button.pack(pady=10)

output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=15, state=tk.DISABLED)
output.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
