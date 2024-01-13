import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("movies_metadata.csv", low_memory=False)
df.drop_duplicates(inplace=True)
links_small = pd.read_csv("links_small.csv")
links_small = links_small[links_small["tmdbId"].notnull()]["tmdbId"].astype("int")
df = df.drop([19730, 29503, 35587])
df["id"] = df["id"].astype("int")
smd = df[df["id"].isin(links_small)].copy()

# Use loc to avoid SettingWithCopyWarning
smd.loc[:, "tagline"] = smd["tagline"].fillna("")
smd.loc[:, "description"] = (smd["overview"] + smd["tagline"]).fillna("")

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

    # Use loc to avoid SettingWithCopyWarning
    print(smd.loc[idx, "description"])

    rec = titles.iloc[movie_indices]
    print(rec)
    return rec


get_recommendations("The Godfather").head(10)
