import pandas as pd
import numpy as np
from fcmeans import FCM
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def recommend_songs(song_name):
    # Read audio features from CSV file
    df = pd.read_csv("data.csv")

    # Select relevant features
    audio_features = df[['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
                         'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo']].values

    # Perform fuzzy C-means clustering
    fcm = FCM(n_clusters=5)
    fcm.fit(audio_features)

    clusters = fcm.predict(audio_features)

    df_clusters = pd.DataFrame({'name': df['name'], 'cluster': clusters})

    # Get song recommendations based on a user-specified song
    input_song = df[df['name'] == song_name][['valence', 'acousticness', 'danceability', 'duration_ms',
                                              'energy', 'instrumentalness', 'key', 'liveness',
                                              'loudness', 'mode', 'speechiness', 'tempo']].values
    input_song_cluster = fcm.predict(input_song)[0]

    recommendations = df_clusters[df_clusters['cluster'] == input_song_cluster]['name'].values

    # Calculate cosine similarity and sort recommendations based on similarity score
    input_song_features = input_song.reshape(1, -1)
    cluster_song_indices = df_clusters[df_clusters['cluster'] == input_song_cluster].index
    cluster_song_features = audio_features[cluster_song_indices]
    cosine_similarities = cosine_similarity(input_song_features, cluster_song_features).flatten()
    sorted_recommendations = recommendations[np.argsort(-cosine_similarities)][:10]

    return sorted_recommendations

def opposite_songs(song_name):
    # Read audio features from CSV file
    df = pd.read_csv("data.csv")

    # Select relevant features
    audio_features = df[['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
                         'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo']].values

    # Perform fuzzy C-means clustering
    fcm = FCM(n_clusters=5)
    fcm.fit(audio_features)

    clusters = fcm.predict(audio_features)

    df_clusters = pd.DataFrame({'name': df['name'], 'cluster': clusters})

    # Get song recommendations based on a user-specified song
    input_song = df[df['name'] == song_name][['valence', 'acousticness', 'danceability', 'duration_ms',
                                              'energy', 'instrumentalness', 'key', 'liveness',
                                              'loudness', 'mode', 'speechiness', 'tempo']].values
    input_song_cluster = fcm.predict(input_song)[0]

    # Get songs in the same fuzzy cluster as the input song
    input_cluster_mask = (df_clusters['cluster'] == input_song_cluster)
    input_cluster_indices = df_clusters[input_cluster_mask].index
    input_cluster_songs = df.iloc[input_cluster_indices]

    # Get fuzzy part of other clusters
    other_clusters = set(df_clusters['cluster'].unique()) - {input_song_cluster}
    other_cluster_songs = pd.concat([df.iloc[df_clusters[df_clusters['cluster']==c].index] for c in other_clusters])

    # Select relevant audio features
    input_song_features = input_song[0,:].reshape(1, -1)
    input_cluster_features = audio_features[input_cluster_indices]
    other_cluster_features = other_cluster_songs[['valence', 'acousticness', 'danceability', 'duration_ms',
                                                   'energy', 'instrumentalness', 'key', 'liveness',
                                                   'loudness', 'mode', 'speechiness', 'tempo']].values

    # Compute cosine similarities
    input_similarities = cosine_similarity(input_song_features, input_cluster_features).flatten()
    other_similarities = cosine_similarity(input_song_features, other_cluster_features).flatten()

    # Recommend least similar songs
    similarity_threshold = 0.5
    similarities = np.concatenate([input_similarities, other_similarities])
    song_names = pd.concat([input_cluster_songs['name'], other_cluster_songs['name']])
    sorted_opposite_songs = song_names[np.argsort(similarities)][:10]

    return sorted_opposite_songs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    song = request.json['song']
    recommendations = recommend_songs(song)
    return jsonify(recommendations.tolist())

@app.route('/get_opposite_songs', methods=['POST'])
def get_opposite_songs():
    song = request.json['song']
    opposite_songs_list = opposite_songs(song)
    return jsonify(opposite_songs_list.tolist())

if __name__ == '__main__':
    app.run(debug=True)
