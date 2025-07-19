import faiss
import numpy as np
import librosa
import os

# Load FAISS index và tên file
index = faiss.read_index("embeddings/audio_index.faiss")
filenames = np.load("embeddings/filenames.npy")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zcr, axis=1),
        np.mean(centroid, axis=1)
    ])
    
    return features.astype('float32').reshape(1, -1)

#Hàm tìm kiếm
def search(query_path, k=5, return_results=False):
    query_vec = extract_features(query_path)
    distances, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append((filenames[idx], distances[0][i]))

    if return_results:
        return results
    else:
        print(f"\n🔍 Top {k} similar files:")
        for i, (fname, dist) in enumerate(results):
            print(f"{i+1}. {fname} (distance: {dist:.2f})")

# Tìm kiếm với FAISS index động (từ thư mục người dùng chọn)
def search_dynamic(query_path, index, filenames, k=5):
    query_vec = extract_features(query_path).reshape(1, -1)
    distances, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append((filenames[idx], distances[0][i]))

    return results
