import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. BÖLÜM: VERİ YAPILARI (Müfredat Uyumu)
# ==========================================
class Node:
    def __init__(self, movie_title):
        self.movie_title = movie_title
        self.next = None

class MovieLinkedList:
    def __init__(self): 
        self.head = None
        
    def add_movie(self, title):
        new_node = Node(title)
        if not self.head: 
            self.head = new_node
        else:
            curr = self.head
            while curr.next: 
                curr = curr.next
            curr.next = new_node
            
    def get_all(self):
        res, curr = [], self.head
        while curr: 
            res.append(curr.movie_title)
            curr = curr.next
        return res

class ActionStack:
    def __init__(self): 
        self.items = []
        
    def push(self, item): 
        self.items.append(item)
        
    def get_all(self): 
        return self.items[::-1]

class WatchQueue:
    def __init__(self): 
        self.items = []
        
    def enqueue(self, item): 
        self.items.append(item)
        
    # --- YENİ EKLENEN ÇIKARMA FONKSİYONU (FIFO) ---
    def dequeue(self):
        if len(self.items) > 0:
            return self.items.pop(0) # Listenin ilk elemanını (en eski ekleneni) çıkarır
        return None

    def get_all(self): 
        return self.items

# ==========================================
# 2. BÖLÜM: MODEL MİMARİSİ (NCF)
# ==========================================
class NCFModel(nn.Module):
    def __init__(self, num_users, num_items):
        super(NCFModel, self).__init__()
        self.user_embed = nn.Embedding(num_users, 50) 
        self.item_embed = nn.Embedding(num_items, 50)
        self.fc_layers = nn.Sequential(
            nn.Linear(100, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, u, i):
        x = torch.cat([self.user_embed(u), self.item_embed(i)], dim=-1)
        return self.fc_layers(x).squeeze()

# ==========================================
# 3. BÖLÜM: VERİ VE MODEL YÜKLEME
# ==========================================
@st.cache_data
def load_movies():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    cols = ['item_id', 'movie_title', 'release', 'v_release', 'url'] + [f'g{i}' for i in range(19)]
    return pd.read_csv(url, sep='|', names=cols, encoding='latin-1')

movies_df = load_movies()

def load_trained_model():
    model = NCFModel(num_users=943, num_items=1682) 
    try:
        state_dict = torch.load("ncf_model.pth", map_location=torch.device('cpu'))
        if not isinstance(state_dict, dict):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        return None
    except RuntimeError as e:
        st.error(f"Boyut Uyuşmazlığı: {e}")
        return None

trained_model = load_trained_model()

# ==========================================
# 4. BÖLÜM: ARAYÜZ (UI)
# ==========================================
st.set_page_config(page_title="Hibrit Öneri Sistemi", layout="wide")
st.title("🎬 Hibrit Film Öneri Sistemi")

# Sidebar Metrikleri
st.sidebar.header("📊 Model Metrikleri")
st.sidebar.metric("Eğitim Hatası (RMSE)", "1.5 Yıldız", "-11.5 Düşüş")
st.sidebar.metric("Seyreklik Oranı", "%93.70")

# State Yönetimi
if 'history' not in st.session_state: 
    st.session_state.history = ActionStack()
if 'queue' not in st.session_state: 
    st.session_state.queue = WatchQueue()

col1, col2 = st.columns([2, 1])

with col1:
    u_id = st.number_input("Kullanıcı ID (1-943)", 1, 943, 11)
    
    # 1. Öneri Butonu: Collaborative Filtering (NCF)
    if st.button("🚀 Önerileri Hesapla"):
        st.session_state.history.push(f"User {u_id} Analysis")
        if trained_model:
            items = torch.arange(len(movies_df))
            users = torch.full_like(items, u_id)
            with torch.no_grad():
                preds = trained_model(users, items).numpy()
            top_idx = preds.argsort()[-5:][::-1]
            recs = movies_df.iloc[top_idx]['movie_title'].tolist()
            
            st.subheader("🤖 NCF Model Önerileri")
            for r in recs: 
                st.success(f"🎥 {r}")
        else:
            st.warning("Model dosyası ('ncf_model.pth') bulunamadı veya yüklenemedi.")

    st.divider()

    # 2. Öneri: Content-Based (Arama)
    sel_movie = st.selectbox("Bir film seçin:", movies_df['movie_title'].values)

    # --- SIKINTIYI ÇÖZEN EKLEME BUTONU ---
    if st.button("➕ İzleme Sırasına (Daha Sonra İzle) Ekle"):
        st.session_state.queue.enqueue(sel_movie)
        st.success(f"'{sel_movie}' izleme sırasına başarıyla eklendi!")
        st.rerun()

    if st.button("🔍 Benzer Filmleri Bul"):
        st.session_state.history.push(sel_movie)
        
        movie_idx = movies_df[movies_df['movie_title'] == sel_movie].index[0]
        genre_matrix = movies_df.iloc[:, 5:24].values 
        target_vector = genre_matrix[movie_idx]

        dot_product = np.dot(genre_matrix, target_vector)
        norms = np.linalg.norm(genre_matrix, axis=1) * np.linalg.norm(target_vector)
        similarity = dot_product / (norms + 1e-9)

        similarity[movie_idx] = -1 
        top_indices = similarity.argsort()[-5:][::-1]
        recs = movies_df.iloc[top_indices]['movie_title'].tolist()

        st.subheader(f"✨ '{sel_movie}' Benzerleri")
        st.caption("Bu sonuçlar matematiksel Kosinüs Benzerliği kullanılarak hesaplanmıştır.")
        for r in recs: 
            st.info(f"🎞️ {r}")

with col2:
    st.subheader("📚 Veri Yapıları Paneli")
    st.write("**İzleme Geçmişi (Stack - LIFO)**")
    st.write(st.session_state.history.get_all())
    
    st.write("**İzleme Sırası (Queue - FIFO)**")
    st.write(st.session_state.queue.get_all())
    
    # --- YENİ EKLENEN LİSTEDEN ÇIKARMA (DEQUEUE) BUTONU ---
    if st.button("🍿 Sıradaki Filmi İzle (Sıradan Çıkar)"):
        watched_movie = st.session_state.queue.dequeue()
        if watched_movie:
            st.success(f"'{watched_movie}' filmini izlediniz ve sıradan düşüldü.")
            st.rerun() # Ekranı anında günceller
        else:
            st.warning("İzleme sıranızda çıkarılacak film kalmadı!")
