from flask import Flask, render_template, request, jsonify, session, Response
import sys
import pickle
import json
import gc
import weakref
from pathlib import Path
from utils import *
from options import args
from models import model_factory
from flask_socketio import SocketIO, emit
from datetime import datetime
import random
import re
import xml.etree.ElementTree as ET

app = Flask(__name__)
app.secret_key = '1903bjk'
socketio = SocketIO(app, cors_allowed_origins="*")

# Memory-efficient chat system
class ChatManager:
    def __init__(self, max_messages=100):  # Reduced from 300
        self.messages = []
        self.active_users = {}
        self.max_messages = max_messages
    
    def add_message(self, message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_messages(self):
        return self.messages
    
    def add_user(self, sid, username):
        self.active_users[sid] = {
            'username': username,
            'connected_at': datetime.now()
        }
    
    def remove_user(self, sid):
        return self.active_users.pop(sid, None)
    
    def get_user_count(self):
        return len(self.active_users)
    
    def get_username(self, sid):
        user = self.active_users.get(sid)
        return user['username'] if user else None
    
    def update_username(self, sid, new_username):
        if sid in self.active_users:
            self.active_users[sid]['username'] = new_username

chat_manager = ChatManager()

def generate_username():
    adjectives = ['Cool', 'Awesome', 'Swift', 'Bright', 'Happy', 'Smart', 'Kind', 'Brave', 'Calm', 'Epic', "Black"]
    nouns = ['Otaku', 'Ninja', 'Samurai', 'Dragon', 'Phoenix', 'Tiger', 'Wolf', 'Eagle', 'Fox', 'Bear']
    return f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(100, 999)}"

def clean_message(message):
    # HTML tag'leri temizle
    message = re.sub(r'<[^>]*>', '', message)
    # Uzunluk kontrolü
    if len(message) > 500:
        message = message[:500]
    return message.strip()

# Lazy loading için wrapper class
class LazyDict:
    def __init__(self, file_path):
        self.file_path = file_path
        self._data = None
        self._loaded = False
    
    def _load_data(self):
        if not self._loaded:
            try:
                with open(self.file_path, "r", encoding="utf-8") as file:
                    self._data = json.load(file)
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load {self.file_path}: {str(e)}")
                self._data = {}
                self._loaded = True
    
    def get(self, key, default=None):
        self._load_data()
        return self._data.get(key, default)
    
    def __contains__(self, key):
        self._load_data()
        return key in self._data
    
    def items(self):
        self._load_data()
        return self._data.items()
    
    def keys(self):
        self._load_data()
        return self._data.keys()
    
    def __len__(self):
        self._load_data()
        return len(self._data)

# Sitemap route'ları
@app.route('/sitemap.xml')
def sitemap():
    """Dinamik sitemap.xml oluşturur"""
    try:
        # XML root element
        urlset = ET.Element('urlset')
        urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
        urlset.set('xmlns:image', 'http://www.google.com/schemas/sitemap-image/1.1')

        # Base URL
        base_url = request.url_root.rstrip('/')
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Ana sayfa
        url = ET.SubElement(urlset, 'url')
        ET.SubElement(url, 'loc').text = f'{base_url}/'
        ET.SubElement(url, 'lastmod').text = current_date
        ET.SubElement(url, 'changefreq').text = 'daily'
        ET.SubElement(url, 'priority').text = '1.0'

        # Chat sayfası
        url = ET.SubElement(urlset, 'url')
        ET.SubElement(url, 'loc').text = f'{base_url}/chat'
        ET.SubElement(url, 'lastmod').text = current_date
        ET.SubElement(url, 'changefreq').text = 'hourly'
        ET.SubElement(url, 'priority').text = '0.8'

        # Anime sayfaları (sadece ilk 50 anime - SEO için)
        if recommendation_system and recommendation_system.id_to_anime:
            anime_count = 0
            for anime_id, anime_data in recommendation_system.id_to_anime.items():
                if anime_count >= 50:  # Reduced from 100
                    break

                try:
                    anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
                    safe_name = anime_name.replace(' ', '-').replace('/', '-').replace('?', '').replace('&', 'and')
                    safe_name = re.sub(r'[^\w\-]', '', safe_name)

                    url = ET.SubElement(urlset, 'url')
                    ET.SubElement(url, 'loc').text = f'{base_url}/anime/{anime_id}/{safe_name}'
                    ET.SubElement(url, 'lastmod').text = current_date
                    ET.SubElement(url, 'changefreq').text = 'weekly'
                    ET.SubElement(url, 'priority').text = '0.6'

                    # Sadece gerekli durumlarda resim URL'si ekle
                    if anime_count < 20:  # Sadece ilk 20 anime için resim
                        image_url = recommendation_system.get_anime_image_url(int(anime_id))
                        if image_url:
                            image_elem = ET.SubElement(url, 'image:image')
                            ET.SubElement(image_elem, 'image:loc').text = image_url
                            ET.SubElement(image_elem, 'image:title').text = anime_name
                            ET.SubElement(image_elem, 'image:caption').text = f'Poster image for {anime_name}'

                    anime_count += 1
                except Exception as e:
                    print(f"Error processing anime {anime_id}: {e}")
                    continue

        # XML'i string'e çevir
        xml_str = ET.tostring(urlset, encoding='unicode')
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        full_xml = xml_declaration + xml_str

        return Response(full_xml, mimetype='application/xml')

    except Exception as e:
        print(f"Sitemap generation error: {e}")
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>',
            mimetype='application/xml')

@app.route('/robots.txt')
def robots_txt():
    """Robots.txt dosyası"""
    robots_content = f"""User-agent: *
Allow: /
Allow: /chat

Sitemap: {request.url_root.rstrip('/')}/sitemap.xml
"""
    return Response(robots_content, mimetype='text/plain')

@app.route('/anime/<int:anime_id>/<path:anime_name>')
def anime_detail(anime_id, anime_name):
    """Anime detay sayfası (SEO için)"""
    if not recommendation_system or str(anime_id) not in recommendation_system.id_to_anime:
        return render_template('error.html', error="Anime not found"), 404

    anime_data = recommendation_system.id_to_anime.get(str(anime_id))
    anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)

    # Anime bilgilerini lazy loading ile al
    image_url = recommendation_system.get_anime_image_url(anime_id)
    mal_url = recommendation_system.get_anime_mal_url(anime_id)
    genres = recommendation_system.get_anime_genres(anime_id)
    anime_type = recommendation_system._get_type(anime_id)

    # Benzer animeler öner (sadece 5 tane)
    similar_animes = []
    try:
        recommendations, _, _ = recommendation_system.get_recommendations([anime_id], num_recommendations=5)
        similar_animes = recommendations
    except:
        pass

    anime_info = {
        'id': anime_id,
        'name': anime_name,
        'image_url': image_url,
        'mal_url': mal_url,
        'genres': genres,
        'similar_animes': similar_animes,
        'type': anime_type
    }

    # JSON-LD structured data oluştur
    structured_data = generate_anime_structured_data(anime_info)

    return render_template('anime_detail.html', anime=anime_info, structured_data=json.dumps(structured_data))

def generate_anime_structured_data(anime_info):
    """Anime için JSON-LD structured data oluşturur"""
    structured_data = {
        "@context": "https://schema.org",
        "@type": anime_info["type"],
        "name": anime_info['name'],
        "url": f"{request.url_root.rstrip('/')}/anime/{anime_info['id']}/{anime_info['name'].replace(' ', '-')}"
    }

    if anime_info['genres']:
        structured_data["genre"] = anime_info['genres']

    if anime_info['image_url']:
        structured_data["image"] = anime_info['image_url']

    if anime_info['mal_url']:
        structured_data["sameAs"] = anime_info['mal_url']

    return structured_data

@app.route('/sitemap-index.xml')
def sitemap_index():
    """Sitemap index dosyası"""
    try:
        sitemapindex = ET.Element('sitemapindex')
        sitemapindex.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')

        base_url = request.url_root.rstrip('/')
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Ana sitemap
        sitemap = ET.SubElement(sitemapindex, 'sitemap')
        ET.SubElement(sitemap, 'loc').text = f'{base_url}/sitemap.xml'
        ET.SubElement(sitemap, 'lastmod').text = current_date

        xml_str = ET.tostring(sitemapindex, encoding='unicode')
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        full_xml = xml_declaration + xml_str

        return Response(full_xml, mimetype='application/xml')

    except Exception as e:
        print(f"Sitemap index generation error: {e}")
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></sitemapindex>',
            mimetype='application/xml')

@app.route('/chat')
def chat():
    return render_template('chat.html')

# SocketIO event'leri
@socketio.on('connect')
def on_connect():
    username = generate_username()
    chat_manager.add_user(request.sid, username)

    # Kullanıcıya mevcut mesajları gönder
    emit('chat_history', chat_manager.get_messages())

    # Kullanıcı katıldı mesajı
    join_message = {
        'username': 'System',
        'message': f'{username} joined the chat',
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'system'
    }

    chat_manager.add_message(join_message)
    emit('new_message', join_message, broadcast=True)
    emit('user_count', chat_manager.get_user_count(), broadcast=True)

@socketio.on('disconnect')
def on_disconnect():
    user = chat_manager.remove_user(request.sid)
    if user:
        username = user['username']
        leave_message = {
            'username': 'System',
            'message': f'{username} left the chat',
            'timestamp': datetime.now().strftime('%H:%M'),
            'type': 'system'
        }

        chat_manager.add_message(leave_message)
        emit('new_message', leave_message, broadcast=True)
        emit('user_count', chat_manager.get_user_count(), broadcast=True)

@socketio.on('send_message')
def handle_message(data):
    username = chat_manager.get_username(request.sid)
    if not username:
        return

    message = clean_message(data.get('message', ''))
    if not message:
        return

    message_obj = {
        'username': username,
        'message': message,
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'user'
    }

    chat_manager.add_message(message_obj)
    emit('new_message', message_obj, broadcast=True)

@socketio.on('change_username')
def handle_username_change(data):
    old_username = chat_manager.get_username(request.sid)
    if not old_username:
        return

    new_username = clean_message(data.get('username', ''))
    if not new_username or len(new_username) < 2:
        return

    chat_manager.update_username(request.sid, new_username)

    change_message = {
        'username': 'System',
        'message': f'{old_username} changed name to {new_username}',
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'system'
    }

    chat_manager.add_message(change_message)
    emit('new_message', change_message, broadcast=True)
    emit('username_changed', {'username': new_username})

class AnimeRecommendationSystem:
    def __init__(self, checkpoint_path, dataset_path, animes_path, images_path, mal_urls_path, type_seq_path, genres_path):
        self.model = None
        self.dataset = None
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.animes_path = animes_path
        
        # Lazy loading ile memory optimization
        self.id_to_anime = LazyDict(animes_path)
        self.id_to_url = LazyDict(images_path)
        self.id_to_mal_url = LazyDict(mal_urls_path)
        self.id_to_type_seq = LazyDict(type_seq_path)
        self.id_to_genres = LazyDict(genres_path)
        
        # Cache için weak reference kullan
        self._cache = {}
        
        self.load_model_and_data()

    def load_model_and_data(self):
        try:
            print("Loading model and data...")
            args.bert_max_len = 128

            # Dataset'i yükle
            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)

            print("31")
            self.model = model_factory(args)
            print("3131")
            self.load_checkpoint()

            # Garbage collection
            gc.collect()
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def load_checkpoint(self):
        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Checkpoint'i bellekten temizle
            del checkpoint
            gc.collect()
            
        except Exception as e:
            raise Exception(f"Failed to load checkpoint from {self.checkpoint_path}: {str(e)}")

    def get_anime_genres(self, anime_id):
        genres = self.id_to_genres.get(str(anime_id), [])
        return [genre.title() for genre in genres] if genres else []

    def get_all_animes(self):
        """Tüm anime listesini döndürür - cache kullanır"""
        cache_key = 'all_animes'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        animes = []
        # Sadece gerekli durumlarda yükle
        for k, v in list(self.id_to_anime.items())[:1000]:  # İlk 1000 anime
            anime_name = v[0] if isinstance(v, list) and len(v) > 0 else str(v)
            animes.append((int(k), anime_name))
        
        animes.sort(key=lambda x: x[1])
        self._cache[cache_key] = animes
        return animes

    def get_anime_image_url(self, anime_id):
        return self.id_to_url.get(str(anime_id), None)

    def get_anime_mal_url(self, anime_id):
        return self.id_to_mal_url.get(str(anime_id), None)

    def get_filtered_anime_pool(self, filters):
        """Filtrelere göre anime havuzunu önceden filtreler"""
        if not filters:
            return None

        if filters.get('show_hentai') and len([k for k, v in filters.items() if v]) == 1:
            hentai_animes = []
            # Sadece gerekli verileri kontrol et
            for anime_id_str in list(self.id_to_anime.keys())[:500]:  # Limit
                anime_id = int(anime_id_str)
                if self._is_hentai(anime_id):
                    hentai_animes.append(anime_id)
            return hentai_animes

        return None

    def _is_hentai(self, anime_id):
        """Anime'nin hentai olup olmadığını kontrol eder"""
        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 3:
            return False
        return type_seq_info[2]

    def _get_type(self, anime_id):
        """Anime tipini döndürür"""
        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 2:
            return "Unknown"
        return type_seq_info[1]

    def get_recommendations(self, favorite_anime_ids, num_recommendations=20, filters=None):  # Reduced from 40
        try:
            if not favorite_anime_ids:
                return [], [], "Please add some favorite animes first!"

            smap = self.dataset["smap"]
            inverted_smap = {v: k for k, v in smap.items()}

            converted_ids = []
            for anime_id in favorite_anime_ids:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])

            if not converted_ids:
                return [], [], "None of the selected animes are in the model vocabulary!"

            # Hentai filtresi özel durumu
            filtered_pool = self.get_filtered_anime_pool(filters)
            if filtered_pool is not None:
                return self._get_recommendations_from_pool(favorite_anime_ids, filtered_pool, num_recommendations, filters)

            # Normal öneriler
            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

            max_predictions = min(75, len(inverted_smap))  # Reduced from 125

            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]
                top_scores, top_indices = torch.topk(last_logits, k=max_predictions, dim=1)

            recommendations = []
            scores = []

            for idx, score in zip(top_indices.numpy()[0], top_scores.detach().numpy()[0]):
                if idx in inverted_smap:
                    anime_id = inverted_smap[idx]

                    if anime_id in favorite_anime_ids:
                        continue

                    if str(anime_id) in self.id_to_anime:
                        # Filtreleme kontrolü
                        if filters and not self._should_include_anime(anime_id, filters):
                            continue

                        anime_data = self.id_to_anime.get(str(anime_id))
                        anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
                        
                        # Lazy loading ile image ve mal url al
                        image_url = self.get_anime_image_url(anime_id)
                        mal_url = self.get_anime_mal_url(anime_id)

                        recommendations.append({
                            'id': anime_id,
                            'name': anime_name,
                            'score': float(score),
                            'image_url': image_url,
                            'mal_url': mal_url,
                            'genres': self.get_anime_genres(anime_id)
                        })
                        scores.append(float(score))

                        if len(recommendations) >= num_recommendations:
                            break

            # Memory cleanup
            del logits, last_logits, top_scores, top_indices
            gc.collect()

            return recommendations, scores, f"Found {len(recommendations)} recommendations!"

        except Exception as e:
            return [], [], f"Error during prediction: {str(e)}"

    def _get_recommendations_from_pool(self, favorite_anime_ids, anime_pool, num_recommendations, filters):
        """Önceden filtrelenmiş anime havuzundan öneriler alır"""
        try:
            smap = self.dataset
            converted_ids = []
            for anime_id in favorite_anime_ids:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])

            if not converted_ids:
                return [], [], "None of the selected animes are in the model vocabulary!"

            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]

            # Anime havuzundaki her anime için skor hesapla
            anime_scores = []
            for anime_id in anime_pool:
                if anime_id in favorite_anime_ids:
                    continue

                if anime_id in smap:
                    model_id = smap[anime_id]
                    if model_id < last_logits.shape[1]:
                        score = last_logits[0, model_id].item()
                        anime_scores.append((anime_id, score))

            # Skorlara göre sırala
            anime_scores.sort(key=lambda x: x[1], reverse=True)

            recommendations = []
            for anime_id, score in anime_scores[:num_recommendations]:
                if str(anime_id) in self.id_to_anime:
                    anime_data = self.id_to_anime.get(str(anime_id))
                    anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
                    
                    recommendations.append({
                        'id': anime_id,
                        'name': anime_name,
                        'score': float(score),
                        'image_url': self.get_anime_image_url(anime_id),
                        'mal_url': self.get_anime_mal_url(anime_id),
                        'genres': self.get_anime_genres(anime_id)
                    })

            # Memory cleanup
            del logits, last_logits
            gc.collect()

            return recommendations, [r['score'] for r in recommendations], f"Found {len(recommendations)} filtered recommendations!"

        except Exception as e:
            return [], [], f"Error during filtered prediction: {str(e)}"

    def _should_include_anime(self, anime_id, filters):
        """Filtrelere göre anime'nin dahil edilip edilmeyeceğini kontrol eder"""
        if 'blacklisted_animes' in filters:
            if anime_id in filters['blacklisted_animes']:
                return False

        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 2:
            return True

        anime_type = type_seq_info[0]
        is_sequel = type_seq_info[1]
        is_hentai = type_seq_info[2]

        # Sequel filtresi
        if 'show_sequels' in filters:
            if not filters['show_sequels'] and is_sequel:
                return False

        # Hentai filtresi
        if 'show_hentai' in filters:
            if filters['show_hentai']:
                if not is_hentai:
                    return False
            else:
                if is_hentai:
                    return False

        # Tür filtreleri
        if 'show_movies' in filters:
            if not filters['show_movies'] and anime_type == 'MOVIE':
                return False

        if 'show_tv' in filters:
            if not filters['show_tv'] and anime_type == 'TV':
                return False

        if 'show_ova' in filters:
            if not filters['show_ova'] and anime_type in ['ONA', 'OVA', 'SPECIAL']:
                return False

        return True

recommendation_system = None

@app.route('/')
def index():
    if recommendation_system is None:
        return render_template('error.html', error="Recommendation system not initialized. Please check server logs.")

    animes = recommendation_system.get_all_animes()
    return render_template('index.html', animes=animes)

@app.route('/api/search_animes')
def search_animes():
    query = request.args.get('q', '').lower()
    animes = []
    
    # Sadece ilk 200 anime'yi arama - performance için
    count = 0
    for k, v in recommendation_system.id_to_anime.items():
        if count >= 200:
            break
            
        anime_names = v if isinstance(v, list) else [v]
        match_found = False
        
        for name in anime_names:
            if query in name.lower():
                match_found = True
                break

        if not query or match_found:
            main_name = anime_names[0] if anime_names else "Unknown"
            animes.append((int(k), main_name))
            count += 1

    animes.sort(key=lambda x: x[1])
    return jsonify(animes)

@app.route('/api/add_favorite', methods=['POST'])
def add_favorite():
    if 'favorites' not in session:
        session['favorites'] = []

    data = request.get_json()
    anime_id = int(data['anime_id'])

    if anime_id not in session['favorites']:
        # Maksimum 20 favori anime (memory için)
        if len(session['favorites']) >= 20:
            return jsonify({'success': False, 'message': 'Maximum 20 favorite animes allowed'})
        
        session['favorites'].append(anime_id)
        session.modified = True
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/api/remove_favorite', methods=['POST'])
def remove_favorite():
    if 'favorites' not in session:
        session['favorites'] = []

    data = request.get_json()
    anime_id = int(data['anime_id'])

    if anime_id in session['favorites']:
        session['favorites'].remove(anime_id)
        session.modified = True
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/api/clear_favorites', methods=['POST'])
def clear_favorites():
    session['favorites'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/get_favorites')
def get_favorites():
    if 'favorites' not in session:
        session['favorites'] = []

    favorite_animes = []
    for anime_id in session['favorites']:
        if str(anime_id) in recommendation_system.id_to_anime:
            anime_data = recommendation_system.id_to_anime.get(str(anime_id))
            anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
            favorite_animes.append({'id': anime_id, 'name': anime_name})

    return jsonify(favorite_animes)


@app.route('/api/get_recommendations', methods=['POST'])
def get_recommendations():
    if 'favorites' not in session or not session['favorites']:
        return jsonify({'success': False, 'message': 'Please add some favorite animes first!'})

    data = request.get_json() or {}
    filters = data.get('filters', {})

    # Blacklist bilgisini ekle
    blacklisted_animes = data.get('blacklisted_animes', [])
    if blacklisted_animes:
        filters['blacklisted_animes'] = blacklisted_animes

    recommendations, scores, message = recommendation_system.get_recommendations(
        session['favorites'],
        filters=filters
    )

    if recommendations:
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'message': message
        })
    else:
        return jsonify({'success': False, 'message': message})


@app.route('/api/mal_logo')
def get_mal_logo():
    # MyAnimeList logo URL'ini döndür
    return jsonify({
        'success': True,
        'logo_url': 'https://cdn.myanimelist.net/img/sp/icon/apple-touch-icon-256.png'
    })


def main():
    global recommendation_system

    args.num_items = 12689

    try:
        images_path = "Data/id_to_url.json"
        mal_urls_path = "Data/id_to_malurl.json"
        type_seq_path = "Data/id_to_typenseq.json"

        if not os.path.exists(images_path):
            print(f"Warning: {images_path} not found. Images will not be displayed.")

        if not os.path.exists(mal_urls_path):
            print(f"Warning: {mal_urls_path} not found. MAL links will not be available.")

        recommendation_system = AnimeRecommendationSystem(
            "Data/AnimeRatings54M/pretrained_bert.pth",
            "Data/preprocessed/AnimeRatings54M_min_rating7-min_uc10-min_sc10-splitleave_one_out/dataset.pkl",
            "Data/animes.json",
            images_path,
            mal_urls_path,
            type_seq_path,
            "Data/id_to_genres.json"
        )
        print("Recommendation system initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize recommendation system: {e}")
        sys.exit(1)

    app.run(debug=False, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
