<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anime Recommendation System</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='purpletheme.css') }}" id="theme-purple">
    <link rel="preload" href="{{ url_for('static', filename='purpletheme.css') }}" as="style">

    <style>
        /* Critical CSS - Sayfa yükleme sırasında görünümü engellemek için */
        html {
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
        }

        html.loaded {
            visibility: visible;
            opacity: 1;
        }

        /* Loading screen */
        .page-loader {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            opacity: 1;
            transition: opacity 0.5s ease-out;
        }

        .page-loader.hidden {
            opacity: 0;
            pointer-events: none;
        }

        .loader-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Preload critical animations */
        .preload-animations {
            position: absolute;
            top: -9999px;
            left: -9999px;
            opacity: 0;
            pointer-events: none;
        }

    

        .theme-option:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(3px);
        }

        .theme-option.active {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .theme-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }

        .theme-option.active .theme-color {
            border-color: rgba(255, 255, 255, 0.8);
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
        }

        .theme-purple .theme-color {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .theme-green .theme-color {
            background: linear-gradient(45deg, #4caf50, #2e7d32);
        }

        .theme-orange .theme-color {
            background: linear-gradient(45deg, #ffa500, #ff4500);
        }
       

        /* Mobile responsive */
        @media (max-width: 768px) {
            .theme-switcher {
                top: 15px;
                right: 15px;
                padding: 6px;
            }

            .theme-switcher-toggle {
                font-size: 18px;
                padding: 6px;
            }

            .theme-options {
                right: -20px;
                min-width: 130px;
            }

            .theme-option {
                font-size: 13px;
                padding: 6px 10px;
            }
        }
    </style>
</head>
<body>
    <!-- Page Loader -->
    <div class="page-loader" id="pageLoader">
        <div class="loader-spinner"></div>
    </div>

    <!-- Preload animations -->
    <div class="preload-animations">
        <div class="theme-switcher-toggle"></div>
        <div class="theme-option"></div>
        <div class="anime-card"></div>
    </div>


    <button class="back-button" onclick="showFavorites()">← Back to Favorites</button>

    <div class="container">
        <div class="header">
            <h1>Just one episode...</h1>
            <p>Discover your next favorite anime</p>
        </div>

        <!-- Favorites Section -->
        <div class="favorites-section" id="favoritesSection">
            <h2 class="section-title">Select Your Favorite Animes</h2>

           <div class="search-container" style="position: relative;">
  <input type="text" class="search-input" id="animeSearch" placeholder="Search for an anime..." autocomplete="off">
  <button class="blacklist-button" id="blacklistButton" onclick="openBlacklistModal()" title="Manage Blacklist">
    🗑️
    <span class="blacklist-counter" id="blacklistCounter">0</span>
</button>
               <div class="blacklist-modal" id="blacklistModal">
    <div class="blacklist-content">
        <div class="blacklist-header">
            <h2 class="blacklist-title">Blacklisted Animes</h2>
            <button class="blacklist-close" onclick="closeBlacklistModal()">×</button>
        </div>
        <div class="blacklist-list" id="blacklistList"></div>
        <div class="blacklist-actions">
            <button class="clear-blacklist-btn" onclick="clearBlacklist()">Clear All Blacklist</button>
        </div>
    </div>
</div>
  <!-- Modern SVG ikonu -->
  <svg class="search-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
       style="width: 20px; height: 20px; position: absolute; right: 10px; top: 50%; transform: translateY(-50%); color: #888;">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>

  <div class="anime-dropdown" id="animeDropdown"></div>
</div>


            <div class="favorites-list" id="favoritesList">
                <div class="empty-state">No favorites selected yet. Search and add some animes!</div>
            </div>

            <div class="action-buttons">
                <button class="btn btn-primary" onclick="getRecommendations()">Get Recommendations</button>
                <button class="btn btn-secondary" onclick="clearFavorites()">Clear All</button>
            </div>
        </div>

        <!-- Recommendations Section -->

        <div class="recommendations-section" id="recommendationsSection">
            <div class="favorites-section">
                <h2 class="section-title">Recommended for You</h2>
                <div class="filter-container">
                    <div class="filter-buttons">
                        <button class="filter-btn" data-filter="show_sequels" onclick="toggleFilter(this)">Sequels</button>
                        <button class="filter-btn active" data-filter="show_movies" onclick="toggleFilter(this)">Movies</button>
                        <button class="filter-btn active" data-filter="show_tv" onclick="toggleFilter(this)">TV Series</button>
                        <button class="filter-btn" data-filter="show_ova" onclick="toggleFilter(this)">OVA/ONA/SPECIAL</button>
                    </div>
                    <button class="apply-filters-btn" onclick="applyFilters()">Apply Filters</button>
                </div>
                <div class="recommendations-grid" id="recommendationsGrid"></div>
            </div>
        </div>

        <div class="about-section" id="aboutSection">
            <div class="about-content">
                <div class="about-item">
                    <h3>About the Model</h3>
                    <p>Despite its strong performance, I couldn't find any anime recommendation websites using a BERT-based recommender system. This recommendation system uses a BERT transformer specifically for anime recommendations. Unlike other systems, it takes only users' favorite anime titles as input, rather than ratings. I believe the BERT model handled the recommendation task very well, including effectively addressing the cold start problem by generating meaningful recommendations even with limited user data.</p>
                </div>

                <div class="about-item">
                    <h3>Purpose</h3>
                    <p>This project was developed purely for entertainment, with the aim of helping people discover new anime and enjoy their viewing experience. At the same time, it serves as an experiment — exploring how modern NLP models like BERT can be applied to reimagine recommendation systems. By shifting the focus from numeric ratings to users' actual favorites, the model seeks to capture deeper patterns of preference and better reflect individual taste.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let allAnimes = [];
        let malLogoUrl = '';
        let currentTheme = 'purple';

        function initializePage() {
            const pageLoader = document.getElementById('pageLoader');
            const html = document.documentElement;

            html.classList.add('loaded');

            setTimeout(() => {
                pageLoader.classList.add('hidden');
                setTimeout(() => {
                    pageLoader.style.display = 'none';
                }, 500);
            }, 300);
        }

        function preloadCriticalResources() {
            const themes = ['purple'];
            themes.forEach(theme => {
                const link = document.createElement('link');
                link.rel = 'preload';
                link.as = 'style';
                link.href = `/static/${theme}theme.css`;
                document.head.appendChild(link);
            });

     
        }

        function toggleThemeOptions() {
            const themeOptions = document.getElementById('themeOptions');
            themeOptions.classList.toggle('show');
        }


        document.addEventListener('click', function(e) {
            if (!e.target.closest('.theme-switcher')) {
                document.getElementById('themeOptions').classList.remove('show');
            }
        });


        document.addEventListener('DOMContentLoaded', function() {
            initializePage();
            preloadCriticalResources();
            loadBlacklist();

            loadSavedTheme();
            loadFavorites();
            loadMalLogo();

            const searchInput = document.getElementById('animeSearch');
            const dropdown = document.getElementById('animeDropdown');

           
            let searchTimeout;
            searchInput.addEventListener('input', function() {
                clearTimeout(searchTimeout);
                const query = this.value.trim();

                searchTimeout = setTimeout(() => {
                    if (query.length > 0) {
                        searchAnimes(query);
                    } else {
                        hideDropdown();
                    }
                }, 300); 
            });

         
            document.addEventListener('click', function(e) {
                if (!e.target.closest('.search-container')) {
                    hideDropdown();
                }
            });
        });

       
        async function loadMalLogo() {
            try {
                const cachedLogo = localStorage.getItem('malLogoUrl');
                if (cachedLogo) {
                    malLogoUrl = cachedLogo;
                    return;
                }

                const response = await fetch('/api/mal_logo');
                const data = await response.json();
                if (data.success) {
                    malLogoUrl = data.logo_url;
                    localStorage.setItem('malLogoUrl', malLogoUrl);
                }
            } catch (error) {
                console.error('Error loading MAL logo:', error);
            }
        }

        let searchCache = new Map();
        async function searchAnimes(query) {
            try {
                
                if (searchCache.has(query)) {
                    showDropdown(searchCache.get(query));
                    return;
                }

                const response = await fetch(`/api/search_animes?q=${encodeURIComponent(query)}`);
                const animes = await response.json();

             
                searchCache.set(query, animes);

               
                if (searchCache.size > 50) {
                    const firstKey = searchCache.keys().next().value;
                    searchCache.delete(firstKey);
                }

                showDropdown(animes);
            } catch (error) {
                console.error('Error searching animes:', error);
            }
        }

       
        function showDropdown(animes) {
            const dropdown = document.getElementById('animeDropdown');

           
            requestAnimationFrame(() => {
                dropdown.innerHTML = '';
                dropdown.style.display = 'block';

                
                const fragment = document.createDocumentFragment();

                if (animes.length === 0) {
                    const option = document.createElement('div');
                    option.className = 'anime-option';
                    option.textContent = 'No animes found';
                    fragment.appendChild(option);
                } else {
                    animes.slice(0, 30).forEach(anime => {
                        const option = document.createElement('div');
                        option.className = 'anime-option';
                        option.textContent = anime[1];
                        option.onclick = () => addToFavorites(anime[0], anime[1]);
                        fragment.appendChild(option);
                    });
                }

                dropdown.appendChild(fragment);

             
                setTimeout(() => {
                    dropdown.classList.add('show');
                }, 50);
            });
        }

        function hideDropdown() {
            const dropdown = document.getElementById('animeDropdown');
            dropdown.classList.remove('show');

            setTimeout(() => {
                dropdown.style.display = 'none';
                dropdown.innerHTML = '';
            }, 300);
        }

        async function addToFavorites(animeId, animeName) {
            try {
                const response = await fetch('/api/add_favorite', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        anime_id: animeId,
                        anime_name: animeName
                    })
                });

                const result = await response.json();

                if (result.success) {
                    loadFavorites();
                    document.getElementById('animeSearch').value = '';
                    hideDropdown();
                } else {
                    showMessage("Error", 'error');
                }
            } catch (error) {
                console.error('Error adding favorite:', error);
                showMessage('Error adding favorite', 'error');
            }
        }

        async function removeFromFavorites(animeId) {
            try {
                const response = await fetch('/api/remove_favorite', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        anime_id: animeId,

                    })
                });

                const result = await response.json();

                if (result.success) {
                    loadFavorites();
                } else {
                    showMessage("Error", 'error');
                }
            } catch (error) {
                console.error('Error removing favorite:', error);
            }
        }

        async function removeFromFavorites(animeId) {
            try {
                const response = await fetch('/api/remove_favorite', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        anime_id: animeId,

                    })
                });

                const result = await response.json();

                if (result.success) {
                    loadFavorites();
                } else {
                    showMessage("Error", 'error');
                }
            } catch (error) {
                console.error('Error removing favorite:', error);
            }
        }

        async function loadFavorites() {
            try {
                const response = await fetch('/api/get_favorites');
                const favorites = await response.json();
                displayFavorites(favorites);
            } catch (error) {
                console.error('Error loading favorites:', error);
            }
        }

        function displayFavorites(favorites) {
            const favoritesList = document.getElementById('favoritesList');

            requestAnimationFrame(() => {
                if (favorites.length === 0) {
                    favoritesList.innerHTML = '<div class="empty-state">No favorites selected yet. Search and add some animes!</div>';
                } else {
                    const fragment = document.createDocumentFragment();
                    favorites.forEach(anime => {
                        const div = document.createElement('div');
                        div.className = 'favorite-tag';
                        div.innerHTML = `
                            <span>${anime.name}</span>
                            <button class="remove-btn" onclick="removeFromFavorites(${anime.id})">×</button>
                        `;
                        fragment.appendChild(div);
                    });
                    favoritesList.innerHTML = '';
                    favoritesList.appendChild(fragment);
                }
            });
        }

        async function clearFavorites() {
            try {
                const response = await fetch('/api/clear_favorites', {
                    method: 'POST'
                });

                const result = await response.json();

                if (result.success) {
                    loadFavorites();
                }
            } catch (error) {
                console.error('Error clearing favorites:', error);
            }
        }

        async function getRecommendations() {
    const recommendationsSection = document.getElementById('recommendationsSection');
    const favoritesSection = document.getElementById('favoritesSection');
    const backButton = document.querySelector('.back-button');
    const recommendationsGrid = document.getElementById('recommendationsGrid');

    recommendationsGrid.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <div>Getting your recommendations...</div>
        </div>
    `;

    favoritesSection.style.display = 'none';
    recommendationsSection.style.display = 'block';
    backButton.style.display = 'block';

    try {
        const response = await fetch('/api/get_recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filters: currentFilters,
                blacklisted_animes: blacklistedAnimes.map(anime => anime.id) 
            })
        });

        const result = await response.json();

        if (result.success) {
            displayRecommendations(result.recommendations);
        } else {
            recommendationsGrid.innerHTML = `
                <div class="loading">
                    <div style="color: #ff6b6b;">${result.message}</div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        recommendationsGrid.innerHTML = `
            <div class="loading">
                <div style="color: #ff6b6b;">Error getting recommendations</div>
            </div>
        `;
    }
}

        function displayRecommendations(recommendations) {
    const recommendationsGrid = document.getElementById('recommendationsGrid');

    if (recommendations.length === 0) {
        recommendationsGrid.innerHTML = '<div class="loading">No recommendations found</div>';
        return;
    }

    requestAnimationFrame(() => {
        const fragment = document.createDocumentFragment();

        recommendations.forEach((anime, index) => {
            const imageElement = anime.image_url
                ? `<img src="${anime.image_url}" alt="${anime.name}" class="anime-image" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">`
                : '';

            const placeholderElement = anime.image_url
                ? `<div class="no-image-placeholder" style="display: none;">🎌</div>`
                : `<div class="no-image-placeholder">🎌</div>`;

            const malButton = anime.mal_url
                ? `<a href="${anime.mal_url}" target="_blank" class="mal-button">
                    ${malLogoUrl ? `<img src="${malLogoUrl}" alt="MAL" class="mal-logo">` : '🔗'}
                    View on MyAnimeList
                   </a>`
                : `<div class="mal-message">MyAnimeList link not available</div>`;

            const genresElementFront = anime.genres && anime.genres.length > 0
                ? `<div class="genres-container-front">
                    ${anime.genres.slice(0, 3).map(genre => `<span class="genre-tag-front">${genre}</span>`).join('')}
                   </div>`
                : '';

            const div = document.createElement('div');
            div.className = 'anime-card-wrapper';
            div.style.animationDelay = `${index * 0.1}s`;
            div.setAttribute('data-anime-id', anime.id);
            div.innerHTML = `
                <button class="delete-button" onclick="deleteAnimeCard(event, ${anime.id})" title="Remove from recommendations">
                    🗑️
                </button>
                <div class="anime-rank"></div>
                <div class="anime-card" onclick="flipCard(this)">
                    <div class="card-inner">
                        <div class="card-front">
                            <div class="anime-image-container">
                                ${imageElement}
                                ${placeholderElement}
                            </div>
                            <div class="anime-info">
                                <div class="anime-name">${anime.name}</div>
                                <div class="anime-bottom-section">
                                    ${genresElementFront}
                                    <button class="already-watched-btn" onclick="addToFavoritesFromRecommendations(event, ${anime.id}, '${anime.name.replace(/'/g, "\\'")}')">
                                        Already Watched & Liked
                                    </button>
                                </div>
                                <div class="flip-hint">Click card to see MyAnimeList link →</div>
                            </div>
                        </div>
                        <div class="card-back">
                            <div class="mal-button-container">
                                ${malButton}
                            </div>

                        </div>
                    </div>
                </div>
            `;
            fragment.appendChild(div);
        });

        recommendationsGrid.innerHTML = '';
        recommendationsGrid.appendChild(fragment);
    });
}


        function flipCard(card) {
            const cardInner = card.querySelector('.card-inner');
            const scoreElement = card.querySelector('.anime-score');
            const buttonElement = card.querySelector('.already-watched-btn');

            if (cardInner.classList.contains('flipped')) {
                cardInner.classList.remove('flipped');
                if (scoreElement) scoreElement.classList.add('hidden');
                if (buttonElement) buttonElement.classList.add('hidden');

                setTimeout(() => {
                    if (scoreElement) scoreElement.classList.remove('hidden');
                    if (buttonElement) buttonElement.classList.remove('hidden');
                }, 300);
            } else {
                cardInner.classList.add('flipped');
            }

            const malButton = card.querySelector('.mal-button');
            if (malButton) {
                malButton.addEventListener('click', function(e) {
                    e.stopPropagation();
                });
            }
        }

        function showFavorites() {
            document.getElementById('favoritesSection').style.display = 'block';
            document.getElementById('recommendationsSection').style.display = 'none';
            document.querySelector('.back-button').style.display = 'none';
            window.scrollTo(0, 0);
        }

        function showMessage(message, type) {
            alert(message);
        }

        async function addToFavoritesFromRecommendations(event, animeId, animeName) {
            event.stopPropagation();

            try {
                const response = await fetch('/api/add_favorite', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        anime_id: animeId,
                        anime_name: animeName
                    })
                });

                const result = await response.json();

                if (result.success) {
                    showFavorites();
                    loadFavorites();
                    getRecommendations();
                } else {
                    showMessage("This anime is already in your favorites!", 'info');
                }
            } catch (error) {
                console.error('Error adding favorite:', error);
                showMessage('Error adding to favorites', 'error');
            }
        }

        let currentFilters = {
            show_sequels: false,
            show_movies: true,
            show_tv: true,
            show_ova: false,
        };

        function toggleFilter(button) {
            const filterName = button.getAttribute('data-filter');
            const isActive = button.classList.contains('active');

            if (isActive) {
                button.classList.remove('active');
                currentFilters[filterName] = false;
            } else {
                button.classList.add('active');
                currentFilters[filterName] = true;
            }
        }

        function applyFilters() {
    getRecommendations(); 
}

        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                document.body.style.animationPlayState = 'paused';
            } else {
                document.body.style.animationPlayState = 'running';
            }
        });
       function deleteAnimeCard(event, animeId) {
    event.stopPropagation();

    const cardWrapper = event.target.closest('.anime-card-wrapper');
    const animeName = cardWrapper.querySelector('.anime-name').textContent;

    addToBlacklist(animeId, animeName);

    cardWrapper.style.transform = 'scale(0.8)';
    cardWrapper.style.opacity = '0';
    cardWrapper.style.transition = 'all 0.3s ease';

    setTimeout(() => {
        cardWrapper.remove();

        const remainingCards = document.querySelectorAll('.anime-card-wrapper');
        remainingCards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
        });

        getRecommendations();
    }, 300);
}
        let blacklistedAnimes = [];

function loadBlacklist() {
    const saved = localStorage.getItem('blacklistedAnimes');
    if (saved) {
        blacklistedAnimes = JSON.parse(saved);
    }
    updateBlacklistCounter();
}

function saveBlacklist() {
    localStorage.setItem('blacklistedAnimes', JSON.stringify(blacklistedAnimes));
    updateBlacklistCounter();
}

function updateBlacklistCounter() {
    const counter = document.getElementById('blacklistCounter');
    const button = document.getElementById('blacklistButton');

    if (blacklistedAnimes.length > 0) {
        counter.textContent = blacklistedAnimes.length;
        button.classList.add('show');
    } else {
        counter.textContent = '0';
        button.classList.remove('show');
    }
}

function addToBlacklist(animeId, animeName) {
    const existingIndex = blacklistedAnimes.findIndex(anime => anime.id === animeId);
    if (existingIndex === -1) {
        blacklistedAnimes.push({ id: animeId, name: animeName });
        saveBlacklist();
    }
}
function displayBlacklistItems() {
    const blacklistList = document.getElementById('blacklistList');

    if (blacklistedAnimes.length === 0) {
        blacklistList.innerHTML = '<div class="empty-blacklist">No animes in blacklist</div>';
        return;
    }

    const fragment = document.createDocumentFragment();
    blacklistedAnimes.forEach(anime => {
        const div = document.createElement('div');
        div.className = 'blacklist-item';
        div.innerHTML = `
            <span class="blacklist-name">${anime.name}</span>
            <button class="remove-blacklist-btn" onclick="removeFromBlacklist(${anime.id})">Remove</button>
        `;
        fragment.appendChild(div);
    });

    blacklistList.innerHTML = '';
    blacklistList.appendChild(fragment);
}
function removeFromBlacklist(animeId) {
    blacklistedAnimes = blacklistedAnimes.filter(anime => anime.id !== animeId);
    saveBlacklist();
    displayBlacklistItems();
}
function showCustomConfirm(title, message, onConfirm) {
    const modal = document.createElement('div');
    modal.className = 'custom-confirm-modal';
    modal.innerHTML = `
        <div class="custom-confirm-content">
            <div class="confirm-header">
                <h3>${title}</h3>
            </div>
            <div class="confirm-message">
                <p>${message}</p>
            </div>
            <div class="confirm-actions">
                <button class="confirm-btn cancel" onclick="closeCustomConfirm()">Cancel</button>
                <button class="confirm-btn confirm" onclick="confirmAction()">Yes, Clear All</button>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    window.confirmAction = () => {
        onConfirm();
        closeCustomConfirm();
    };

    window.closeCustomConfirm = () => {
        modal.remove();
        delete window.confirmAction;
        delete window.closeCustomConfirm;
    };

    setTimeout(() => modal.classList.add('show'), 50);
}
function clearBlacklist() {
    showCustomConfirm(
        'Clear All Blacklist',
        'Are you sure you want to clear all blacklisted animes?',
        () => {
            blacklistedAnimes = [];
            saveBlacklist();
            displayBlacklistItems();
        }
    );
}

function openBlacklistModal() {
    displayBlacklistItems();
    document.getElementById('blacklistModal').classList.add('show');
}

function closeBlacklistModal() {
    document.getElementById('blacklistModal').classList.remove('show');
}
    </script>
</body>
</html>
