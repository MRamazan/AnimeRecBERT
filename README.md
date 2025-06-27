# AnimeRecBERT: BERT-Based Anime Recommendation System

**AnimeRecBERT** is a personalized anime recommendation system based on BERT transformer architecture. Inspired from [](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch), this project introduces customizations tailored for an anime recommendation system:

- 🕒 **No positional encoding**, since the dataset contains no temporal information
- 🧠 **Model architecture adjustments** for improved performance on non-sequential preference data
- 🎌 **Anime-specific user-item dataset**
- 🖥️ **GUI interface** for real-time recommendations

This project provides a solid foundation for further development in personalized anime recommendation using transformer-based models.

| Metric      | Value   |
|-------------|---------|
| Recall@1    | 0.507   |
| Recall@10   | 0.919   |
| Recall@100  | 0.999   |
| NDCG@10     | 0.715   |
| NDCG@100    | 0.733   |
