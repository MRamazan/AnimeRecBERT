# AnimeRecBERT: BERT-Based Anime Recommendation System

**AnimeRecBERT** is a personalized anime recommendation system based on BERT transformer architecture. Inspired from [https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch), this project introduces customizations tailored for an anime recommendation system and inference.

- 🕒 **No positional encoding**, since the dataset contains no temporal information, i removed positional encoding and it performed better.
- 🎌 **Anime-specific user-item dataset**
- 🖥️ **GUI interface** for real-time recommendations

This project provides a solid foundation for further development in personalized anime recommendation using transformer-based models.

<div style="display: flex; align-items: flex-start; gap: 30px;">
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Recall@1</td><td>0.507</td></tr>
      <tr><td>Recall@5</td><td>0.841</td></tr>
      <tr><td>Recall@10</td><td>0.919</td></tr>
      <tr><td>Recall@20</td><td>0.967</td></tr>
      <tr><td>Recall@50</td><td>0.994</td></tr>
      <tr><td>Recall@100</td><td>0.9998</td></tr>
      <tr><td>NDCG@1</td><td>0.507</td></tr>
      <tr><td>NDCG@5</td><td>0.689</td></tr>
      <tr><td>NDCG@10</td><td>0.715</td></tr>
      <tr><td>NDCG@20</td><td>0.727</td></tr>
      <tr><td>NDCG@50</td><td>0.732</td></tr>
      <tr><td>NDCG@100</td><td>0.733</td></tr>
    </tbody>
  </table>
  <img src="bertrec_metrics_table.png" alt="Metrics Chart" style="max-width: 400px;">
</div>

## Setup & Usage

### Download Dataset & Pretrained Model

```bash
curl -L -o Data/AnimeRatings54M/animeratings-mini-54m.zip \
     https://www.kaggle.com/api/v1/datasets/download/tavuksuzdurum/animeratings-mini-54m

unzip Data/AnimeRatings54M/animeratings-mini-54m.zip -d Data/AnimeRatings54M/
```

### Install Requirements

```bash
pip install requirements.txt
```
