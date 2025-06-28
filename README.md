# AnimeRecBERT: BERT-Based Anime Recommendation System

**AnimeRecBERT** is a personalized anime recommendation system based on BERT transformer architecture. Inspired from [https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch), this project introduces customizations tailored for an anime recommendation system and inference.

- 🕒 **No positional encoding**, since the dataset contains no temporal information, i removed positional encoding and it performed better.
- 🎌 **Anime-specific user-item dataset**
- 🖥️ **GUI interface** for real-time recommendations

This project provides a solid foundation for further development in personalized anime recommendation using transformer-based models.

## Metrics
The model trained on a large-scale dataset with 560,000 users and 54 million ratings. Below are the Top-K recommendation metrics:

<table>
<tr>
<td>

| Metric       | Value    |
|--------------|----------|
| Recall@100   | 0.9998   |
| NDCG@100     | 0.733    |
| Recall@50    | 0.994    |
| NDCG@50      | 0.732    |
| Recall@20    | 0.967    |
| NDCG@20      | 0.727    |
| Recall@10    | 0.919    |
| NDCG@10      | 0.715    |
| Recall@5     | 0.841    |
| NDCG@5       | 0.689    |
| Recall@1     | 0.507    |
| NDCG@1       | 0.507    |

</td>
<td>

<img src="bertrec_metrics_table.png" alt="Metrics Visualization" width="700">

</td>
</tr>
</table>

## Setup & Usage

### Download Dataset & Pretrained Model

```bash
curl -L -o Data/AnimeRatings54M/animeratings-mini-54m.zip \
     https://www.kaggle.com/api/v1/datasets/download/tavuksuzdurum/animeratings-mini-54m

unzip Data/AnimeRatings54M/animeratings-mini-54m.zip -d Data/AnimeRatings54M/
```

### Install Requirements
Install PyTorch from https://pytorch.org/get-started/locally/
```bash
pip install requirements.txt
```

### All Done, Start GUI
Some model parameters depend on dataloader statistics.
Instead of setting these parameters as constants, the code processes the data as in training, but will only use mappings.
This way, changes in the original data won't cause an error.
```bash
python inference.py   -c Data/AnimeRatings54M/pretrained_bert.pth /
                      -d Data/preprocessed/AnimeRatings54M_min_rating7-min_uc10-min_sc10-splitleave_one_out/dataset.pkl /
                      -a Data/animes.json /
                      --template train_bert
```
<img src="gui.png" alt="BERTRec GUI" width="900">

# Results
## 🌟 My Favorites

| #  | Anime Title                                                                |
|----|----------------------------------------------------------------------------|
| 1  | Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e                            |
| 2  | Giji Harem                                                                 |
| 3  | Ijiranaide, Nagatoro-san                                                   |
| 4  | 86 (Eighty-Six)                                                            |
| 5  | Mushoku Tensei: Isekai Ittara Honki Dasu                                   |
| 6  | Made in Abyss                                                              |
| 7  | Shangri-La Frontier: Kusoge Hunter, Kamige ni Idoman to su                 |
| 8  | Vanitas no Karte                                                           |
| 9  | Jigokuraku                                                                 |

## 🌟 BERT Recommendations Based on My Favorites
**FAVORITE POSITION WONT EFFECT RESULTS!**

| #  | Anime Title                                                                                          | Score     |
|----|------------------------------------------------------------------------------------------------------|-----------|
| 1  | Yofukashi no Uta (Call of the Night)                                                                 | 14.1491   |
| 2  | Tengoku Daimakyou (Heavenly Delusion)                                                                | 13.1946   |
| 3  | Summertime Render                                                                                    | 13.0349   |
| 4  | 86 Part 2 (Eighty-Six Part 2)                                                                        | 12.0690   |
| 5  | Chainsaw Man                                                                                         | 12.0244   |
| 6  | Kage no Jitsuryokusha ni Naritakute! 2nd Season (The Eminence in Shadow Season 2)                    | 11.9574   |
| 7  | Cyberpunk: Edgerunners                                                                               | 11.9093   |
| 8  | Dandadan                                                                                             | 11.8868   |
| 9  | Sousou no Frieren (Frieren: Beyond Journey's End)                                                    | 11.8604   |
|10  | Zom 100: Zombie ni Naru made ni Shitai 100 no Koto (Zom 100: Bucket List of the Dead)                | 11.8430   |
|11  | Tenki no Ko (Weathering With You)                                                                    | 11.8104   |
|12  | Make Heroine ga Oosugiru! (Too Many Losing Heroines!)                                                | 11.7766   |
|13  | Horimiya                                                                                             | 11.7558   |
|14  | Mushoku Tensei: Isekai Ittara Honki Dasu (Jobless Reincarnation)                                     | 11.7015   |
|15  | Natsu e no Tunnel, Sayonara no Deguchi (The Tunnel to Summer, the Exit of Goodbye)                   | 11.6777   |
|16  | Kage no Jitsuryokusha ni Naritakute! (The Eminence in Shadow)                                        | 11.6446   |
|17  | Yamada-kun to Lv999 no Koi wo Suru (Loving Yamada at Lv999)                                          | 11.6185   |
|18  | Boku no Kokoro no Yabai Yatsu (The Dangers in My Heart)                                              | 11.6104   |
|19  | Ore dake Level Up na Ken (Solo Leveling)                                                             | 11.5978   |
|20  | Mushoku Tensei II: Isekai Ittara Honki Dasu Part 2 (Jobless Reincarnation Season 2 Part 2)           | 11.5757   |



