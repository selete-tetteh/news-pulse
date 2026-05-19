# News Pulse

A live NLP pipeline that ingests real news from GDELT, stores it in a MySQL warehouse, and surfaces relevant content through two recommendation approaches: TF-IDF cosine similarity as a baseline and sentence transformers for semantic matching. The two are compared directly so the improvement is quantified, not claimed.

Built as a portfolio project to demonstrate end-to-end data engineering and NLP: automated ingestion, warehouse design, entity extraction, sentiment analysis, and a working Streamlit dashboard.

---

## What It Does

GDELT updates every 15 minutes and covers news across 100+ languages. This project pulls from it on a schedule, classifies articles into seven topic segments, and runs NLP analysis across the full dataset.

The pipeline is live. Clone the repo, set up the environment, and it works.

**Seven segments:**
- Politics and Government
- Business and Markets
- Technology
- Sports
- Entertainment and Culture
- Science and Health
- Crime and Justice

Segments are assigned using GDELT's built-in theme tags rather than a custom classifier. This keeps the pipeline reproducible without proprietary labels.

---

## Architecture

```
GDELT API
    |
    v
src/pipeline/ingest.py        -- pulls and cleans raw GDELT data on a schedule
    |
    v
src/pipeline/transform.py     -- segment routing, sentiment scoring, entity extraction
    |
    v
src/pipeline/load.py          -- writes to MySQL warehouse
    |
    v
MySQL warehouse               -- star schema: fact_articles + dimension tables
    |
    v
src/pipeline/recommend.py     -- builds recommendation index from warehouse
    |
    v
src/dashboard/app.py          -- Streamlit dashboard
```

Raw data never touches the notebooks directly. Everything goes through the warehouse first. The analysis layer always works with clean, structured data. The pipeline can be updated without touching the notebooks.

---

## Notebooks

| Notebook | What it covers |
|----------|----------------|
| `01_pipeline_development.ipynb` | GDELT connection, ingestion logic, data cleaning — developed here before moving to production scripts |
| `02_news_landscape_analysis.ipynb` | Segment-level analysis: sentiment trends, entity extraction, source diversity, publication velocity |
| `03_recommendation_engine.ipynb` | TF-IDF baseline, sentence transformer build, side-by-side evaluation with precision and recall metrics |

---

## Recommendation Engine

Two approaches are built and compared.

**TF-IDF + cosine similarity** is the baseline. It matches articles by word overlap, which is fast and interpretable but misses synonyms and related concepts that don't share vocabulary.

**Sentence transformers** encode articles as dense vectors that capture meaning rather than words. "Central bank raises interest rates" and "Fed hikes borrowing costs" score as similar even though they share almost no vocabulary.

The comparison uses precision at 5 (P@5) across all seven segments and a confusion matrix. Sentence transformers achieved an overall P@5 of 0.878 versus TF-IDF's 0.657, a 22 percentage point gap that held consistent across segments. Full results are in `notebooks/03_recommendation_engine.ipynb`.

### Feature Signal

The recommendation engine does not use full article text. GDELT does not provide it, only metadata, theme tags, entity names, and occasional quotation fragments. The `Quotations` field was tested but found at 15% fill rate with inconsistent formatting, making it unreliable as a signal across the 11.9M article corpus.

Instead, each article is represented by a synthetic feature string: source name, segment label, and extracted entity names concatenated into one string. This is uniform across 100% of articles including all historical rows, so the index is consistent.

### Embedding Cache

The sentence transformer index is built from a stratified sample of 90,000 articles (15,000 per segment) and cached locally as `data/processed/embeddings.npz` alongside `data/processed/embeddings_meta.csv`.

The cache is excluded from Git. To rebuild it on a fresh clone or after a large data update:

```python
from src.pipeline.recommend import Recommender
r = Recommender(rebuild_cache=True)
```

This fetches articles from the warehouse, generates embeddings using `all-MiniLM-L6-v2`, and writes the cache to disk. On a standard machine it takes roughly 10 minutes for 90,000 articles. Subsequent loads use the cache and start in under 10 seconds.

If you are running without internet access, add the following to your `.env` before starting:

```
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

The model files must already be in the Hugging Face cache on your machine for offline mode to work.

---

## NLP Components

**Named entity recognition**: spaCy `en_core_web_sm` extracts people and organisations from article metadata. Entities are stored in the warehouse and used to build a co-occurrence network: which people and organisations appear in the same articles, and how often.

**Sentiment analysis**: VADER applied at the article level. Scores are tracked over time by segment so you can see whether coverage of a topic is trending positive or negative across sources.

Note: GDELT does not provide article titles or body text, only the source name. VADER on source names returns near-zero variance because outlet names carry no sentiment signal. `reports/02_sentiment_by_segment.png` documents this — all segments cluster around 0.0. The chart stays in the repo as a record of what was tested. A real sentiment trend analysis needs full article text or headlines from a different data source. The schema, scoring logic, and time-series query all work. The input signal is the problem.

**Source diversity**: measures how many distinct outlets are covering each topic cluster. A topic covered by three outlets is a different signal than one covered by forty.

**Publication velocity**: how quickly a story propagates after it first appears. Used to identify fast-breaking news versus slow-build topics.

---

## Warehouse Schema

Star schema in MySQL. One fact table, four dimension tables.

```
fact_articles
    article_id (PK)
    source_id (FK -> dim_source)
    date_id (FK -> dim_date)
    segment_id (FK -> dim_segment)
    url
    seendate
    sentiment_score
    sentiment_label
    language

dim_source            -- outlet name
dim_date              -- year, month, week, day_of_week
dim_segment           -- segment name
dim_entity            -- entity name, type (PERSON or ORG)
fact_entity_mentions  -- junction table linking articles to entities
```

Note: `fact_entity_mentions` currently holds 64M+ rows. Queries against it must be scoped — unscoped joins time out even with indexes present. The `recommend.py` implementation fetches entities in chunks of 1,000 article IDs for this reason.

---

## Dashboard

Built in Streamlit. Four views:

- **Segment overview** — top stories per segment using the recommendation engine's `recommend_by_segment()` method
- **Recommendations** — select any article, get the top 5 semantically similar results
- **Sentiment trends** — sentiment by segment over a selected date range
- **Entity network** — co-occurrence graph showing which people and organisations appear together most often

---

## Setup

**Requirements:** Python 3.11, MySQL 8.0, conda

```bash
# Clone the repo
git clone https://github.com/selete-tetteh/news-pulse
cd news-pulse

# Create and activate the environment
conda env create -f environment.yml
conda activate news-pulse

# Download the spaCy language model
python -m spacy download en_core_web_sm

# Set up credentials
cp .env.example .env
# Edit .env and add your MySQL password, and optionally Telegram credentials

# Create the MySQL database and schema
mysql -u root -p < sql/01_schema.sql

# Populate the date dimension (run once)
python -m src.pipeline.populate_dim_date

# Run the pipeline for the first time
python -m src.pipeline.run_pipeline --mode live

# Launch the dashboard
streamlit run src/dashboard/app.py
```

To start the live scheduler (runs every 15 minutes automatically):

```bash
python -m src.pipeline.scheduler
```

To stop it:

```bash
bash scripts/kill_scheduler.sh
```

---

## Running the Tests

```bash
pytest tests/
```

All 22 tests must pass before any pipeline commit.

---

## Repository Structure

```
news-pulse/
├── data/
│   ├── raw/              # Raw GDELT pulls -- excluded from Git
│   └── processed/        # Embedding cache and processed outputs -- excluded from Git
├── launchd/
│   └── com.newspulse.pipeline.plist   # macOS launchd config for scheduled runs
├── notebooks/
│   ├── 01_pipeline_development.ipynb
│   ├── 02_news_landscape_analysis.ipynb
│   └── 03_recommendation_engine.ipynb
├── reports/              # Charts from notebook analysis -- committed to Git
├── scripts/
│   └── kill_scheduler.sh
├── sql/
│   └── 01_schema.sql     # Full warehouse schema DDL
├── src/
│   ├── pipeline/
│   │   ├── ingest.py           # GDELT ingestion and cleaning
│   │   ├── transform.py        # Segment routing, sentiment, entity extraction
│   │   ├── load.py             # Warehouse loading with batched commits
│   │   ├── recommend.py        # Recommendation engine (TF-IDF and sentence transformers)
│   │   ├── run_pipeline.py     # Orchestrator: ingest -> transform -> load
│   │   ├── scheduler.py        # APScheduler wrapper for live runs
│   │   ├── notify.py           # Telegram alerts after each run
│   │   └── populate_dim_date.py # One-time date dimension setup
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
├── tests/
│   └── test_transform.py       # 22 unit tests for the transform module
├── environment.yml
├── README.md
└── .gitignore
```

---

## Data Source

[GDELT Project](https://www.gdeltproject.org/) — free, no rate limits, updated every 15 minutes. No API key required.

---

## Tools

| Layer | Tool |
|-------|------|
| Ingestion and pipeline | Python |
| Warehouse | MySQL |
| NLP | spaCy, sentence-transformers, VADER |
| Analysis | pandas, scikit-learn, scipy |
| Visualisation | plotly, matplotlib |
| Dashboard | Streamlit |
| Scheduling | APScheduler |
| Testing | pytest |
