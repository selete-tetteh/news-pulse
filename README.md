# News Pulse

A live NLP pipeline that ingests real news from GDELT, stores it in a MySQL warehouse, and surfaces relevant content through two recommendation approaches, TF-IDF cosine similarity as a baseline, and sentence transformers for semantic matching. The two are compared directly so the improvement is quantified, not just claimed.

Built as a portfolio project to demonstrate end-to-end data engineering and NLP skills: automated ingestion, warehouse design, entity extraction, sentiment analysis, and a working Streamlit dashboard.

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

Segments are assigned using GDELT's built-in theme tags rather than a custom classifier. This is a deliberate choice so it avoids training a classifier on a manually labelled dataset and keeps the pipeline reproducible without proprietary labels.

---

## Architecture

```
GDELT API
    |
    v
src/pipeline/ingest.py        -- pulls and cleans raw GDELT data on a schedule
    |
    v
MySQL warehouse               -- star schema: fact_articles + dimension tables
    |
    v
notebooks/                    -- analysis and model development
    |
    v
src/dashboard/app.py          -- Streamlit dashboard consuming the warehouse
```

Raw data never touches the notebooks directly. Everything goes through the warehouse first. This separation means the analysis layer is always working with clean, structured data, and the pipeline can be updated independently.

---

## Notebooks

| Notebook | What it covers |
|----------|---------------|
| `01_pipeline_development.ipynb` | GDELT connection, ingestion logic, data cleaning — developed here before moving to production scripts |
| `02_news_landscape_analysis.ipynb` | Segment-level analysis: sentiment trends, entity extraction, source diversity, publication velocity |
| `03_recommendation_engine.ipynb` | TF-IDF baseline, sentence transformer build, side-by-side evaluation with precision and recall metrics |

---

## Recommendation Engine

Two approaches are built and compared.

**TF-IDF + cosine similarity** — the standard baseline. Matches articles by word overlap. Fast and interpretable, but misses synonyms and related concepts that don't share vocabulary.

**Sentence transformers** — encodes articles as dense vectors that capture meaning, not just words. "Central bank raises interest rates" and "Fed hikes borrowing costs" will score as similar even though they share almost no vocabulary.

The comparison is quantified using precision, recall, and a user simulation. The point is not to say sentence transformers are better, it is to show exactly how much better and under what conditions.

---

## NLP Components

**Named entity recognition** — spaCy `en_core_web_sm` extracts people, organisations, and locations from article text. Entities are stored in the warehouse and used to build a co-occurrence network: which people and organisations appear in the same articles, and how often.

**Sentiment analysis** — VADER applied at the article level. Scores are tracked over time by segment so you can see whether coverage of a topic is trending more negative or positive across sources.

**Source diversity** — measures how many distinct outlets are covering each topic cluster. A topic covered by three outlets is a different signal than one covered by forty.

**Publication velocity** — how quickly a story propagates after it first appears. Used to identify fast-breaking news versus slow-build topics.

---

## Warehouse Schema

Star schema in MySQL. One fact table, four dimension tables.

```
fact_articles
    article_id (PK)
    source_id (FK)
    date_id (FK)
    segment_id (FK)
    url
    title
    sentiment_score
    sentiment_label

dim_source       -- outlet name, country, language
dim_date         -- year, month, week, day_of_week
dim_segment      -- segment name and GDELT theme tag
dim_entity       -- entity name, type (PERSON / ORG / GPE)
fact_entity_mentions  -- junction table linking articles to entities
```

---

## Dashboard

Built in Streamlit. Four views:

- **Overview** — current top topic clusters by segment, article volume over time
- **Recommendations** — select any article, get ten recommendations from both engines side by side
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
# Edit .env and add your MySQL password

# Create the MySQL database and schema
mysql -u root -p < sql/01_schema.sql

# Run the pipeline (first ingestion)
python src/pipeline/ingest.py

# Launch the dashboard
streamlit run src/dashboard/app.py
```

---

## Repository Structure

```
news-pulse/
├── data/
│   ├── raw/              # Raw GDELT pulls — excluded from Git
│   └── processed/        # Cleaned and transformed data — excluded from Git
├── sql/
│   ├── 01_schema.sql     # Warehouse schema DDL
│   └── 02_queries.sql    # Analytical queries used in notebooks
├── notebooks/
│   ├── 01_pipeline_development.ipynb
│   ├── 02_news_landscape_analysis.ipynb
│   └── 03_recommendation_engine.ipynb
├── src/
│   ├── pipeline/
│   │   ├── ingest.py     # GDELT ingestion and cleaning
│   │   ├── transform.py  # NLP processing and entity extraction
│   │   └── load.py       # Warehouse loading
│   └── dashboard/
│       └── app.py        # Streamlit dashboard
├── reports/              # Charts and Excel output — excluded from Git
├── tests/                # Unit tests
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
