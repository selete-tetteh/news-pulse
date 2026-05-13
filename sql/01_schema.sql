-- =============================================================
-- News Pulse — Warehouse Schema
-- Database: news_pulse
-- Created: 2026-05-13
-- Description: Star schema for the News Pulse NLP pipeline.
--              Raw GDELT data is ingested by src/pipeline/ingest.py
--              and loaded into this schema. All analysis queries
--              the warehouse directly — raw files are never read
--              by notebooks or the dashboard.
-- =============================================================

CREATE DATABASE IF NOT EXISTS news_pulse;
USE news_pulse;

-- -------------------------------------------------------------
-- Dimension Tables
-- These store descriptive context. Values here are stable and
-- reused across many articles, so we store them once and
-- reference them by ID in the fact table.
-- -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_source (
    source_id     INT AUTO_INCREMENT PRIMARY KEY,
    source_name   VARCHAR(255) NOT NULL,
    country       VARCHAR(100),
    language      VARCHAR(50),
    UNIQUE KEY uq_source_name (source_name)
);

-- Stores pre-computed calendar attributes so analytical queries
-- can filter and group by time without running date functions
-- on every row.
CREATE TABLE IF NOT EXISTS dim_date (
    date_id       INT PRIMARY KEY,   -- stored as YYYYMMDD integer for fast joins
    full_date     DATE NOT NULL,
    year          SMALLINT NOT NULL,
    quarter       TINYINT NOT NULL,
    month         TINYINT NOT NULL,
    month_name    VARCHAR(10) NOT NULL,
    week          TINYINT NOT NULL,
    day_of_month  TINYINT NOT NULL,
    day_of_week   TINYINT NOT NULL,  -- 1 = Monday, 7 = Sunday
    day_name      VARCHAR(10) NOT NULL,
    is_weekend    BOOLEAN NOT NULL
);

-- The seven topic segments. GDELT theme tags used for routing
-- are stored here so the mapping is documented in the warehouse,
-- not buried in pipeline code.
CREATE TABLE IF NOT EXISTS dim_segment (
    segment_id    INT AUTO_INCREMENT PRIMARY KEY,
    segment_name  VARCHAR(100) NOT NULL,
    gdelt_themes  TEXT,              -- comma-separated GDELT theme tags that map to this segment
    UNIQUE KEY uq_segment_name (segment_name)
);

-- Named entities extracted by spaCy. Entity type follows spaCy
-- conventions: PERSON, ORG, GPE (geopolitical entity), LOC.
CREATE TABLE IF NOT EXISTS dim_entity (
    entity_id     INT AUTO_INCREMENT PRIMARY KEY,
    entity_name   VARCHAR(255) NOT NULL,
    entity_type   VARCHAR(20) NOT NULL,   -- PERSON | ORG | GPE | LOC
    UNIQUE KEY uq_entity (entity_name, entity_type)
);

-- -------------------------------------------------------------
-- Fact Tables
-- These store the events — one row per article, one row per
-- entity mention. They are append-only; rows are never updated.
-- -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_articles (
    article_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    source_id         INT NOT NULL,
    date_id           INT NOT NULL,
    segment_id        INT NOT NULL,
    url               TEXT NOT NULL,
    title             TEXT,
    seendate          DATETIME,
    sentiment_score   FLOAT,             -- VADER compound score: -1.0 to 1.0
    sentiment_label   VARCHAR(10),       -- POSITIVE | NEGATIVE | NEUTRAL
    language          VARCHAR(10),
    ingested_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id)  REFERENCES dim_source(source_id),
    FOREIGN KEY (date_id)    REFERENCES dim_date(date_id),
    FOREIGN KEY (segment_id) REFERENCES dim_segment(segment_id),
    INDEX idx_date_id    (date_id),
    INDEX idx_source_id  (source_id),
    INDEX idx_segment_id (segment_id)
);

-- Junction table for the many-to-many relationship between
-- articles and entities. One article mentions many entities;
-- one entity appears across many articles.
-- This structure is what enables the co-occurrence network —
-- two entities co-occur when they share an article_id.
CREATE TABLE IF NOT EXISTS fact_entity_mentions (
    mention_id    BIGINT AUTO_INCREMENT PRIMARY KEY,
    article_id    BIGINT NOT NULL,
    entity_id     INT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES fact_articles(article_id),
    FOREIGN KEY (entity_id)  REFERENCES dim_entity(entity_id),
    INDEX idx_article_id (article_id),
    INDEX idx_entity_id  (entity_id)
);

-- -------------------------------------------------------------
-- Seed Data — Segments
-- Loaded once at setup. These do not change unless the project
-- scope changes.
-- -------------------------------------------------------------

INSERT IGNORE INTO dim_segment (segment_name, gdelt_themes) VALUES
    ('Politics and Government', 'POLITICS,GOV,ELECTIONS,LEADER'),
    ('Business and Markets',    'ECON,BUSINESS,MARKETS,FINANCE,TRADE'),
    ('Technology',              'TECH,CYBER,AI,INTERNET,INNOVATION'),
    ('Sports',                  'SPORTS,SPORT'),
    ('Entertainment and Culture','ENTERTAIN,CULTURE,ARTS,MEDIA'),
    ('Science and Health',      'HEALTH,SCIENCE,MEDICAL,ENV'),
    ('Crime and Justice',       'CRIME,LEGAL,JUSTICE,LAW'),
    ('General',                 '');