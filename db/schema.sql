-- TVH Findability POC — Supabase (PostgreSQL) schema
-- Source of truth for structured product facts.
-- Pinecone only stores embeddings + a ref_no pointer back into products.

CREATE TABLE IF NOT EXISTS products (
    ref_no                  TEXT PRIMARY KEY,
    name                    TEXT,
    description             TEXT NOT NULL,
    category                TEXT,
    page_no                 INTEGER,
    catalogue               TEXT NOT NULL,
    attributes_json         JSONB DEFAULT '{}'::jsonb,
    image_url               TEXT,
    website_url             TEXT,
    catalog_page_image_url  TEXT,
    keywords                TEXT,
    created_at              TIMESTAMPTZ DEFAULT now(),
    updated_at              TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_products_catalogue ON products (catalogue);
CREATE INDEX IF NOT EXISTS idx_products_category  ON products (category);

CREATE TABLE IF NOT EXISTS frequently_bought_together (
    source_ref  TEXT NOT NULL REFERENCES products (ref_no),
    related_ref TEXT NOT NULL REFERENCES products (ref_no),
    score       NUMERIC NOT NULL,
    PRIMARY KEY (source_ref, related_ref)
);

CREATE INDEX IF NOT EXISTS idx_fbt_source ON frequently_bought_together (source_ref);

-- Every search is logged here. user_feedback closes the evaluation loop:
-- "correct" / "incorrect" ratings become the fine-tuning / regression set.
CREATE TABLE IF NOT EXISTS query_logs (
    id               BIGSERIAL PRIMARY KEY,
    raw_query        TEXT NOT NULL,
    parsed_intent    JSONB,
    matched_refs     TEXT[],
    recommended_refs TEXT[],
    user_feedback    TEXT,          -- 'correct' | 'incorrect' | null (no feedback yet)
    created_at       TIMESTAMPTZ DEFAULT now()
);
