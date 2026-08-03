---
name: db-schema
description: >-
  Enforce schema discipline: queries-first design, index-every-filter, safe migrations,
  verify-at-scale thinking. Prevents the common failure of designing schemas for demos
  that break in production. Use when creating tables, planning migrations, designing
  data models, or when user mentions database, schema, migration, indexing, multi-tenant.
---

## Overview

Enforce queries-first schema discipline — design every table around its access patterns, index every filter, and verify migrations are safe at production scale.

# Database Schema Design

## Persistence

ACTIVE on every schema change. Every CREATE TABLE, every ALTER, every migration. Think at scale before writing DDL.

## The Discipline

Before writing ANY schema:

```
1. QUERIES FIRST — what queries will run against this? (schema serves queries, not the other way around)
2. SCALE QUESTION — what happens at 10M rows? 100M? Does this still work?
3. INDEX PLAN — every WHERE, JOIN, ORDER BY gets an index. No exceptions.
4. MIGRATION SAFETY — can this ALTER run without locking the table for minutes?
5. VERIFY — explain analyze the critical queries. Prove they use indexes.
```

## Rules (never break)

- Never CREATE TABLE without knowing the top 5 queries it serves
- Never deploy a migration without testing against production-sized data
- Never use FLOAT for money (DECIMAL or integer cents)
- Never skip explicit `created_at`/`updated_at` on any table
- Never add a column with DEFAULT on a large table (locks it)
- Never DROP COLUMN in one step (stop writing → deploy → then drop)

Before creating any table:

- [ ] Define the access patterns FIRST (queries drive schema, not the other way around)
- [ ] Choose normalization level based on read/write ratio
- [ ] Plan for the 10x scale (what happens at 10M rows?)
- [ ] Identify foreign keys and cascade behavior
- [ ] Define indexes for every WHERE/JOIN/ORDER BY clause
- [ ] Plan soft-delete vs hard-delete strategy
- [ ] Consider multi-tenancy needs upfront

## Normalization Decision

| Situation | Strategy | Why |
|-----------|----------|-----|
| Read-heavy, rarely changes | Denormalize | Avoid JOINs at query time |
| Write-heavy, consistency critical | Normalize (3NF) | Single source of truth |
| Analytics/reporting | Star schema (denormalized) | Optimized for aggregation |
| User-facing CRUD | 3NF with strategic denorm | Balance of integrity + speed |

## Indexing Strategy

### Rules of Thumb

```sql
-- Index every column that appears in:
-- WHERE, JOIN ON, ORDER BY, GROUP BY

-- Composite index order matters: most selective first
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
-- This supports: WHERE user_id = ? AND status = ?
-- Also supports: WHERE user_id = ? (leftmost prefix)
-- Does NOT support: WHERE status = ? (not leftmost)

-- Covering index (includes all columns query needs)
CREATE INDEX idx_orders_covering ON orders(user_id, status) INCLUDE (total, created_at);
-- Query can be answered entirely from index (no table lookup)
```

### When NOT to Index

- Columns with < 10 distinct values (low cardinality) — except in composite indexes
- Tables with < 1000 rows (full scan is faster)
- Write-heavy tables where read is rare (indexes slow writes)
- Columns that change frequently (index maintenance cost)

## Migration Safety

### The Golden Rules

1. **Never DROP COLUMN in production without a 2-step process:**
   - Step 1: Stop writing to the column (deploy code change)
   - Step 2: Drop the column (next migration, after deploy confirmed)

2. **Never rename columns directly:**
   - Add new column → copy data → update code → drop old column

3. **Always make migrations reversible:**
   ```sql
   -- UP
   ALTER TABLE users ADD COLUMN display_name VARCHAR(255);
   
   -- DOWN (must always exist)
   ALTER TABLE users DROP COLUMN display_name;
   ```

4. **Lock-free schema changes for large tables:**
   ```sql
   -- BAD: locks table for minutes on 10M+ rows
   ALTER TABLE orders ADD COLUMN priority INT DEFAULT 0;
   
   -- GOOD: add nullable first, backfill in batches
   ALTER TABLE orders ADD COLUMN priority INT;
   -- Then backfill in 10K batches with sleep between
   ```

5. **Test migrations against production-sized data:**
   - Dump production schema (not data)
   - Run migration against it
   - Measure lock time and query impact

## Multi-Tenant Patterns

| Pattern | Isolation | Complexity | When to Use |
|---------|-----------|------------|-------------|
| **Shared table + tenant_id** | Logical | Low | < 1000 tenants, simple data |
| **Schema per tenant** | Strong | Medium | Enterprise, compliance needs |
| **Database per tenant** | Maximum | High | Regulated industries, massive scale |

### Shared Table (Most Common)

```sql
-- Every table has tenant_id
CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  name VARCHAR(255) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Row-level security (PostgreSQL)
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON projects
  USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- Composite index (tenant first!)
CREATE INDEX idx_projects_tenant ON projects(tenant_id, created_at DESC);
```

## Common Patterns

### Soft Delete

```sql
CREATE TABLE posts (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  deleted_at TIMESTAMPTZ,  -- NULL = active, timestamp = deleted
  -- ...
);

-- Default scope excludes deleted
CREATE VIEW active_posts AS SELECT * FROM posts WHERE deleted_at IS NULL;

-- Index for active records only (partial index)
CREATE INDEX idx_posts_active ON posts(created_at) WHERE deleted_at IS NULL;
```

### Audit Trail

```sql
CREATE TABLE audit_log (
  id BIGSERIAL PRIMARY KEY,
  table_name TEXT NOT NULL,
  record_id UUID NOT NULL,
  action TEXT NOT NULL,  -- INSERT, UPDATE, DELETE
  old_data JSONB,
  new_data JSONB,
  actor_id UUID,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trigger-based (automatic)
CREATE OR REPLACE FUNCTION audit_trigger() RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO audit_log(table_name, record_id, action, old_data, new_data, actor_id)
  VALUES (TG_TABLE_NAME, COALESCE(NEW.id, OLD.id), TG_OP,
          to_jsonb(OLD), to_jsonb(NEW), current_setting('app.actor_id', true)::UUID);
  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;
```

### Polymorphic Associations

```sql
-- BAD: string type column
-- commentable_type: 'Post' | 'Video' | 'Comment'

-- GOOD: separate FKs (type-safe, indexable)
CREATE TABLE comments (
  id UUID PRIMARY KEY,
  body TEXT NOT NULL,
  post_id UUID REFERENCES posts(id),
  video_id UUID REFERENCES videos(id),
  -- Exactly one must be non-null
  CONSTRAINT one_parent CHECK (
    (post_id IS NOT NULL)::int + (video_id IS NOT NULL)::int = 1
  )
);
```

## Common Mistakes

- Designing schema before knowing the queries (build for access patterns)
- Using UUIDs as primary key without considering index bloat (use ULIDs for time-ordered)
- Not adding `created_at`/`updated_at` to every table
- Cascading deletes without understanding the full graph
- N+1 schema design (forcing the app to make N queries)
- Storing money as FLOAT (use DECIMAL/NUMERIC or integer cents)
- Indexing everything (slows writes, wastes storage)
- Not planning for time zones (always use TIMESTAMPTZ)
