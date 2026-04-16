# Quant Knowledge Routes

This project now treats quant knowledge as separate sources instead of one mixed blob.

## Current sources

### 1. Factor registry
- Path: `git_ignore_folder/factor_outputs/manifest.csv`
- Path: `git_ignore_folder/factor_outputs/leaderboard.csv`
- Purpose:
  - accepted factor metadata
  - IC ranking
  - tags
  - logic summaries
  - source type (`agent_generated` vs `literature_report`)

### 2. Implementation experience
- Path: `git_ignore_folder/research_store/knowledge/costeer_knowledge_base.pkl`
- Purpose:
  - past successful implementations
  - failed implementations
  - code repair experience
  - implementation feedback

### 3. Factor-improvement papers
- Dropbox folder: `papers/factor_improvement/`
- Structured summary store: `git_ignore_folder/research_store/knowledge_v2/paper_improvement/paper_improvement_ideas.jsonl`
- Ingest command: `rdagent ingest_factor_papers`
- Purpose:
  - summarize papers about improving factors
  - provide idea-level knowledge for proposing and improving factors

### 4. Error cases
- Path: `git_ignore_folder/research_store/knowledge_v2/error_cases/factor_error_cases.jsonl`
- Purpose:
  - store structured bug patterns and repair hints
  - keep coding-stage knowledge smaller and cleaner than raw feedback dumps

## Workflow routing

### Factor hypothesis generation
Read from:
- factor leaderboard
- factor manifest
- paper improvement summaries

Do not read from:
- raw implementation code
- raw execution logs

### Factor hypothesis expansion
Read from:
- factor leaderboard
- paper improvement summaries

Do not read from:
- raw implementation failures unless the task is already in coding stage

### Factor coding and fixing
Read from:
- implementation experience knowledge base
- structured error cases

Do not read from:
- full paper text by default
- factor leaderboard as primary code-repair knowledge

### Factor selection and modeling
Read from:
- factor leaderboard
- factor manifest

Do not read from:
- implementation code
- raw paper text

## Why this structure

The old structure mixed:
- task definitions
- implementation code
- long feedback
- paper ideas
- ranking metadata

That made retrieval noisy and fragile. The new routing keeps each workflow step focused on the smallest relevant knowledge source.

## PostgreSQL migration target

This routing layer is designed so the current file-based stores can later move to PostgreSQL + pgvector without changing workflow logic:
- factor registry -> PostgreSQL tables
- paper improvement summaries -> PostgreSQL tables + pgvector
- error cases -> PostgreSQL tables + pgvector
- implementation experience -> PostgreSQL tables or a compatibility import path

The workflow should keep calling the routing layer rather than reading raw files directly.
