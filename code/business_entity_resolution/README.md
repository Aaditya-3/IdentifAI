# Business Entity Resolution Pipeline

This repository contains an end-to-end, single-environment pipeline for resolving noisy business entities. It leverages an embedded SQLite inverted index for sub-quadratic candidate generation and a LightGBM classifier optimized specifically for the precision-heavy $F_{0.5}$ metric.

## 1. Environment Setup
This pipeline adheres strictly to the offline, single-environment constraints (no external APIs, open-source models only, $<8B$ parameters). 

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt