# Project Blueprint — NOVAres SecureHealth

## Objective

Build a modular analytics repository and dashboard for synthetic health insurance data, covering:

1. Risk Scoring
2. Fraud / Abuse Detection
3. Pricing Optimization
4. Prospect Profiling / Plan Recommendation

## Current data available

Confirmed:
- insured_members.csv
- policies.csv
- providers.csv

Pending:
- claims_corrected.csv
- member_year_features_corrected.csv
- provider_month_features.csv
- prospect_survey_synthetic.csv

## Development phases

### Phase 1 — Data foundation
- audit datasets
- define keys and joins
- create analytical base tables
- document variables

### Phase 2 — Risk scoring
- define targets
- create member risk features
- build risk outputs

### Phase 3 — Fraud / abuse
- define member/provider anomaly rules
- create suspicion scores
- prioritize explainability

### Phase 4 — Pricing
- compare premiums vs expected cost
- estimate pricing adequacy
- define premium guidance bands

### Phase 5 — Prospect profiler
- map survey responses to archetypes
- estimate risk and expected utilization
- recommend suitable plans

## Final dashboard pages

1. Portfolio Overview
2. Risk Scoring
3. Fraud & Abuse
4. Pricing Intelligence
5. Prospect Profiler
