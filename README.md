# NOVAres SecureHealth

A modular health insurance analytics project focused on risk scoring, fraud / abuse detection, pricing optimization, and prospect plan recommendation.

## Project overview

NOVAres SecureHealth is designed as a professional analytics repository + Streamlit dashboard to explore and demonstrate how health insurance data can be transformed into actionable intelligence for insurers, brokers, and healthcare risk teams.

The project is organized around four main modules:

1. Risk Scoring of insured members
2. Fraud / Abuse Detection
3. Pricing Optimization
4. Prospect Profiler / Plan Recommendation

## Current project status

This repository is being built from validated synthetic datasets.  
The current stage focuses on:

- structuring the analytical workflow
- defining analytical bases
- designing targets and outputs
- organizing the repository for modular and scalable development

## Available datasets

Current base datasets:

- `insured_members.csv`
- `policies.csv`
- `providers.csv`

Expected additional datasets:

- `claims_corrected.csv`
- `member_year_features_corrected.csv`
- `provider_month_features.csv`
- `prospect_survey_synthetic.csv`

## Main analytical modules

### 1. Risk Scoring
Estimate expected risk for each insured member in terms of:
- clinical severity
- utilization propensity
- expected cost

### 2. Fraud / Abuse Detection
Identify anomalous patterns at:
- member level
- provider level
- claim behavior level

### 3. Pricing Optimization
Assess pricing adequacy and estimate more appropriate premium ranges by segment.

### 4. Prospect Profiler / Plan Recommendation
Use a short survey to classify prospects and recommend the most suitable insurance plan.

## Repository structure

```text
data/           raw, processed and output datasets
notebooks/      analytical development notebooks
src/            reusable Python code
app/            Streamlit dashboard
docs/           project documentation
