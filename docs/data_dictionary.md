# Data Dictionary — NOVAres SecureHealth

## insured_members.csv

Current observed columns:
- member_id
- policy_id
- join_date
- age
- age_band
- sex
- region
- city_tier
- socioeconomic_band
- employment_status
- marital_status
- dependents_n
- family_complexity
- smoker_flag
- alcohol_risk_flag
- physical_activity_level
- bmi_group
- chronic_condition_flag
- chronic_condition_count
- chronic_group
- recurrent_medication_flag
- prior_hospitalization_24m_flag
- self_management_adherence
- archetype
- baseline_risk_score
- utilization_propensity
- acute_event_propensity
- misuse_exposure_propensity
- price_sensitivity
- coverage_preference
- network_preference
- preferred_plan_type

## policies.csv

Current observed columns:
- policy_id
- member_id
- policy_start_date
- policy_end_date
- active_flag
- plan_type
- plan_tier
- coverage_scope
- provider_network_type
- deductible_amount
- copay_level
- annual_coverage_limit
- maternity_coverage_flag
- pharmacy_coverage_flag
- chronic_care_program_flag
- premium_monthly
- premium_annual
- underwriting_load_factor
- discount_factor
- recommended_plan_flag
- pricing_adequacy_ratio
- renewal_flag
- cancellation_flag
- sales_channel
- broker_id

## providers.csv

Current observed columns:
- provider_id
- provider_name
- provider_type
- specialty_group
- region
- city_tier
- network_status
- provider_archetype
- contract_type
- base_cost_multiplier
- diagnostic_intensity
- admission_intensity
- average_claim_expected
- historical_volume_band
- historical_suspicion_flag
- provider_quality_proxy
- fraud_exposure_score
