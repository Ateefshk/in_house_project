# OULAD data quality report

Source tables were read without modifying the input files.

## Input inventory

| file | rows | columns | bytes |
| --- | --- | --- | --- |
| studentInfo.csv | 32593 | 12 | 3462763 |
| studentRegistration.csv | 32593 | 5 | 1132550 |
| studentVle.csv | 10655280 | 6 | 453836331 |
| assessments.csv | 206 | 6 | 8211 |
| studentAssessment.csv | 173912 | 5 | 5690483 |
| courses.csv | 22 | 3 | 526 |
| vle.csv | 6364 | 6 | 270612 |

## Outcome distribution

| final_result | rows |
| --- | --- |
| Pass | 12361 |
| Withdrawn | 10156 |
| Fail | 7052 |
| Distinction | 3024 |

## Enrollment key and duplicate checks

| file | duplicate_full_rows | duplicate_enrollment_keys |
| --- | --- | --- |
| studentInfo.csv | 0 | 0.0 |
| studentRegistration.csv | 0 | 0.0 |
| studentVle.csv | 787170 |  |
| assessments.csv | 0 |  |
| studentAssessment.csv | 0 |  |
| courses.csv | 0 |  |
| vle.csv | 0 |  |

## Cutoff cohorts

| cutoff_days | cutoff_end_day_inclusive | eligible_enrollments | eligible_students | at_risk_prevalence | excluded_unregistered_by_cutoff | excluded_registered_after_cutoff | excluded_missing_registration_date | excluded_withdrawn_timing_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 6 | 29002 | 25968 | 0.4709330391007517 | 3361 | 93 | 45 | 92 |
| 14 | 13 | 27970 | 25152 | 0.45084018591347874 | 4427 | 59 | 45 | 92 |
| 21 | 20 | 27787 | 25033 | 0.44675567711519776 | 4642 | 26 | 45 | 93 |
| 28 | 27 | 27435 | 24758 | 0.43947512301804265 | 5002 | 18 | 45 | 93 |

Missing values and per-column types are in `missing_values.csv`; exclusions with their reasons are in `excluded_records.csv`.