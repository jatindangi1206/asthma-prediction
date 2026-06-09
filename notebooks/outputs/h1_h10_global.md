# H1-H10 derived from the processed HRV cohort

Cohort: 109 patients.

| ID | Characteristic | Cohort evidence | Verdict |
|---|---|---|---|
| H1 | Bounded & strictly positive | 100% patients all-positive; median 0 rows >300 ms | CONFIRMED |
| H2 | Bimodal value distribution | 0% by BC>0.555; 99% by GMM dBIC>10 | PRESENT |
| H3 | Power-law (1/f) spectrum | median slope -0.54 | 1/f-like |
| H4 | Non-stationary baseline | 0% non-stationary (ADF p>=0.05); median rolling-CV 0.10 | MIXED |
| H5 | Circadian rhythm | median 3.0% PSD in 20-28h; 5.5% in 9-14h | WEAK |
| H6 | Heavy-tailed noise | median first-diff excess kurtosis 0.66; sigma_x 15.7 ms | HEAVY-TAILED |
| H7 | Sampling regularity | median step 10.0 min; 95% steps ~10 min | REGULAR |
| H8 | Missingness (gaps) | median 55.4% NaN gap-fill rows | PRESENT |
| H9 | Online operation | system requirement, not a data property | N/A |
| H10 | Between-patient variability | SD of patient means 7.9 ms; median patient SD 32.5 ms | HIGH |
