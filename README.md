# aimleict
CAPSTONE PROJECT
## Data findings

1. **Service type:** Express shipments have the highest delay rate at 0.4897 (48.97%), compared with 0.2056 (20.56%) for Standard and 0.0857 (8.57%) for Economy shipments.

2. **Hub tier:** The delay rate increases from 0.0400 (4.00%) for Tier 1 hubs to 0.2150 (21.50%) for Tier 2 and 0.4400 (44.00%) for Tier 3 hubs.

3. **Distance:** Delay rates rise with distance: 0.0219 (2.19%) for 0–200 km, 0.1368 (13.68%) for 201–400 km, 0.4364 (43.64%) for 401–600 km, and 0.8523 (85.23%) for 601+ km.

4. **Correlations with delivery time:** The three strongest correlations with `delivery_hours`, excluding the target itself, are `promised_hours` (+0.8372), `distance_km` (+0.4013), and `n_stops` (+0.3247).

5. **Declared-value transformation:** The median-filled `declared_value_inr` column has a mean of INR 1,533.0542 and a median of INR 1,121.5, while its natural-log transformation has a mean of 7.0277 and a median of 7.0224. The log transformation reduces the influence of the right-skewed declared-value distribution.
## Model results (regression)

### Baseline performance

The mean baseline achieved a validation R² of **-0.0212** and an RMSE of **18.8552 hours**. The negative R² means that predicting the training-set mean performs slightly worse than predicting the validation-set mean for every validation shipment.

### Model comparison

Linear regression achieved the lowest validation RMSE of **4.8912 hours**, with an R² of **0.9313**. Gradient boosting achieved a validation RMSE of **4.9165 hours** and an R² of **0.9306**. Random forest achieved an RMSE of **5.4679 hours**, while the selected decision tree achieved an RMSE of **6.2796 hours** at depth 6.

Linear regression was selected as the final model because the assignment specifies choosing the candidate with the lowest validation RMSE.

### Decision-tree depth and bias-variance trade-off

Depth 1 exhibits high bias: its training RMSE is 11.6334 hours and its validation RMSE is 12.1324 hours. The model is too simple to capture the relationships in the data.

Depth 6 has the lowest validation RMSE at 6.2796 hours. At unlimited depth, training RMSE falls to 0.0000 hours, but validation RMSE rises to 7.7890 hours. This gap of 7.7890 hours indicates high variance and overfitting. Increasing tree depth improves training fit but does not guarantee better generalization.

### Gradient-boosting learning rate

The learning rate of **0.1** performs best, achieving validation R² of 0.9306 and RMSE of 4.9165 hours. At 0.01, validation R² falls to 0.7127 because learning progresses too slowly with 100 estimators. At 1.0, validation R² falls to 0.8440, consistent with overly aggressive updates that overshoot a better solution.

### Cross-validation and final model choice

Five-fold cross-validation on the combined training and validation data produced a mean R² of 0.9261 (standard deviation 0.0048) for linear regression and 0.9352 (standard deviation 0.0044) for gradient boosting.

Cross-validation favors gradient boosting, whereas validation RMSE favors linear regression. Linear regression remains the selected model because the assignment defines the final selection criterion as the lowest validation RMSE.

### Final test performance

The selected linear regression model achieved a test R² of **0.9493** and an RMSE of **4.5235 hours**. The test set was held out during model selection and evaluated only once.
## Model results (classification)

### Baseline and classifier comparison

The blind baseline achieved validation accuracy of **0.7667 (76.67%)**, but its precision, recall, and F1 were all zero because it predicted every parcel as on time and detected no delayed parcels. This demonstrates why accuracy alone is insufficient for an imbalanced classification problem.

Logistic regression achieved validation F1 of **0.8046**, with precision of 0.7778 and recall of 0.8333. The selected decision tree at depth 6 achieved F1 of 0.7470, while the default random forest using all 17 features achieved F1 of 0.7848.

### Feature importance and feature selection

The five bookkeeping noise columns account for approximately **22.44%** of the default random forest's total feature importance. All five rank outside the top seven by F-score, and each has an F-score below 10.

The seven highest-ranked features are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag`. The `SelectKBest` experiment achieved its highest validation F1 of **0.8333** with seven features. The corresponding F1 scores for 5, 10, 12, and 17 features were 0.6988, 0.7857, 0.8049, and 0.7848, respectively.

### Hyperparameter tuning

Grid search achieved a best cross-validation accuracy of 0.8952 and validation F1 of 0.7750. Randomized search achieved a best cross-validation accuracy of 0.8905 and validation F1 of 0.7848. The untuned random forest on the same seven selected features achieved the highest validation F1, **0.8333**, and was selected as the final configuration.

Tuning did not improve validation F1 on this split. This illustrates that a more extensive search does not necessarily improve performance on a relatively small validation set of 180 parcels. The validation result is used for configuration selection, while the held-out test set provides the final evaluation.

### Alert threshold and operational trade-off

The alert threshold was selected using the operational rule: choose the highest threshold whose validation recall is at least 0.80. The chosen threshold was **0.5**, with validation precision of 0.7955, recall of 0.8333, and F1 of 0.8140.

At threshold 0.6, precision improves to 0.8889, but recall falls to 0.7619, below the required 0.80. Therefore, threshold 0.5 is the highest threshold that satisfies the operations rule. It prioritizes detecting delayed parcels while limiting unnecessary alerts.

### Final test performance

At the selected threshold of 0.5, the final model achieved test precision of **0.6889**, recall of **0.7561**, and F1 of **0.7209**, with accuracy of 0.8667. Test recall was lower than validation recall (0.8333), but remained above the required minimum of 0.55. The test set contained only 41 delayed parcels, so each additional missed or detected delayed parcel can noticeably affect the reported metrics.

The final configuration used `SelectKBest(f_classif, k=7)` and a random forest with 100 estimators, unlimited depth, and `min_samples_leaf=1`. The final test set was evaluated once after model and threshold selection.
## How to run

Run the following commands from the repository root directory.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Generate the datasets

```bash
python generate_data.py
```

### 3. Run the analysis and modelling pipeline

```bash
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
```

### 4. Package the final model

```bash
python package_model.py
```

This rebuilds the final classification pipeline using the saved configuration, fits it on the prescribed training split, saves the model and supporting metadata under `models/`, and generates a model reload consistency check.

### 5. Score new shipments

```bash
python predict_delay.py
```

The prediction script reads `data/new_shipments.csv`, cleans the data using the saved training medians, loads the packaged model and hub segments, and writes shipment-level recommendations to `outputs/new_shipment_actions.csv`.

**Reproducibility:** The project uses fixed random seeds for the prescribed splits and models. Install the pinned dependencies before loading the committed model pickle. If the evaluation environment uses different library versions, rerun `package_model.py` in that environment before scoring shipments.

## Operations recommendation

### Hub segmentation

The K-Means analysis retains three operational segments. The segments are named by their average observed delay rate.

| Segment | Number of hubs | Mean delivery time (hours) | Delay rate | Mean distance (km) |
|---|---:|---:|---:|---:|
| Standard | 7 | 47.24 | 2.57% | 188.93 |
| Watch | 8 | 56.81 | 15.50% | 314.18 |
| Escalate | 9 | 64.65 | 46.00% | 470.48 |

**Recommended actions:**

- **Standard:** Maintain routine operations and monitor delivery performance.
- **Watch:** Monitor delays closely and consider proactive customer communication when a shipment is flagged.
- **Escalate:** Route flagged shipments to the hub manager for investigation and intervention.

### Tier and segment mismatches

Hub tier and observed-performance segments do not always agree. The following mismatches deserve attention:

| Hub | City | Hub tier | Performance segment |
|---|---|---|---|
| HUB07 | Kolkata | Tier 1 | Watch |
| HUB13 | Nagpur | Tier 2 | Escalate |
| HUB15 | Kochi | Tier 2 | Escalate |
| HUB22 | Dehradun | Tier 3 | Watch |

These differences suggest that hub tier alone should not determine operational priority. In particular, HUB13 and HUB15 warrant investigation because they belong to the Escalate segment despite being Tier 2 hubs. HUB07 and HUB22 should also be reviewed because their Watch classification differs from what a tier-only approach might suggest.

### Delay prediction and threshold

The final classifier uses a probability threshold of **0.50**. A shipment is flagged when its predicted delay probability is at least this threshold.

On the held-out classification test set:

- Precision: **68.89%**
- Recall: **75.61%**
- F1-score: **72.09%**

Approximately 7 of every 10 flagged predictions were delayed in the test set, while the model detected approximately 8 of every 10 delayed shipments. These are rounded interpretations of the measured metrics.

For the 48 new shipments, the prediction output contains:

| Action | Shipments |
|---|---:|
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| No action | 40 |
| **Total** | **48** |

The threshold should be reviewed using operational costs. Missing a genuinely delayed shipment may be more costly than contacting a customer unnecessarily. The test-set results should not be treated as a guarantee of future performance.

### Why retain three clusters?

The separation ratio increases from **5.3928 for K=3** to **6.3047 for K=4** and **6.8866 for K=5**. However, the K=4 solution contains a cluster of only four hubs, and the K=5 solution contains clusters of three and two hubs.

K=3 is retained as a practical compromise between cluster separation, segment size, and operational interpretability. Small clusters in the higher-K solutions may be less useful for establishing stable operating procedures.

## Production plan

The current repository provides a starting point for a repeatable shipment-risk workflow. Before production deployment, the model should be integrated with live shipment events and validated against real operational outcomes.

### Repository components and operational use

| Production characteristic | Repository component or implementation |
|---|---|
| Accuracy | `outputs/part_b_metrics.csv`, `outputs/part_c_final_test.csv` |
| Reliability | `outputs/part_d_reload_check.csv` verifies that reloaded-model probabilities match the original model probabilities. |
| Reproducibility | Fixed random seeds, pinned dependencies in `requirements.txt`, and `package_model.py` |
| Maintainability | Shared cleaning and feature engineering in `src/shipment_prep.py` |
| Consistency | Saved imputation medians and feature order under `models/` |
| Monitorability | `outputs/new_shipment_actions.csv` provides probabilities, flags, segments, and recommended actions. |
| Performance | The final classifier uses seven selected features and a Random Forest. Measure prediction latency and throughput on production infrastructure before deployment. |

### Monitoring plan

The following are proposed initial monitoring thresholds. They are operational starting points, not thresholds established by the current synthetic experiment.

| Quantity to monitor | Proposed trigger | Response |
|---|---|---|
| Delayed-shipment rate | Above 30% for two consecutive weeks, excluding special-event periods | Investigate hub capacity, routes, weather, and handling processes. |
| Flagged-shipment rate | Above 30%, or a change of more than 10 percentage points from the established baseline | Check shipment mix, input quality, and prediction behaviour. |
| Recall on matured shipments | Below 70% over a rolling window of 100 shipments with confirmed outcomes | Review missed delays, recalibrate the threshold, and evaluate retraining. |
| Feature drift | A sustained material shift in distance, promised hours, stop count, or other model inputs relative to the training baseline | Investigate changes in operations and assess model validity. |
| Data quality | Missing or invalid required fields exceed the agreed data-quality tolerance | Fix upstream data collection and prevent unreliable predictions from being used automatically. |

### Retraining schedule

- Evaluate model performance monthly once sufficient confirmed outcomes are available.
- Retrain quarterly as an initial schedule, or earlier when sustained drift or declining performance meets the agreed triggers.
- Validate every candidate model on a time-appropriate holdout set before deployment.
- Record the model version, feature order, imputation values, threshold, evaluation metrics, and deployment date for each release.
- Compare sale-event periods against comparable sale-event periods. In particular, the **September 15–28 sale window** should be evaluated against similar promotional windows rather than ordinary weeks alone.
- Keep human review for escalations until real-world performance and operational impact have been validated.

## Limitations

1. **Synthetic data:** The project uses generated shipment data. Its performance does not establish that the model will achieve the same results on real courier operations.
2. **Limited test set:** The classification test set contains 180 shipments, including 41 delayed shipments. Precision and recall estimates may change substantially on larger or different datasets.
3. **Single split and threshold selection:** The selected threshold and reported final test performance depend on the prescribed split and modelling procedure. Further time-based and out-of-sample validation is needed.
4. **Correlation is not causation:** Hub segments and feature rankings describe patterns in the generated data. They do not prove that changing a particular feature will reduce delays.
5. **Clustering limitations:** The three hub segments are descriptive groupings. Their boundaries may change as shipment patterns change, and segment labels should be recalculated when the clustering model is refitted.
6. **Distribution drift:** Weather, route changes, staffing, capacity, and promotional events can change shipment patterns and reduce predictive performance.
7. **Operational costs are not fully modelled:** The threshold should be selected with real costs for missed delays, customer calls, and hub-manager escalations.
8. **Production readiness:** The current scripts demonstrate an offline pipeline, not a fully monitored production service. Live data integration, access controls, logging, alerting, and operational validation remain necessary.
