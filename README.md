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
