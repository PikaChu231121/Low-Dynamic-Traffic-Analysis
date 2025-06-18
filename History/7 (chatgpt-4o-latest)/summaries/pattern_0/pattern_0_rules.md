# Rule Summary for Pattern 0

# Summary of High-Level Behavioral Rules

## Key Rules and Threshold Effects

1. **Impact of Task Success Ratio (x1):**
   - Higher values of `x1` (task success ratio) generally correlate with higher predicted compute loads. This suggests that as the task success ratio increases, the system anticipates a higher compute load in the next slot.

2. **Influence of Vehicle Density (x2):**
   - The vehicle density (`x2`) is often used in logarithmic form in the equations, indicating a non-linear relationship. Higher vehicle densities tend to increase the predicted compute load, especially when combined with other factors like task success ratio.

3. **Role of V2U Density (x3):**
   - The V2U density (`x3`) significantly affects the predicted compute load when it is non-zero. Higher values of `x3` tend to increase the predicted compute load, indicating that more vehicle-to-UAV interactions lead to higher computational demands.

4. **Current and Previous Compute Load (x4 and x5):**
   - The current (`x4`) and previous (`x5`) compute loads are consistently used across equations, suggesting their critical role in predicting future loads. The model often predicts the next compute load based on a weighted combination of these two indicators.

## Temporal Trends

- **Moving Averages:**
  - The use of moving averages in some equations indicates that the model considers temporal trends and smooths out short-term fluctuations to predict the next compute load more accurately.

- **Lag Effects:**
  - The presence of previous-slot compute load (`x5`) in the equations highlights the model's reliance on temporal dependencies. This suggests that recent history is a strong predictor of immediate future states.

## Model Performance

- **Worsening Performance:**
  - The model's performance tends to worsen (higher MAE and NMAE) when there are sudden changes or spikes in the input indicators, particularly when `x3` is zero or very low, suggesting that the model struggles with abrupt changes in V2U density.
  - High prediction errors are also observed when the task success ratio (`x1`) is high, but the compute load is expected to be low. This indicates potential overfitting or a lack of sufficient data to generalize well in these scenarios.

## Indicator Effects on Output

- **Task Success Ratio (x1):**
  - A higher task success ratio generally leads to an increase in predicted compute load, indicating that successful task completions drive up computational demand.

- **Vehicle Density (x2):**
  - Vehicle density has a logarithmic effect, suggesting diminishing returns on its impact on compute load as density increases.

- **V2U Density (x3):**
  - The presence of V2U density significantly increases the predicted compute load, highlighting the importance of vehicle-to-UAV interactions in computational demand.

- **Compute Loads (x4 and x5):**
  - Current and previous compute loads are strong predictors of future load, with their effects being additive in nature. This indicates a persistent and cumulative effect of computational demands over time.

Overall, the model captures complex interactions between the indicators, with particular emphasis on recent compute loads and vehicle-to-UAV interactions. The performance is generally stable but can degrade with abrupt changes in key indicators.