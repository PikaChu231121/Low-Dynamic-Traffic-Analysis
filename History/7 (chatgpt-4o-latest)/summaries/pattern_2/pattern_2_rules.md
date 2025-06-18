# Rule Summary for Pattern 2

## Summary of Behavioral Rules and Patterns

### Causal and Threshold Rules

1. **Impact of V2U Rate (x1):**
   - A higher V2U rate generally correlates with a higher predicted task success ratio. This is evident in cases where `x1` is significantly greater than zero, leading to better predictions.
   - When `x1` is low or zero, the model tends to rely more on other indicators like compute load (x3), which can lead to increased prediction errors.

2. **Effect of Compute Load (x3):**
   - A higher compute load tends to decrease the predicted task success ratio. This is particularly noticeable when `x3` is non-zero, and the model equations incorporate terms that subtract or logarithmically transform `x3`.
   - There is a threshold effect where low values of `x3` (close to zero) are associated with higher task success ratios, but as `x3` increases, the success ratio tends to decrease.

3. **Influence of V2I Rate (x2):**
   - The V2I rate appears to have a moderating effect. Higher values of `x2` are associated with more accurate predictions, suggesting that it may stabilize or enhance the impact of other indicators.

### Temporal Trends

- **Model Behavior Over Time:**
  - As the dataset progresses, there is a noticeable shift in the equations used, indicating adaptation to changing conditions or inputs. For instance, early entries use simpler linear models, while later entries incorporate more complex expressions involving logarithms and exponentials.
  - Over time, the model appears to place more emphasis on `x1` and `x2`, possibly due to their stabilizing effects compared to the more volatile `x3`.

### Model Performance and Errors

- **Performance Degradation:**
  - The model's performance worsens when `x3` is high, leading to larger prediction errors. This suggests that the compute load is a critical factor that can negatively impact task success if not managed properly.
  - High normalized mean absolute error (NMAE) values in later entries indicate that the model struggles with certain configurations, particularly when `x3` is substantial or when `x1` and `x2` are low.

- **Error Patterns:**
  - In scenarios with low `x1` and high `x3`, the model's predictions deviate more from the true values, indicating a need for improved handling of these conditions.
  - The presence of exponential and logarithmic terms in later models suggests attempts to capture non-linear relationships, which may contribute to improved accuracy in some cases but also increase complexity and potential for error.

### Indicator Effects on Output

- **V2U Rate (x1):** Strong positive correlation with task success ratio. Higher `x1` generally leads to better outcomes.
- **Compute Load (x3):** Negative correlation with task success ratio. Increased `x3` often results in decreased success, highlighting the importance of managing computational demands.
- **V2I Rate (x2):** Acts as a stabilizing factor, with higher values supporting better predictions and potentially mitigating the adverse effects of high `x3`.

Overall, the data suggests that effective management of V2U rates and compute loads is crucial for maintaining high task success ratios in vehicular-UAV collaborative systems. The model's ability to adapt to changing conditions and inputs is reflected in the evolving complexity of its predictive equations.