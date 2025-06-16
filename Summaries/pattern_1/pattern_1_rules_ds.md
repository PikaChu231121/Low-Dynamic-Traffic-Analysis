# Rule Summary for Pattern 1

### Behavioral Rules and Indicator-Outcome Relationships  

#### **General Trends**  
- **Vehicle Density (x1) Dominance**: Higher vehicle density consistently correlates with higher predicted V2U rates (y_pred), though actual outcomes (y_true) often stabilize or decline at very high densities (>5).  
- **Compute Load (x3) Sensitivity**: Low compute load (x3 < 0.01) often leads to large prediction errors (MAE spikes), suggesting the model struggles with low-load scenarios. Moderate compute loads (0.01–0.015) align better with predictions.  
- **UAV Density (x2) Insensitivity**: Despite being an input, UAV density shows minimal direct impact on outcomes, as it remains constant across all logs.  

#### **Threshold-Based Triggers**  
- **Low Vehicle Density (x1 < 1.5)**: Predictions are highly unreliable (MAE up to 2.1), especially when compute load is near zero. Actual V2U rates (y_true) are often zero here.  
- **Moderate Vehicle Density (1.5 < x1 < 4)**: System stabilizes—predictions align closely with outcomes (low MAE, NMAE < 0.5). Compute load variations have less impact.  
- **High Vehicle Density (x1 > 4)**: Predictions overshoot actual V2U rates (y_pred > y_true), with errors escalating (MAE up to 1.5). Compute load spikes (>0.01) exacerbate overestimation.  

#### **Stable vs. Unstable Regions**  
- **Stable**: Vehicle density between 1.5–4, compute load 0.005–0.01. Predictions are accurate (MAE < 0.2).  
- **Unstable**:  
  - Very low vehicle density (x1 < 1) or compute load (x3 ≈ 0).  
  - Very high vehicle density (x1 > 5), where predictions diverge sharply from reality.  

#### **Local Patterns**  
- **Compute Load Swings**: Sudden drops in compute load (e.g., x3 → 0) cause wild prediction swings (e.g., logs 5, 30–32).  
- **Mid-Range x1 (2–3)**: Small compute load changes (x3 ± 0.005) have negligible impact—system is robust here.  
- **High x1 with Low x3**: Predictions spike unrealistically (e.g., log 33: x1=5.18, x3=0.018 → y_pred=3.25 vs. y_true=1.99).  

#### **Error Patterns**  
- **Normalized Error (NMAE)**: Explodes in low-load scenarios (NMAE > 5 common when x3 < 0.005), indicating poor model calibration for edge cases.  
- **Overconfidence in High x1**: Model consistently overestimates V2U rates at high vehicle density, regardless of equation complexity.  

### Key Takeaways  
- **Vehicle density is the primary driver** of V2U rate predictions, but the relationship becomes non-linear and unreliable at extremes.  
- **Compute load is a secondary modulator**, with low values destabilizing predictions and moderate values improving accuracy.  
- **The system is most reliable** in moderate vehicle density and compute load ranges—outside these bounds, predictions degrade significantly.