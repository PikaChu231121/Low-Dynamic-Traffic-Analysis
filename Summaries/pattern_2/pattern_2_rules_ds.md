# Rule Summary for Pattern 2

### Behavioral Rules and Indicator-Outcome Relationships  

#### **General Trends**  
- **Task success ratio (y)** is generally high (0.8–0.96), with minor fluctuations tied to input indicators.  
- **V2U rate (x1)** and **V2I rate (x2)** show stronger influence on outcomes than **compute load (x3)**, especially at higher values.  

#### **Causal Patterns**  
- **High V2U/V2I rates correlate with higher task success**:  
  - When `x1 > 1.5` and `x2 > 0.2`, `y` consistently exceeds 0.9.  
  - Extremely high `x2` (e.g., >3.5) sometimes leads to slight saturation or instability in predictions (e.g., overprediction to `y=1.0`).  
- **Compute load (x3) has nonlinear effects**:  
  - Low-to-moderate `x3` (0–0.01) has negligible impact on `y`.  
  - Very low `x3` (near 0) combined with high `x1/x2` can trigger overprediction (e.g., `y_pred=1.0`), suggesting a system blind spot.  

#### **Threshold-Based Triggers**  
- **Stable region**: `x1 > 1.5` and `x2 > 0.5` → `y` stabilizes around 0.92–0.95.  
- **Unstable region**:  
  - When `x3 ≈ 0` and `x1/x2` are high, predictions may spike to `y=1.0` (likely due to division-by-zero-like effects in the model).  
  - Low `x1/x2` (near 0) with nonzero `x3` leads to lower `y` (0.8–0.85).  

#### **Local Patterns**  
- **V2U rate (x1) dominates at mid-range values**:  
  - For `x1 ≈ 2.0–2.5`, even small increases in `x1` boost `y` (e.g., entries 5–11 show steady rise).  
- **V2I rate (x2) gains influence at extremes**:  
  - When `x2 > 3.0`, its impact on `y` becomes less predictable (e.g., entries 27–32 show oscillation despite high `x2`).  

#### **System Stability**  
- **Most stable predictions**: When `x1 > 1.0`, `x2 > 1.0`, and `x3 > 0.005` → `MAE` is consistently low (<0.01).  
- **Least stable predictions**: When `x3 ≈ 0` → `MAE` spikes (e.g., entries 15–19, 29, 32).  

#### **Key Takeaways**  
1. **V2U/V2I rates are primary drivers** of task success; compute load is secondary.  
2. **Near-zero compute load destabilizes predictions**, likely due to model limitations.  
3. **High V2I rates (>3.0) may indicate system saturation**, where further increases don’t improve outcomes.  
4. **Optimal operating region**: Moderate-to-high `x1/x2` with nonzero `x3` ensures stable, high task success.  

---  
*Note: All observations are based on empirical trends in the logs, ignoring model-specific equations.*