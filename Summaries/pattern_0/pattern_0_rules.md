# Rule Summary for Pattern 0

### Behavioral Rules and Indicator-Outcome Relationships  

#### **General Trends**  
- **Task Success Ratio (x1)**: Higher values (≥0.93) often correlate with stable or slightly decreasing compute load, but sudden drops in load (e.g., to 0) can occur unpredictably even with high success rates.  
- **Vehicle Density (x2)**:  
  - Low density (<1.5): Compute load remains low (≤0.015) unless V2U density (x3) is nonzero.  
  - High density (>5): Compute load becomes more volatile, with occasional spikes (e.g., 0.018) despite high task success ratios.  
- **V2U Density (x3)**:  
  - Zero values: Compute load tends to stay low or decay.  
  - Nonzero values (>1.5): Often precede moderate compute load (0.005–0.012), but influence diminishes if x3 fluctuates sharply.  

#### **Threshold-Based Triggers**  
- **Compute Load Spikes**: Occur when:  
  - *x3 suddenly rises* (e.g., from 0 to >2) *and* x4 (current load) is near zero (e.g., logs 33, 43).  
  - *x2 > 5 and x1 drops below 0.94* (e.g., log 45: load spikes to 0.013 despite x1=0.943).  
- **Load Collapses**: Happen when:  
  - *x3 drops below 1.5* after sustained high values, *and* x4 is already low (e.g., logs 15–17).  

#### **Stable vs. Unstable Regions**  
- **Stable**:  
  - *Low x2 (<2) + low x3 (≈0)*: Predictable low load (≤0.01).  
  - *High x1 (>0.95) + moderate x3 (1.5–2.5)*: Load fluctuates mildly (0.005–0.01).  
- **Unstable**:  
  - *High x2 (>5) + fluctuating x3*: Load exhibits erratic spikes/drops (e.g., logs 33–49).  
  - *x4 near zero*: Next-slot load is prone to abrupt changes (e.g., logs 5, 15).  

#### **Local Patterns**  
- **Short-Term Influence of x3**:  
  - Rapid increases in x3 (e.g., from 0 to 2+ within 1–2 slots) often lead to delayed load spikes (1–2 slots later).  
- **Hysteresis Effect**:  
  - After a spike (e.g., x4=0.018), load tends to decay gradually even if x3 remains high (e.g., logs 33–35).  

#### **Anomalies**  
- **False Predictions**:  
  - When x4=0, predictions often overestimate next-slot load (e.g., logs 5, 16, 19).  
  - High x1 (>0.95) sometimes fails to prevent load spikes if x2 and x3 are both elevated (e.g., log 33).  

### Summary  
The system is most stable under low vehicle/V2U densities but becomes volatile when either density exceeds moderate thresholds. V2U density acts as an amplifier for compute load, while vehicle density exacerbates unpredictability at high levels. Task success ratio alone is insufficient to prevent load spikes when other indicators are unstable.