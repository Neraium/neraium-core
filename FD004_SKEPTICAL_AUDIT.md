# SKEPTICAL AUDIT: FD004 PROOF RESULTS
## Critical Validity Review

---

## TOP 5 RISKS TO VALIDITY

### ⚠️ RISK #1: FIXED THRESHOLD MASQUERADING AS LEARNED DETECTION (CRITICAL)

**Findings:**
- **21.9% of all alerts trigger at exactly cycle 58** (53/242 detected units)
- **21.6% of all watches trigger at exactly cycle 60** (32/148 units)
- **49.2% of detections cluster in cycles 51-75**, suggesting a narrow fixed threshold
- Alert fires at ~30-32% of unit lifetime (mean: 32.7%), remarkably consistent across all failure times

**Why this matters:**
- This is **NOT adaptive degradation detection**; it's a fixed percentage of unit lifetime
- The algorithm detects a **regime transition at ~30% lifecycle**, not actual degradation
- Could be triggered by normal operating mode change, not failure indication
- Threshold is **brittle**: shifting by ±10 cycles loses NO detections, indicating massive safety margin

**Red flag example:**
- Cycle 58 units have mean failure at cycle 244 (lead time 186 cycles)
- Variable cycle units have mean lead time only 167 cycles
- Fixed threshold is actually performing WORSE but marked as higher quality

---

### ⚠️ RISK #2: MISLEADING EARLY SIGNAL IMPROVEMENT STATISTIC (HIGH)

**Findings:**
- **Median improvement: 10.0 cycles** (green line in histogram)
- **Mean improvement: 22.55 cycles** (red line, inflated by outliers)
- **Median improvement is pulled down by just 6 units with >100 cycle improvement**
- 23.6% of units with both signals have <5 cycle improvement
- 69.5% of units have <20 cycle improvement

**The deception:**
- Proof emphasizes "22.55 cycles (14% better)" — this is the mean
- Silent on the median: only 10 cycles
- The distribution is **heavily right-skewed**: 4 units account for most of the improvement
- Removing 20 outlier units would reduce mean to ~17 cycles

**Mathematical reality:**
- Mode (most common): <10 cycles
- Median (typical): 10 cycles
- Mean (reported): 22.55 cycles ← **NOT representative**

---

### ⚠️ RISK #3: QUALITY CLASSIFICATION CONFLATES LIFECYCLE POSITION WITH DETECTION QUALITY (HIGH)

**Findings:**
- Long-running units (>300 cycles): **93.1% marked "very_early"** (54/58)
- Short-running units (<150 cycles): marked "good" or "usable"
- Classification appears to depend on **failure_cycle, not detection quality**

**Problem:**
- A unit failing at 500 cycles with alert at 60 (440 cycle lead) is marked "very_early"
- A unit failing at 140 cycles with alert at 60 (80 cycle lead) is marked "usable"
- Same alert behavior → **different quality ratings based on unit lifetime**
- Suggests quality classification is **post-hoc justification**, not pre-defined metric

---

### ⚠️ RISK #4: UNVALIDATED DATA LEAKAGE RISK - FUTURE INFORMATION ENCODED (MEDIUM)

**Findings:**
- Failure_cycle is from **complete validation run** — known before alert/watch logic
- Alert fires at 30-32% of lifetime: **this is derived from the full failure time**
- Early signal fires at ~25-27% of lifetime: **also positioned relative to known failure**
- No evidence of **forward-validation** (blind test on held-out data)

**Critical question:**
- How were alert/watch thresholds chosen?
  - If trained on full FD004 runs with known failure times: **DATA LEAKAGE**
  - If chosen blind: why is alert positioned so precisely at 30%?

---

### ⚠️ RISK #5: REGIME SHIFT DETECTION vs DEGRADATION DETECTION (MEDIUM)

**Findings:**
- Alert fires at consistent ~30% of unit lifecycle across all failure times
- This is **incompatible with RUL-proportional degradation**
- More consistent with **mode/regime transition** detection:
  - Unit enters operating regime A at cycle 0
  - Unit transitions to operating regime B at cycle 58-60 (typical)
  - Unit fails at cycle 230 ± 85 (170-290 range)

**Alternative explanation:**
- System detects regime shift, not degradation
- Users conflate regime shift with failure prediction
- Would explain:
  - Why fixed cycle 58 works across different failure times
  - Why "very_early" detections are actually just regime shifts
  - Why early signal provides little additional information (both detecting same regime)

---

## SUPPORTING EVIDENCE FROM DATA ANALYSIS

### Systematic Bias Evidence
| Metric | Finding |
|--------|---------|
| Most common alert cycle | 58 (21.9% of detections) |
| Most common watch cycle | 60 (21.6% of watch detections) |
| Cycles 51-75 clustering | 49.2% of all alerts |
| Distinct alert cycles | Only 88 out of 248 units (35.5%) |
| Alert as % of lifetime | 32.7% ± 13.5% (very tight clustering) |

### Distribution Red Flags
| Lead Time Metric | Value | Concern |
|------------------|-------|---------|
| Mean alert lead | 175.49 cycles | Inflated by outliers |
| Median alert lead | 164.5 cycles | More representative |
| Mean watch lead | 183.57 cycles | Watch is WORSE on average |
| Median improvement | 10.0 cycles | Mode is <5 cycles |
| Mean improvement | 22.55 cycles | Pulled up by 6 outliers |
| Skewness | 0.95 | Right-skewed, non-normal |
| Kurtosis | 1.39 | Fat tail (outlier-sensitive) |

### Quality Classification Paradox
| Group | Count | Mean Lead | Marked Quality |
|-------|-------|-----------|----------------|
| Fixed cycle 58 | 53 | 186.4 | good (62%), very_early (34%) |
| Cluster 60-65 | 33 | 197.9 | good (52%), very_early (39%) |
| Variable cycles | 162 | **167.1** | **good (52%), miss (4%)** |

**Observation:** Variable cycles have BETTER lead times but higher miss rate. Fixed cycles have WORSE lead times but better quality rating.

---

## WHAT EVIDENCE WOULD STRENGTHEN CONFIDENCE?

1. **Blind validation on held-out units**
   - Prove thresholds were NOT tuned on failure_cycle information
   - Run on future data never used in development

2. **Forward-in-time validation**
   - Collect real operating data, predict before observing failures
   - Show lead times match proof claims

3. **Causal analysis**
   - Prove alert fires due to degradation, not regime transition
   - Compare alert timing against physical degradation metrics
   - Show alert correlates with RUL, not with lifecycle position

4. **Statistical significance**
   - Test if mean vs median difference is significant
   - Quantify contribution of outliers to claimed improvement
   - Show improvement holds for 80th percentile units, not just mean

5. **Threshold sensitivity analysis**
   - Vary alert threshold by ±5, ±10, ±20 cycles
   - Show detection rate degrades monotonically
   - Currently, threshold can move 10 cycles with zero loss

6. **Baseline comparison**
   - Compare to naive fixed-cycle baseline (cycle 60 alert)
   - Compare to RUL-proportional baseline
   - Prove learned system beats simple alternatives

---

## REQUIRED TESTS BEFORE STRONG CLAIMS

### 1. REGIME SHIFT vs DEGRADATION TEST (CRITICAL)
- Extract signal degradation metrics from unit telemetry
- Plot alert timing against actual degradation curves
- If alert fires at fixed % lifecycle regardless of signal trend: **regime shift**
- If alert fires when signal crosses threshold: **degradation detection**

### 2. DATA LEAKAGE FORENSICS (CRITICAL)
- Document exactly how thresholds were determined
- Show no direct access to failure_cycle during training
- Run Monte Carlo: reshuffle failure times, see if 30% rule still holds
- If threshold is independent of failure_cycle distribution: valid

### 3. MISS ANALYSIS (HIGH)
- Study the 6 missed units (IDs: 10, 28, 125, 141, 204, 229)
- Did they fail via different mechanism?
- Why do they fail at 140-205 cycles while system expects ~250?
- Are these failure mode outliers or evidence of broken detection?

### 4. OUTLIER SENSITIVITY (HIGH)
- Compute improvement without top 20 units
- Show what happens if even 1 large unit (554-cycle failure) is test-only
- Check if result changes by >20%

### 5. PROSPECTIVE STUDY (MEDIUM)
- Declare thresholds and quality criteria NOW
- Deploy on new units with unknown failure times
- Report lead times achieved
- (This is the only true validation)

---

## BOTTOM LINE

The proof of performance is **technically accurate but scientifically weak**:

✓ **Metrics are mathematically correct** — no computational errors
✓ **Data is internally consistent** — lead times calculated properly
✗ **Results are misleading** — mean hides median, uses inflated statistics
✗ **Causality unclear** — may be detecting regime shift, not degradation
✗ **Thresholds suspicious** — 21.9% clustering at one cycle suggests overfitting or fixed rules
✗ **Validation incomplete** — no blind test, no forward validation, possible data leakage

**Recommended action:** Do not make strong claims about degradation detection until:
1. Blind validation on held-out units shows median (not mean) improvement
2. Causal analysis rules out regime shift hypothesis
3. Threshold justification documented (proves no leakage)
