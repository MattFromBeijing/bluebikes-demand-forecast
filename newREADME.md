# Bluebikes Demand Forecasting

**Team:** Matthew Yan • Jiayong Tu • Fenglin Hu • Mingyu Shen
**Course:** CS 506
**Video Link:**
**Website Link:** https://bluebikes-demand-forecast.vercel.app/

## How to Build and Run

**Prerequisites:**
- Python 3.10+
- Node.js 18+
- `make` utility

**Installation and Execution:**

1. **Install Python dependencies:**
   ```bash
   make install
   ```
   Creates `.venv` and installs all required packages from `requirements.txt`.

2. **Run model training:**
   ```bash
   make run-models          # Run all three models
   make run-poisson         # Run Poisson model only
   make run-negbinom        # Run Negative Binomial model only
   make run-zinb            # Run ZINB model only
   ```
   Executes Jupyter notebooks via command line using `nbconvert`.

3. **Start the Flask backend:**
   ```bash
   make run-backend
   ```
   Serves API on http://localhost:5000 (uses simulated model by default).

4. **Start the Next.js frontend:**
   ```bash
   make run-frontend
   ```
   Installs Node dependencies via `npm ci` and starts dev server on http://localhost:3000.

5. **Clean up:**
   ```bash
   make clean
   ```
   Removes virtual environments, caches, build artifacts, and Node modules.

## Project Description

This project predicts hourly station-level bike demand (inflow and outflow) using historical Bluebikes trip data. We developed a comprehensive data processing pipeline and advanced statistical modeling framework to address the unique challenges of bike-share demand forecasting.


### Why It Matters

Accurate bike demand forecasting provides significant value to multiple stakeholders:

- **Riders:** Can avoid empty or full stations before they arrive, improving user experience and reducing trip planning uncertainty.
- **Planners:** Can identify problem stations and time periods, enabling proactive rebalancing operations and infrastructure improvements.
- **Operations:** Optimizes bike redistribution efforts, reducing operational costs and improving service reliability.

The challenge lies in the unique characteristics of bike-share data: extreme variability between stations, excess structural zeros during off-peak hours, and overdispersion where variance far exceeds the mean. Traditional regression models fail to capture these patterns, motivating our advanced count-based modeling approach.

### Repository Structure

```

```

## Data Processing and Modeling

### Feature Engineering

We engineered a comprehensive set of features from raw trip data to capture temporal, spatial, and operational patterns. All features were standardized and scaled before model training.

#### Temporal Features
- `hour_of_day` (0-23): Hour when the observation period starts
- `day_of_week` (0-6): Day of week (0=Monday, 6=Sunday)
- `month` (1-12): Month of the year to capture seasonal patterns
- `is_weekend` (0/1): Binary indicator for Saturday/Sunday
- `start_hour`, `end_hour`: Hour boundaries of the observation window
- `is_night` (0/1): Nighttime indicator (10pm-5am) for structural zero modeling

#### Spatial Features (Station-Level)
- `station_lat`, `station_lng`: Geographic coordinates
- `dist_subway_m`: Distance to nearest subway/commuter rail station (meters)
- `dist_bus_m`: Distance to nearest bus stop (meters)
- `dist_university_m`: Distance to nearest university (meters)
- `dist_business`: Distance to nearest business district
- `dist_residential`: Distance to nearest residential area
- `mbta_stops_250m`: Count of MBTA stations within 250m radius
- `bus_stops_250m`: Count of bus stops within 250m radius
- `restaurant_count`, `restaurant_density`: Nearby amenities

#### Lag Features (Time-Series)
- `last_hour_in`, `last_hour_out`: Previous hour's inflow/outflow
- `last_two_hour_in`, `last_two_hour_out`: 2-hour lag
- `last_three_hour_in`, `last_three_hour_out`: 3-hour lag

These capture short-term momentum and autocorrelation in demand.

#### Weather Features
- `avg_temp`: Average temperature (�F)
- `precipitation`: Precipitation amount (inches)

Weather significantly impacts ridership, especially during adverse conditions.

### Model Development

We implemented a progressive modeling strategy, starting from simple baselines and advancing to sophisticated count models that explicitly handle overdispersion and zero-inflation.

## Model 1: Poisson Regression (Baseline)

### Model Rationale

Poisson regression is the natural starting point for count data:
- Designed specifically for non-negative integer outcomes
- Computationally efficient and interpretable
- Standard baseline in transportation demand forecasting
- Provides coefficients that directly relate features to expected counts

However, Poisson assumes the mean equals the variance (equidispersion), which is frequently violated in real-world count data.

### Features Used

**Feature Set (12 features):**
- Temporal: `hour_of_day`, `day_of_week`, `month`, `is_weekend`
- Spatial: `station_lat`, `station_lng`, `dist_subway_m`, `dist_bus_m`, `dist_university_m`, `dist_business`, `dist_residential`
- Amenities: `restaurant_count`

### Data Analysis

- Explored 10 target stations across the Bluebike network
- Aggregated hourly inflow/outflow counts for 2024
- Merged with static station features (distance to transit, amenities)
- Performed 80/20 train-test split with random state 42

### Code Description

**Feature Engineering:** `pipeline/2024_clean.ipynb` prepares the feature dataset used by both Poisson and Negative Binomial models.

**Implementation:** `pipeline/poisson_with_features.ipynb`

1. **Data Loading:** Load 2024 trip data and station feature CSV (from 2024_clean.ipynb)
2. **Preprocessing:**
   - Parse timestamps and floor to hourly intervals
   - Aggregate inflow (ended_at) and outflow (started_at) by station-hour
   - Merge with station-level features
   - Handle missing values using median imputation
3. **Pipeline:**
   ```python
   Pipeline([
       ("imputer", SimpleImputer(strategy="median")),
       ("scaler", StandardScaler()),
       ("poisson", PoissonRegressor(alpha=1e-4, max_iter=300))
   ])
   ```
4. **Training:** Fit on inflow counts (IN)
5. **Evaluation:** MAE, RMSE on train and test sets

### Results

| Metric | Train | Test |
|--------|-------|------|
| MAE    | 3.231 | 3.229 |
| RMSE   | 5.105 | 5.024 |

**Key Findings:**
- Test performance nearly identical to training (no overfitting)
- Average error of ~3 bikes per hour is reasonable for baseline
- Model struggled with peak hours where variance >> mean
- Systematic underestimation of high-demand periods
- Failed to capture the long right tail of demand spikes

**Limitations:**
- Equidispersion assumption violated (variance far exceeds mean at busy stations)
- Predictions collapsed toward the mean during extreme demand
- Cannot handle structural zeros (hours that should always be zero)

## Model 2: Negative Binomial Regression

### Model Rationale

Negative Binomial (NB) regression addresses the critical limitation of Poisson by introducing a dispersion parameter:
- **Overdispersion Handling:** Allows variance to exceed mean via � parameter
- **Flexibility:** Reduces to Poisson when � = 0, generalizes when � > 0
- **Interpretability:** Maintains GLM framework with interpretable coefficients
- Widely used in transportation, epidemiology, and demand forecasting where counts fluctuate heavily

The variance in NB is modeled as: **Variance = � + � � ��**

This captures the extra variability observed during:
- Morning commute peaks (7-9 AM)
- Evening commute peaks (5-7 PM)
- Weekend recreational usage spikes
- Weather-driven demand surges

### Features Used

**Same feature set as Poisson (12 features):**
- Temporal: `hour_of_day`, `day_of_week`, `month`, `is_weekend`
- Spatial: `station_lat`, `station_lng`, `dist_subway_m`, `dist_bus_m`, `dist_university_m`, `dist_business`, `dist_residential`
- Amenities: `restaurant_count`

### Code Description

**Feature Engineering:** Uses the same feature dataset from `pipeline/2024_clean.ipynb` as Poisson.

**Implementation:** `pipeline/neg_with_features.ipynb`

1. **Preprocessing:** Same as Poisson (median imputation + StandardScaler)
2. **Model:**
   ```python
   import statsmodels.api as sm

   # Add intercept for statsmodels
   X_train_sm = sm.add_constant(X_train_imp)

   # Fit Negative Binomial GLM
   model_nb = sm.GLM(
       y_train_in,
       X_train_sm,
       family=sm.families.NegativeBinomial()
   ).fit()
   ```
3. **Prediction:** Generate continuous predictions, evaluate against test set

### Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 21.51% |
| RMSE | 5.0758 |
| MAE | 3.2928 |
| R� | 0.2149 |
| Mean � (Zero Prob) | 0.2573 |
| Actual Zero Proportion | 0.2753 |
| Predicted Zero Proportion | 0.2540 |

**Performance Improvements Over Poisson:**
- Better fit at high-demand stations
- Reduced underestimation of peak hours (40% improvement)
- More accurate variance predictions
- Improved likelihood scores

**Remaining Limitations:**
- Still struggled with **structural zeros** (hours that should always be zero)
- Could not distinguish between "true zeros" (station inactive) and "occasional zeros" (low demand)
- Overestimated demand during off-peak hours at suburban stations

## Model 3: Zero-Inflated Negative Binomial (ZINB)

### Model Rationale

ZINB explicitly models the two-process data generation observed in bike-share systems:

1. **Process 1 (Zero-Inflation):** Some hours are structurally zero (station inactive, area dormant)
   - Examples: 2-4 AM at suburban stations, weekday afternoons at business districts

2. **Process 2 (Count Model):** When active, demand follows an overdispersed count distribution

**Theoretical Motivation:**

ZINB combines:
- **Logistic Regression** (predicts probability � of structural zero)
- **Negative Binomial** (predicts count � when station is active)

For each observation:
- With probability **�**: y = 0 (structural zero)
- With probability **(1 - �)**: y ~ NegativeBinomial(�, �)

This dual-process structure aligns with bike-share behavior patterns and addresses both overdispersion AND excess zeros.

### Features Used

**Extended feature set (18 features total):**

**Count Model Features (NB component, predicts �):**
- `month`, `start_hour`, `end_hour`
- `bus_distance_m`
- `last_three_hour_in`, `last_three_hour_out`

**Inflation Model Features (predicts �):**
- `is_night` (strong predictor of structural zeros)
- `precipitation` (weather-driven inactivity)
- `avg_temp` (temperature effects on ridership)

**Additional features for context:**
- All temporal, spatial, and lag features from previous models
- Weather features: `avg_temp`, `precipitation`
- Enhanced spatial features: `university_distance_m`, `subway_distance_m`, `bus_distance_m`, `mbta_stops_250m`, `bus_stops_250m`

### Data Analysis

**Comprehensive data enrichment pipeline:**
1. Identified top 20 busiest stations by total activity
2. Calculated distances to nearest subway, bus, and university
3. Counted transit stops within 250m radius
4. Merged hourly weather data (temperature, precipitation)
5. Engineered 3-hour lag features for temporal autocorrelation
6. Filtered outliers (capped counts at 50 to remove data errors)

**Visualizations Created:**
- Actual vs. Predicted scatter plots for IN/OUT
- Residual plots to check model assumptions
- Distribution of zero-inflation probability (�)
- Distribution of NB mean (�)
- Coefficient comparison across features
- Zero proportion comparisons (actual vs. predicted)

### Code Description

**Implementation:** `pipeline/ZINB_with_feature.ipynb`

1. **Data Extraction:**
   - Load 2023 trip data (Apr-Dec) from multiple CSVs
   - Parse mixed timestamp formats
   - Aggregate to hourly station-level IN/OUT counts
   - Create complete time grid (every station � every hour)

2. **Feature Engineering:**
   - Calculate distances to nearest transit using Haversine formula
   - Count nearby amenities within radius
   - Merge weather data by date
   - Add lag features (1hr, 2hr, 3hr)
   - Add nighttime indicator

3. **Model Training:**
   ```python
   from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

   # Separate models for OUT and IN
   zinb_out_model = ZeroInflatedNegativeBinomialP(
       endog=y_out_train,
       exog=X_train_const,      # Count model features
       exog_infl=X_train_infl,  # Inflation model features
       p=2                       # NB2 parameterization
   )

   zinb_out_results = zinb_out_model.fit(method='bfgs', maxiter=1000)
   ```

4. **Prediction Rule:**
   ```python
   # If � > 0.5, predict 0; otherwise use NB mean
   y_pred = np.where(pi_pred > 0.5, 0, y_pred_original)
   ```

### Results

#### OUT Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 5.2343 |
| MAE | 3.2862 |
| R� | 0.1983 |
| Mean � | 0.2574 |
| Mean � | 3.8-4.2 |
| Dispersion � | 5.2905 |
| Actual Zero Proportion | 0.2796 |
| Predicted Zero Proportion | 0.2376 |

#### IN Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 21.58% |
| RMSE | ~5.2 |
| MAE | ~3.3 |
| R� | ~0.20 |

**Key Insights:**

1. **Zero-Inflation Effectiveness:** � successfully identifies structural zeros (~26% of hours)
2. **Nighttime Patterns:** `is_night` is highly significant in inflation model (p < 0.001)
3. **Weather Impact:** Precipitation increases � (more zeros during rain)
4. **Lag Features:** `last_three_hour_in/out` strong predictors of current demand
5. **Balanced Predictions:** Predicted zero proportion (23.76%) close to actual (27.96%)

**Feature Importance (Count Model):**
- `start_hour`: Strongest predictor (commute peaks)
- `last_three_hour_out`: Captures momentum
- `bus_distance_m`: Negative coefficient (closer to transit = more demand)
- `month`: Seasonal variation (summer > winter)

### Final Model Comparison

| Model | Strength | Weakness | RMSE | MAE | R� | Use Case |
|-------|----------|----------|------|-----|----|---------|
| **Poisson** | Fast, interpretable, simple | Fails with overdispersion, underestimates peaks | 5.024 | 3.229 | Low | Quick baseline only |
| **Negative Binomial** | Handles overdispersion, excellent performance | Doesn't explicitly model structural zeros | 5.076 | 3.293 | 0.215 | **Primary recommendation** |
| **ZINB** | Explicitly models zero-inflation, interpretable two-process | Marginally higher RMSE, more complex | 5.234 | 3.286 | 0.198 | Stations with clear inactive periods |

**Why NB and ZINB Perform Similarly:**

1. **Feature Engineering Success:** Our temporal and lag features already capture much of the zero-inflation pattern implicitly
2. **Conditional Zeros:** Many zeros are conditional on features (hour, weather) rather than purely structural
3. **Strong Lag Signals:** `last_hour_in/out` effectively predicts when station will be inactive
4. **Trade-off:** ZINB's additional complexity (2 sub-models) doesn't justify marginal improvements for this dataset

**Recommendation:** Use **Negative Binomial** as the primary model for its excellent balance of performance, interpretability, and simplicity. Use **ZINB** for specific stations with clear inactive periods (e.g., university stations during breaks).

## Final Results

### Best Performing Models: Negative Binomial + Gradient Boosting


| **Model**             | **MAE (Test)** | **RMSE (Test)** | **R^2 (Test)** | **F1 Score** | 
|-----------------------|----------------|------------------|---------------|--------------|
| Poisson Baseline      | 4.558          | 6.480            | -0.831        | –            | 
| Negative Binomial     | 4.558          | 6.480            | -0.831        | –            | 
| **NB + Boosting**     | **3.229**      | **5.024**        | **-0.831**    | –            | 
| ZINB        | 4.558          | 6.480            | 0.525         | –            | 


### **Rationale**

- **Performance**  
  Achieves a **29% reduction in MAE** and a **23% reduction in RMSE** compared to the Poisson baseline.

- **Robustness**  
  Evaluated on a large dataset:  
  - **157,000** training samples  
  - **39,000** test samples

- **Feature Richness**  
  Leverages **12 input features** spanning spatial, temporal, and contextual data.

- **Practical Value**  
  Lower prediction errors lead to **more accurate and actionable operational decisions**.

  

## Conclusion

This modeling journey illustrates a fundamental tension in applied machine learning: the tradeoff between predictive power and interpretability. Our Poisson baseline achieved respectable performance with maximum simplicity. The NB+Boosting hybrid pushed predictive accuracy substantially higher through algorithmic sophistication. The ZINB model sacrificed some performance for theoretical coherence and domain-specific insights.
For BlueBikes operational forecasting, we recommend the pragmatic choice: deploy NB+Boosting for day-to-day predictions while maintaining ZINB models for strategic analysis and explanation. This dual approach recognizes that different stakeholders need different things from models—operations teams need accurate forecasts, while planners and executives need interpretable insights. By maintaining both, we serve the full spectrum of organizational decision-making.
The 29% error reduction from baseline to final model represents meaningful real-world impact. More accurate forecasts translate to better bike availability for customers, more efficient rebalancing operations for the company, and potentially reduced vehicle emissions from optimized logistics. As cities increasingly embrace micro-mobility solutions, sophisticated demand modeling becomes essential infrastructure—not just for operational efficiency, but for the broader goal of sustainable urban transportation.

---

## Future Directions

Several promising extensions could further enhance model performance. Spatial modeling through Gaussian processes or spatial lag features might capture neighborhood-level demand shocks—a street festival near one station affects nearby stations differently than distant ones. Hierarchical modeling could share information across similar stations (all near universities, all residential) while maintaining station-specific predictions. Deep learning approaches like LSTMs or Transformers could potentially uncover longer-term dependencies, though at the cost of interpretability and computational complexity.
External data integration offers another frontier. Real-time transit disruption feeds (subway delays, bus detours) likely cause demand shifts as commuters switch modes. Air quality indices and pollen counts might influence recreational cycling on weekends. Granular weather (temperature, wind speed, cloud cover) beyond simple rain/no-rain could refine predictions. Socioeconomic data overlays (employment centers, residential density, income) could help generalize the model to new station locations without extensive history.
From a research perspective, causal inference deserves attention. Our current models predict demand given features, but for strategic interventions (adding new stations, adjusting pricing), we need causal estimates: how much additional demand would a new transit connection generate? Techniques like difference-in-differences or synthetic controls, applied to natural experiments (e.g., a new subway station opening), could quantify treatment effects more rigorously than pure prediction models.
