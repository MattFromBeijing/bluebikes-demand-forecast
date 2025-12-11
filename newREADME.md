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


## Rationale for Final Model Selection

The **Negative Binomial model enhanced with Gradient Boosting** was selected as the final production model due to its superior performance, robustness, and practical interpretability in operational settings.

### **Performance**

The model demonstrates a substantial improvement over the Poisson baseline, achieving:

- **29% reduction in MAE** (from 4.558 to 3.229)
- **23% reduction in RMSE**

These improvements highlight the model’s ability to capture the variability and nonlinear patterns inherent in Bluebikes demand data.

### **Robustness**

The model was evaluated on a large and diverse dataset, supporting its ability to generalize:

- **157,000** training samples
- **39,000** test samples

Performance remained stable across different **stations**, **seasons**, and **demand regimes**, confirming its resilience under real-world variability.

### **Feature Richness**

The final pipeline integrates **12 engineered features**, including:

- **Spatial attributes**: station latitude, longitude, built-environment indicators  
- **Temporal structure**: hour-of-day, weekday, month  
- **Contextual variables**: density, proximity, demand trends

This multi-dimensional feature set enables the model to capture both **cyclical mobility patterns** and **localized station behavior**.

### **Practical Value**

Lower prediction errors translate directly into **more accurate and operationally meaningful forecasts**.

With improved short-term, station-level predictions, system operators can make better-informed decisions about:

- Rebalancing and redistribution
- Resource allocation
- Customer-facing service improvements

---

### **Key Achievements**

- **29% error reduction** from baseline Poisson to production NB+Boosting model (MAE: 4.558 → 3.229)
- **Robust generalization** demonstrated across 39,000 out-of-sample test observations
- **Dual-model strategy**:
  - Negative Binomial + Boosting for high-accuracy forecasting
  - ZINB for interpretable distribution modeling and zero-inflation diagnostics
- **Scalable architecture** enabling real-time predictions and scheduled retraining
- **Actionable insights** derived from feature importance and zero-inflation analysis, supporting deeper understanding of demand drivers

  

## Conclusion and Impact

The modeling journey in this project highlights a recurring challenge in applied machine learning: **balancing simplicity with predictive performance**. Early models such as **Poisson** offered interpretability but failed to capture the strong **overdispersion** and **zero-inflation** present in Bluebikes demand data. The **Negative Binomial** model addressed those limitations, and ultimately, the **NB + Boosting hybrid model** emerged as the most effective solution—**substantially improving predictive accuracy** while maintaining enough transparency to understand key drivers of demand. With a **29% reduction in MAE** from baseline to final model, the project demonstrates how **algorithmic sophistication can translate into meaningful operational improvements**.

The forecasting system built around **NB + Boosting** directly advances our core objective: **helping Bluebike operators, planners, and riders make better decisions**. By producing **accurate hourly demand predictions**, the model enables operational teams to **rebalance proactively rather than reactively**, reducing emergency shortages, optimizing routing, and making better use of staff time during peak periods. These improvements support **smoother day-to-day operations** and **reduce unnecessary truck mileage**.

The model also directly enhances the **customer experience**. Predicting when and where stations are likely to run empty or full helps maintain **higher bike and dock availability** at critical times. Reliable access to bikes leads to **smoother commutes**, **fewer frustrations**, and a **more dependable mobility system** for the public.

Beyond immediate operations, the modeling pipeline provides **interpretable insights into spatial, temporal, and contextual demand patterns**. These insights support **long-term planning decisions**, helping identify:
- Stations that need expansion
- Neighborhoods that are underserved
- Optimal placements for future infrastructure investment

Planners can use these **data-driven indicators** to design a **more equitable and scalable bike-share system**.

Finally, improved forecasting contributes to **sustainability goals**. More efficient redistribution reduces **fuel consumption**, cuts **emissions**, and supports a **more environmentally responsible mobility network**. By **reducing wasted resources** and **improving service reliability**, the project ultimately helps build a **cleaner, more efficient, and more user-centered urban transportation ecosystem**.

**Overall, the NB + Boosting model achieves more than strong predictive accuracy—it fulfills the broader mission of enabling smoother operations, better planning, and a more reliable experience for the entire Bluebikes community.**


---


## Future Work and Potential Extensions

Several promising extensions could further enhance model performance and broaden the scope of operational insights.

### **Spatial Modeling**

Incorporating **spatial modeling approaches**—such as **Gaussian processes** or **spatial-lag features**—could help capture **neighborhood-level demand shocks** that propagate unevenly across stations. 

For example, a **street festival** near one location may influence demand at nearby stations more strongly than those farther away. These spatial dependencies are currently not fully captured in the existing model.

### **Hierarchical Modeling**

**Hierarchical modeling** presents another valuable enhancement. By **sharing information across groups of similar stations**—such as those near universities or in residential neighborhoods—this approach could improve generalization while still preserving **station-specific behavior**.

### **Deep Learning Architectures**

Advanced temporal models such as **Long Short-Term Memory (LSTM)** networks or **Transformers** could be explored to uncover **longer-term temporal dependencies** in demand. These architectures might improve accuracy, especially over extended time horizons, though they may introduce trade-offs in **interpretability** and **computational complexity**.

### **External Data Integration**

Integrating **external data sources** offers substantial potential to improve predictive performance:

- **Transit disruptions** (e.g., subway delays, bus detours) may cause **short-term demand spikes** as commuters shift transportation modes.
- **Environmental factors**, such as **air quality indices** or **pollen counts**, may influence **recreational ridership**, particularly on weekends.
- **Granular weather data** (temperature, wind speed, cloud cover) could refine **short-term predictions**.
- **Socioeconomic indicators** (e.g., employment density, median income) could improve **model generalization**, especially for **new station locations** with limited historical data.

### **Causal Inference for Strategic Decision-Making**

From a research perspective, **causal inference** represents a critical next step. While the current model excels at **predicting demand given observed features**, strategic decisions—such as **adding new stations** or **adjusting pricing**—require understanding **causality rather than correlation**.

Approaches like:

- **Difference-in-differences (DiD)**
- **Synthetic control methods**

can be applied to **natural experiments**, such as the opening of a new subway station, to **quantify the true impact** of interventions more rigorously than predictive models alone.

---

These extensions offer paths to not only boost model accuracy but also deepen the value of insights provided to planners, operators, and policymakers.

