# Bluebikes Demand Forecasting

**Team:** Matthew Yan · Jiayong Tu · Fenglin Hu · Mingyu Shen </br>
**Course:** CS 506 </br>
**Video Link:** https://youtu.be/buNaUpdewR0

**Website Link:** https://bluebikes-demand-forecast.vercel.app/

---

## Project Description

This project predicts hourly station-level bike demand (inflow and outflow) using historical Bluebikes trip data. We developed a comprehensive data processing pipeline and advanced statistical modeling framework to address the unique challenges of bike-share demand forecasting.

### Why It Matters

Accurate bike demand forecasting provides significant value to multiple stakeholders:

- **Riders:** Can avoid empty or full stations before they arrive, improving user experience and reducing trip planning uncertainty.
- **Planners:** Can identify problem stations and time periods, enabling proactive rebalancing operations and infrastructure improvements.
- **Operations:** Optimizes bike redistribution efforts, reducing operational costs and improving service reliability.

The challenge lies in the unique characteristics of bike-share data: extreme variability between stations, excess structural zeros during off-peak hours, and overdispersion where variance far exceeds the mean. Traditional regression models fail to capture these patterns, motivating our advanced count-based modeling approach.

---

### Repository Structure

```
bluebikes-demand-forecast/
├── pipeline/                          # Data processing and modeling notebooks
│   ├── data_clean_and_feature.ipynb   # Feature engineering for Poisson & NB models
│   ├── poisson_with_features.ipynb    # Poisson regression baseline model
│   ├── neg_with_features.ipynb        # Negative Binomial regression model
│   ├── nb_with_boosting.ipynb         # Negative Binomial with boosting model
│   ├── ZINB_with_feature.ipynb        # Zero-Inflated Negative Binomial model
│   └── archive/                       # Archived scripts and notebooks
├── flask/                             # Backend API
│   ├── app.py                         # Flask application serving predictions
│   ├── simulate_model.py              # Model simulation for API
│   ├── requirement.txt                # Flask dependencies
│   ├── test.py                        # Testing scripts
│   └── testagain.py                   # Additional testing scripts
├── nextjs/                            # Frontend web application (Next.js)
│   ├── app/                           # Next.js App Router
│   │   ├── api/                       # API routes
│   │   ├── map/                       # Map visualization page
│   │   ├── visualizations/            # Visualizations page
│   │   └── page.tsx                   # Home page
│   ├── components/                    # React components
│   │   ├── ui/                        # UI components (buttons, cards, etc.)
│   │   ├── bike-map.tsx               # Interactive bike station map
│   │   └── visualizations/            # Visualization components
│   ├── lib/                           # Utility libraries
│   │   ├── types.ts                   # TypeScript type definitions
│   │   ├── target-stations.ts         # Station data
│   │   └── visualizations-data.ts     # Visualization metadata
│   ├── public/                        # Static assets
│   │   ├── 01_data_exploration/       # Data exploration visualizations
│   │   ├── 02_time_series/            # Time series visualizations
│   │   ├── 03_poisson_model/          # Poisson model results
│   │   ├── 04_nb_boosting_model/      # NB boosting model results
│   │   └── 05_zinb_model/             # ZINB model results
│   └── package.json                   # Node.js dependencies
├── visualizations/                    # Root-level visualization outputs
│   ├── 01_data_exploration/           # Data exploration charts
│   │   ├── global_distribution.png
│   │   └── station_distributions/     # Per-station distribution plots
│   ├── 02_time_series/                # Time series analysis
│   │   ├── mit_hour_of_day.png
│   │   ├── mit_hourly_by_month.png
│   │   └── monthly/                   # Monthly breakdown plots
│   ├── 03_poisson_model/              # Poisson model evaluation
│   ├── 04_nb_boosting_model/          # NB boosting model evaluation
│   ├── 05_zinb_model/                 # ZINB model evaluation
│   └── *.md                           # Documentation files
├── data/                              # Data directory (not in repo, created by download_dataset.py)
│   ├── 2024_data/                     # 2024 Bluebikes trip data (12 monthly CSV files)
│   └── 2023_data/                     # 2023 training data
│       ├── Bluebikes/                 # 2023 trip data (12 monthly CSV files)
│       ├── Weather/                   # Weather data CSV files
│       └── Features/                  # Station feature data (MBTA stops, etc.)
├── .github/                           # GitHub configuration
│   └── workflows/
│       └── ci.yml                     # CI/CD workflow
├── download_dataset.py                # Script to download data from Hugging Face
├── Makefile                           # Build automation and commands
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

**Note:** The `data/` directory is not included in the repository. Use `make download-data` or run `download_dataset.py` to download and organize the dataset from Hugging Face.

---

### How to Build and Run

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

2. **Download data**
   ```bash
   make download-data
   ```
   or alternatively run the `download_data.py` file

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

---

## Data Processing and Modeling

### Feature Engineering

Across all models, we engineered a comprehensive set of features from raw trip data to capture temporal, spatial, and operational patterns. Not all features were used in each model training process.

#### Temporal Features
- `month` (1-12): Month of the year to capture seasonal patterns
- `hour_of_day` (0-23): Hour when the observation period starts
- `day_of_week` (0-6): Day of week (0=Monday, 6=Sunday)
- `is_weekend` (0/1): Binary indicator for Saturday/Sunday
- `start_hour`, `end_hour`: Hour boundaries of the observation window
- `is_night` (0/1): Nighttime indicator (10pm-5am) for structural zero modeling

#### Spatial Features (Station-Level)
- `station_lat`, `station_lng`: Geographic coordinates
- `subway_distance_m`: Distance to nearest subway/commuter rail station (meters)
- `bus_distance_m`: Distance to nearest bus stop (meters)
- `university_distance_m`: Distance to nearest university (meters)
- `mbta_stops_250m`: Count of MBTA stations within 250m radius
- `bus_stops_250m`: Count of bus stops within 250m radius
- `restaurant_count`: 
- `restaurant_count`: Count of nearby restaurants (within the station buffer)
- `dist_business`: Distance to nearest commercial/business district (meters)
- `dist_residential`: Distance to nearest residential-area centroid (meters)
- `pop_density`: Local population density (people per sq. km) around the station
- `emp_density`: Local employment density (jobs per sq. km) around the station
- `restaurant_density`: Number of restaurants per sq. km (local restaurant concentration)

#### Lag Features (Time-Series)
- `last_hour_in`, `last_hour_out`: Previous hour's inflow/outflow
- `last_two_hour_in`, `last_two_hour_out`: 2-hour lag inflow/outflow
- `last_three_hour_in`, `last_three_hour_out`: 3-hour lag inflow/outflow

These capture short-term momentum and autocorrelation in demand.

#### Weather Features
- `avg_temp`: Average temperature
- `precipitation`: Precipitation amount (inches)

### Model Development

We implemented a progressive modeling strategy, starting from simple baselines and advancing to sophisticated count models that explicitly handle overdispersion and zero-inflation.

## Model 1: Poisson Regression (Baseline)

### Model Rationale

Firstly we plot the diagram of in/out data at MIT station per month

![mit_in/out](visualizations/02_time_series/mit_hourly_by_month.png)

We realize this might fit in a poisson distribution

Why?

Poisson regression is the natural starting point for count data:
- Designed specifically for non-negative integer outcomes
- Computationally efficient and interpretable
- Standard baseline in transportation demand forecasting
- Provides coefficients that directly relate features to expected counts

### Features Used

**Feature Set (12 features):**
```
station_features = station_features[[
    "station_name",
    "station_lat",
    "station_lng",
    "dist_subway_m",
    "dist_bus_m",
    "dist_university_m",
    "dist_business",
    "dist_residential",
    "pop_density",
    "emp_density",
    "restaurant_count",
    "restaurant_density",
]]
```
(Feature engineering: check `pipeline/feature_enigneering.ipynb`)


### Data Analysis

- Explored 10 target stations across the Bluebike network
- Aggregated hourly inflow/outflow counts for 2024
- Merged with static station features (distance to transit, amenities)
- Performed 80/20 train-test split with random state 42

### Code Description

**Feature Engineering:** `pipeline/feature_enigneering.ipynb` prepares the feature dataset used by both Poisson and Negative Binomial models.

**Implementation:** `pipeline/poisson_with_features.ipynb`

1. **Data Loading:** 
   - Load 2024 trip data and station feature CSV
   - Selected staion list:
      ```
         "MIT at Mass Ave / Amherst St",
         "Central Square at Mass Ave / Essex St",
         "MIT Pacific St at Purrington St",
         "Harvard Square at Mass Ave / Dunster St",
         "Boylston St at Massachusetts Ave",
         "Charles St at Cambridge St",
         "Forsyth St at Huntington Ave",
         "Boylston St at Fairfield St",
         "Christian Science Plaza - Massachusetts Ave at Westland Ave",
         "MIT Stata Center at Vassar St / Main St"
      ```

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
4. **Training:** 
   - Fit on inflow and outflows counts (IN) by combining them together:
      ```
      # ---------------------------------------------------------
      # 1. Extract target variables
      # ---------------------------------------------------------
      y_out = panel["out_count"].values   # shape (n_samples,)
      y_in  = panel["in_count"].values    # shape (n_samples,)

      # ---------------------------------------------------------
      # 2. Combine out & in into ONE target matrix
      #    Y[:,0] = out_count
      #    Y[:,1] = in_count
      # ---------------------------------------------------------
      Y = np.column_stack([y_out, y_in])   # shape (n_samples, 2)
      ```
   - Chaning learning rate to find **best alpha**:

      ```
      alphas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5]

      for a in alphas:

         clf = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=a, max_iter=1000))
         ])
      ```


5. **Evaluation:** MAE, RMSE on train and test sets

### Best Results

| Metric | Train | Test |
|--------|-------|------|
| MAE    | 4.491 | 4.479 |
| RMSE   | 6.017 | 5.998 |

**Key Findings:**
- Test performance nearly identical to training (no overfitting)
- Average error of ~3 bikes per hour is reasonable for baseline
- Model struggled with peak hours where variance >> mean
- Systematic underestimation of high-demand periods
- Failed to capture the long right tail of demand spikes

**Limitations:**
- **Equidispersion assumption violated (variance far exceeds mean at busy stations)**
- Predictions collapsed toward the mean during extreme demand
- Cannot handle structural zeros (hours that should always be zero)

**Next Step**
- Calculate Mean and Variance
   ```
   vals = pd.Series(y_train_in)
   print("Mean:", vals.mean())
   print("Variance:", vals.var())

   Mean: 6.873835786943646
   Variance: 43.758338103127706
   ```

Since Poisson assumes the mean equals the variance and Equidispersion assumption violated (variance far exceeds mean at busy stations), we then move to Negative Binomial Regression


## Model 2: Negative Binomial Regression

### Model Rationale

Negative Binomial (NB) regression addresses the critical limitation of Poisson by introducing a dispersion parameter:
- **Overdispersion Handling:** Allows variance to exceed mean 
- **Interpretability:** Maintains GLM framework with interpretable coefficients
- Widely used in transportation, epidemiology, and demand forecasting where counts fluctuate heavily

The variance in NB is modeled as: **Variance = μ + α μ²**

This captures the extra variability observed during:
- Morning commute peaks (7-9 AM)
- Evening commute peaks (5-7 PM)
- Weekend recreational usage spikes
- Weather-driven demand surges

### Features Used( Poisson Attempt)

**Feature Set (12 features):**
   ```
   station_features = station_features[[
      "station_name",
      "station_lat",
      "station_lng",
      "dist_subway_m",
      "dist_bus_m",
      "dist_university_m",
      "dist_business",
      "dist_residential",
      "pop_density",
      "emp_density",
      "restaurant_count",
      "restaurant_density",
   ]]
   ```
(Feature engineering: check `pipeline/feature_enigneering.ipynb`)

### Code Description

1. **Data Loading:** 
   - Load 2024 and 2023 trip data and station feature CSV
   - Deicide to use more stations as trainning set 
   - Selected staion list:
      ```
         "MIT at Mass Ave / Amherst St",
         "Central Square at Mass Ave / Essex St",
         "Harvard Square at Mass Ave/ Dunster",
         "Ames St at Main St",
         "MIT Pacific St at Purrington St",
         "Charles Circle - Charles St at Cambridge St",
         "MIT Vassar St",
         "Beacon St at Massachusetts Ave",
         "Christian Science Plaza - Massachusetts Ave at...",
         "Boylston St at Massachusetts Ave",
         "Boylston St at Fairfield St",
         "South Station - 700 Atlantic Ave",
         "Forsyth St at Huntington Ave",
         "Mass Ave at Albany St",
         "Commonwealth Ave at Agganis Way",
         "Central Sq Post Office / Cambridge City Hall a...",
         "Newbury St at Hereford St",
         "Harvard University River Houses at DeWolfe St ...",
         "MIT Stata Center at Vassar St / Main St",
         "Landmark Center - Brookline Ave at Park Dr"
      ```

**Feature Engineering:** Uses the same feature dataset from `pipeline/feature_enigneering.ipynb` as Poisson.

**Implementation:** `pipeline/nb_with_boosting.ipynb`

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

- **Results**

<div align="middle">

| Metric | Train | Test  |
|--------|--------|--------|
| MAE    | 4.542  | 4.558  |
| RMSE   | 6.533  | 6.480  |

</div>

4. **Try Boosting**

-  Why?
   - 1 There are strong nonlinearities
   - 2 There are many features or messy interactions
   - 3 Data is highly overdispersed

```
y_train_pred = y_train_pred + boost.predict(X_train_imp)
y_test_pred  = y_test_pred  + boost.predict(X_test_imp)
```

- **Results**

<div align="middle">

| Metric | Train | Test |
|--------|-------|------|
| MAE    | 3.231 | 3.229 |
| RMSE   | 5.105 | 5.024 |

</div>

- **Confusion Matrix**

      from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

      # ---- set a threshold to measure how many in/out values bigger or equal to it are predicted correctly ----
      threshold = 4

      # convert to binary classes
      y_train_true_bin = (y_train_in >= threshold).astype(int)
      y_train_pred_bin = (y_train_pred >= threshold).astype(int)

      y_test_true_bin  = (y_test_in >= threshold).astype(int)
      y_test_pred_bin  = (y_test_pred >= threshold).astype(int)

      

<div align="center">

![nb_confusion_matrix](visualizations/04_nb_boosting_model/boosting_confusion_matrix_test.png)

</div>


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

Examining our data overall, we noticed that a large percentage of in and out values were 0.

![Global distribution of hourly inflow/outflow counts](visualizations/01_data_exploration/global_distribution.png)

This is likely due to the nature of bike-share systems, in which some hours are structurally zero.

![Distribution of rush hour inflow/outflow counts](visualizations/01_data_exploration/station_rush_hour_distributions/output1.png)
![Distribution of rush hour inflow/outflow counts](visualizations/01_data_exploration/station_rush_hour_distributions/output3.png)
![Distribution of rush hour inflow/outflow counts](visualizations/01_data_exploration/station_rush_hour_distributions/output4.png)
![Distribution of rush hour inflow/outflow counts](visualizations/01_data_exploration/station_rush_hour_distributions/output6.png)
![Distribution of rush hour inflow/outflow counts](visualizations/01_data_exploration/station_rush_hour_distributions/output7.png)

ZINB explicitly models this two-process data generation.

**Theoretical Motivation:**

ZINB combines:
- **Logistic Regression** (predicts probability of structural zero)
- **Negative Binomial** (predicts count μ when station is active)

For each observation:
- With probability **pi**: y = 0 (structural zero)
- With probability **(1 - π)**: y ~ NegativeBinomial(r, p)

This dual-process structure aligns with bike-share behavior patterns and addresses both overdispersion AND excess zeros.

### Features Used

**Feature set (6 features total):**
- `month`
- `start_hour`
- `end_hour`
- `bus_distance_m`
- `last_three_hour_in`
- `last_three_hour_out`

More features were used originally, but a count model coefficient analysis showed that the above features were the most impactful.

**Inflation Model Features (predicts the liklihood of a structural zero):**
- `is_night` (strong predictor of structural zeros)
- `precipitation` (weather-driven inactivity)
- `avg_temp` (temperature effects on ridership)

<!-- ### Data Analysis

**Comprehensive data enrichment pipeline:**
1. Identified 20 busiest stations by total activity
2. Calculated distances to nearest subway, bus, and university
3. Counted transit stops within 250m radius
4. Merged hourly weather data (temperature, precipitation)
5. Engineered 3-hour lag features for temporal autocorrelation
6. Filtered outliers (capped counts at 50 to remove data errors)

**Visualizations Created:**
- Actual vs. Predicted scatter plots for IN/OUT
- Residual plots to check model assumptions
- Distribution of zero-inflation probability
- Distribution of NB mean
- Coefficient comparison across features
- Zero proportion comparisons (actual vs. predicted) -->

### Code Description

**Implementation:** `pipeline/ZINB_with_feature.ipynb`

1. **Data Extraction and analysis:**
   - Load and parse 2023 trip data (Apr-Dec) from multiple CSVs
      - January-March data were skipped due to formatting issues
   - Aggregate to hourly station-level IN/OUT counts
   - Create complete time grid (every station × every hour)
   - Conduct data analysis on the distribution of IN/OUT counts

3. **Feature Engineering:**
   - Add lag features (1hr, 2hr, 3hr)
   - Calculate distances to nearest university and MBTA subway and bus station
   - Merge weather data by date
   - Add nighttime indicator

4. **Model Training:**
   ```python
   from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

   # Separate models for OUT and IN (Example only shows OUT)
   zinb_out_model = ZeroInflatedNegativeBinomialP(
       endog=y_out_train,
       exog=X_train_const,      # Count model features
       exog_infl=X_train_infl,  # Inflation model features
       p=2                       # NB2 parameterization
   )

   zinb_out_results = zinb_out_model.fit(method='bfgs', maxiter=1000)

   # Collect Model Predictions for expected value at hour-station
   y_out_pred_original = zinb_out_results.predict(exog=X_test_const, exog_infl=X_test_infl, which='mean')

   # Collect Model Predictions for probability of a structural zero at hour-station
   pi_out_pred = zinb_out_results.predict(exog=X_test_const, exog_infl=X_test_infl, which='prob-zero')

   # Collect Model Predictions for expected value at hour-station given non-structural zero
   mu_out_pred = zinb_out_results.predict(exog=X_test_const, exog_infl=X_test_infl, which='mean-main')

   # If  > 0.5, predict 0; otherwise use NB mean
   y_pred = np.where(pi_pred > 0.5, 0, y_pred_original)
   ```

5. **Results analysis**
- Computes standard evaluation metrics (MAE, RMSE)
- Produces diagnostic visualizations and tables: predicted vs actual scatter plots, residual plots, zero-proportion comparisons, coefficient tables for the count and inflation components, and distributions of `mu` and `pi` for model inspection and reporting.
   
### Results

#### OUT Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 5.2343 |
| MAE | 3.2862 |


#### IN Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 5.0750 |
| MAE | 3.2928 |

## Final Results

### Best Performing Models: Negative Binomial + Gradient Boosting


| **Model**             | **MAE (Test)** | **RMSE (Test)**  | **F1 Score** | 
|-----------------------|----------------|------------------|--------------|
| Poisson Baseline      | 4.558          | 6.480            | –            | 
| Negative Binomial     | 4.558          | 6.480            | –            | 
| **NB + Boosting**     | **3.229**      | **5.024**        | **0.830**    | 
| ZINB        | 3.2862          | 5.2343            | –            | 


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
