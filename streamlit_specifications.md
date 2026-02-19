
# Streamlit Application Specification: AutoML for Trading Signal Discovery Audit

## 1. Application Overview

### Purpose of the Application

This Streamlit application serves as an interactive blueprint for "Sri Krishnamurthy, CFA Charterholder and Investment Analyst at AlphaQuant Capital," to explore and rigorously audit alpha-generating investment signals discovered through Automated Machine Learning (AutoML). The application highlights the power of AI in accelerating signal discovery while emphasizing the critical need for human diligence and a "four-check" overfitting audit to safeguard against spurious correlations in financial data. It aims to empower financial professionals by demonstrating how to leverage AI tools responsibly, adhering to CFA standards for diligence and reasonable basis.

### High-Level Story Flow

The application guides the user through a structured workflow:

1.  **Introduction**: Sets the stage, introduces the persona and the core challenge of balancing AutoML's potential with its risks in finance.
2.  **Data Preparation**: Simulates the crucial step of preparing financial panel data with strict temporal discipline, creating train, validation, and held-out test sets to prevent look-ahead bias.
3.  **AutoML Exploration (TPOT)**: Initiates TPOT (Tree-based Pipeline Optimization Tool) to discover predictive pipelines using genetic programming. The user triggers the search and reviews the best-found pipeline.
4.  **AutoML Exploration (H2O AutoML)**: Deploys H2O AutoML to leverage stacked ensembles for signal discovery. The user runs the process and inspects the model leaderboard.
5.  **AutoML Exploration (Optuna XGBoost)**: Utilizes Optuna for Bayesian hyperparameter optimization of a targeted XGBoost model, demonstrating a more controlled AutoML approach.
6.  **The Overfitting Audit**: Conducts a rigorous "four-check" audit (Overfitting Ratio, Bonferroni Correction, Held-out Test Performance, Financial Sanity Check) for each AutoML-discovered signal to quantify and detect overfitting. This is the core risk management step.
7.  **Comparative Analysis & Governance**: Compares AutoML approaches against a human-designed baseline across various dimensions and culminates in a governance assessment, reiterating the indispensable role of human oversight and ethical standards in AI-driven finance.

This flow ensures the learner actively engages with each stage of signal discovery and validation, reinforcing the practical application of concepts rather than passive consumption.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
import optuna
from sklearn.metrics import r2_score, mean_squared_error
from source import * # This imports all functions and global variables from source.py
```

### `st.session_state` Design

`st.session_state` is used to preserve data, models, and results across different pages and user interactions.

*   **Initialization**: All keys are initialized at the start of the `app.py` script if they are not already present.
*   **Update**: Keys are updated with results from function calls or user inputs.
*   **Read**: Keys are read on subsequent page loads or interactions to display previous results or use them in further computations.

Here's a list of `st.session_state` keys, their purpose, and how they are used:

| `st.session_state` Key           | Purpose                                                              | Initialization                                  | Updated By                                                                                                                  | Read By                                                                                                                                                                                                                                                                |
| :------------------------------- | :------------------------------------------------------------------- | :---------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `page`                           | Current active page for multi-page simulation                        | `'Introduction'`                                | Sidebar `st.selectbox`                                                                                                      | Conditional rendering logic for pages                                                                                                                                                                                                                                  |
| `df_initialized`                 | Boolean flag to track if data has been generated                     | `False`                                         | Data generation button click                                                                                                | Data Preparation page (to prevent re-generation), subsequent AutoML pages (to ensure data exists)                                                                                                                                                                      |
| `df`                             | Full synthetic financial DataFrame                                   | `None`                                          | `generate_panel_data` (and subsequent slicing in app)                                                                       | Data Preparation page, all AutoML pages for `X_train`, `y_train` etc.                                                                                                                                                                                                  |
| `X_train`, `y_train`             | Training features and target                                         | `None`                                          | Data generation logic                                                                                                       | All AutoML sections for training models, Overfitting Audit for train R2 calculation                                                                                                                                                                                    |
| `X_val`, `y_val`                 | Validation features and target                                       | `None`                                          | Data generation logic                                                                                                       | All AutoML sections for model validation, Overfitting Audit for val R2 calculation                                                                                                                                                                                     |
| `X_test`, `y_test`               | Held-out test features and target                                    | `None`                                          | Data generation logic                                                                                                       | Overfitting Audit for final unbiased evaluation                                                                                                                                                                                                                        |
| `features`                       | List of feature column names                                         | `None`                                          | Data generation logic                                                                                                       | All AutoML sections (to specify features), Overfitting Audit                                                                                                                                                                                                           |
| `target`                         | Target variable column name (`'forward_return'`)                     | `None`                                          | Data generation logic                                                                                                       | All AutoML sections (to specify target), Overfitting Audit                                                                                                                                                                                                             |
| `tpot_model`                     | Fitted TPOTRegressor object                                          | `None`                                          | TPOT execution button click                                                                                                 | TPOT page (to display pipeline), Overfitting Audit (for test predictions), Comparative Analysis (for details)                                                                                                                                                            |
| `tpot_results`                   | Dictionary: `{'train_r2', 'val_r2', 'n_pipelines_tested'}`           | `None`                                          | TPOT execution button click                                                                                                 | TPOT page (to display R2s), Overfitting Audit (for metrics), Comparative Analysis                                                                                                                                                                                      |
| `h2o_aml_object`                 | H2O AutoML `aml` object (leader, leaderboard)                        | `None`                                          | H2O AutoML execution button click                                                                                           | H2O AutoML page (to display leaderboard/leader), Overfitting Audit (for `h2o.get_model` and test predictions), Comparative Analysis                                                                                                                                    |
| `h2o_aml_leader_id`              | `model_id` of H2O AutoML's best model                              | `None`                                          | H2O AutoML execution button click (`aml.leader.model_id`)                                                                   | Overfitting Audit (to explicitly load the best H2O model after re-initializing H2O cluster)                                                                                                                                                                            |
| `h2o_results`                    | Dictionary: `{'train_r2', 'val_r2', 'n_pipelines_tested'}`           | `None`                                          | H2O AutoML execution button click                                                                                           | H2O AutoML page (to display R2s), Overfitting Audit (for metrics), Comparative Analysis                                                                                                                                                                                |
| `optuna_study`                   | Optuna `study` object (best params, trials)                          | `None`                                          | Optuna execution button click                                                                                               | Optuna page (to display study results, plot importance), Overfitting Audit (for details)                                                                                                                                                                               |
| `optuna_best_xgb`                | Fitted XGBoost model from Optuna                                     | `None`                                          | Optuna execution button click                                                                                               | Optuna page (for R2s, feature importances), Overfitting Audit (for test predictions), Comparative Analysis                                                                                                                                                            |
| `optuna_results`                 | Dictionary: `{'train_r2', 'val_r2', 'n_pipelines_tested'}`           | `None`                                          | Optuna execution button click                                                                                               | Optuna page (to display R2s), Overfitting Audit (for metrics), Comparative Analysis                                                                                                                                                                                    |
| `audit_summary`                  | Dictionary summarizing audit results for TPOT, H2O, Optuna            | `None`                                          | Overfitting Audit button click (`overfitting_audit` function)                                                               | Overfitting Audit page (to display summary dashboard), Comparative Analysis                                                                                                                                                                                            |
| `tpot_export_content`            | Content of TPOT exported pipeline file (`tpot_best_pipeline.py`)   | `None`                                          | TPOT execution button click (`tpot.export()`)                                                                               | TPOT page (to display for human review)                                                                                                                                                                                                                                |
| `human_vs_automl_comparison_df` | DataFrame for Human vs. AutoML comparison table                      | `None`                                          | Comparative Analysis button click (generated from `human_vs_automl_comparison` conceptual data)                           | Comparative Analysis page (to display the comparison table)                                                                                                                                                                                                            |
| `optuna_param_importance_fig`    | Plotly figure for Optuna parameter importance                        | `None`                                          | Optuna execution button click (`optuna.visualization.plot_param_importances`)                                               | Optuna page (to display the plot)                                                                                                                                                                                                                                      |

### UI Interactions and Function Calls from `source.py`

#### Application Structure
The application uses a sidebar for navigation and conditional rendering for page content.

```python
st.set_page_config(layout="wide")

# Sidebar for navigation
with st.sidebar:
    st.title("AutoML Audit Navigator")
    page = st.selectbox(
        "Go to",
        [
            "Introduction",
            "1. Data Preparation",
            "2. TPOT AutoML Exploration",
            "3. H2O AutoML Exploration",
            "4. Optuna for XGBoost",
            "5. Overfitting Audit",
            "6. Comparative Analysis & Governance"
        ],
        key="page" # Bind to session state
    )

# Main content area based on selected page
if st.session_state.page == "Introduction":
    # ... Introduction page content ...
elif st.session_state.page == "1. Data Preparation":
    # ... Data Preparation page content ...
# ... and so on for other pages
```

#### Page: Introduction

*   **Markdown:**
    ```python
    st.markdown("# AutoML for Automated Trading Signal Discovery: Auditing Alpha Signals for Overfitting")
    st.markdown("### Case Study: Leveraging AI to Discover and Rigorously Audit Investment Signals")
    st.markdown("**Persona**: Sri Krishnamurthy, CFA Charterholder and Investment Analyst at AlphaQuant Capital.  ")
    st.markdown("**Organization**: AlphaQuant Capital, a quantitative investment firm.")
    st.markdown(f"As an investment analyst at AlphaQuant Capital, my primary goal is to uncover novel alpha-generating signals in financial markets. Historically, this has involved manually developing, testing, and refining investment strategies, a process that is incredibly time-consuming and often limits our exploration to familiar hypotheses. My firm is at the forefront of leveraging cutting-edge AI to accelerate our research, specifically through Automated Machine Learning (AutoML).")
    st.markdown(f"While AutoML promises to rapidly explore a vast space of potential signals, generating new hypotheses faster than any human, it comes with significant risks, particularly overfitting to the inherent noise and temporal complexities of financial data. My role isn't just to find these signals, but to rigorously audit them, ensuring that any AI-generated insight is a genuine discovery and not a spurious correlation. This aligns with CFA Standard V(A) on Diligence and Reasonable Basis, which mandates a deep understanding and rigorous validation of any model, regardless of its origin.")
    st.markdown(f"This application documents my journey through using various AutoML techniques (TPOT, H2O AutoML, Optuna for XGBoost) to find predictive signals for future stock returns, followed by a critical 'four-check' overfitting audit. The objective is not to automatically deploy signals, but to generate robust hypotheses that warrant further human investigation, ultimately enhancing AlphaQuant Capital's research capabilities while upholding the highest standards of diligence and ethical practice.")
    st.markdown("---")
    st.markdown("## Workflow Overview")
    st.image("workflow_overview.png", caption="Conceptual Workflow of AutoML Signal Discovery and Audit") # Placeholder for a conceptual image
    st.markdown("### CFA Curriculum Connection")
    st.markdown(f"**CFA Standard V(A) - Diligence and Reasonable Basis**: Using AutoML does not relieve the analyst of the duty to understand the model. 'I ran AutoML and it found this signal' is not a reasonable basis for an investment decision. The analyst must understand why the discovered pipeline works, validate it out-of-sample, and confirm it makes financial sense.")
    st.markdown(f"**Quantitative Methods**: Multiple-testing corrections (Bonferroni, Benjamini-Hochberg), temporal cross-validation, and the distinction between in-sample optimization and out-of-sample prediction are core quant topics that AutoML forces us to confront.")
    ```

#### Page: 1. Data Preparation

*   **Markdown:**
    ```python
    st.markdown("## 1. Setting Up the Environment and Data Foundations")
    st.markdown(f"Before we dive into the AutoML exploration, we need to prepare our financial dataset. Strict temporal discipline in data splitting is paramount in finance to avoid look-ahead bias and accurately simulate real-world trading conditions.")
    st.markdown("---")
    st.markdown("## 2. Data Preparation: Simulating Financial Data with Strict Temporal Splits")
    st.markdown(f"As an investment analyst, preparing the data is often the most critical first step. For financial time-series, simply randomizing data for train/test splits is a grave mistake; it introduces look-ahead bias. We must maintain strict temporal ordering, splitting our data into distinct training, validation, and a truly held-out future test set. This ensures that our models are only trained on past information and evaluated on unseen future data.")
    st.markdown(f"For this case study, we'll generate a synthetic dataset representing monthly panel data for 200 stocks over 10 years (120 months), with various financial features and a 'forward_return' as our target variable.")
    st.markdown("### **Mathematical Context for Financial Data Generation**")
    st.markdown(r"The `forward_return` (our target variable, denoted as $R_f$) is generated using a combination of linear and non-linear relationships with several synthetic features, plus an irreducible noise component. This reflects the complex and often noisy nature of real-world asset returns.")
    st.markdown(r"For instance, a simplified version of the return generation could be:")
    st.markdown(r"$$ R_f = \beta_1 \cdot \text{{PE\_Z}} + \beta_2 \cdot \text{{Momentum\_12m}} + \beta_3 \cdot \text{{ROE}} + \epsilon $$")
    st.markdown(r"where $\text{{PE\_Z}}$ is price-to-earnings Z-score, $\text{{Momentum\_12m}}$ is 12-month momentum, $\text{{ROE}}$ is return on equity, and $\epsilon$ represents random noise. Our synthetic data generation includes more complex interactions (e.g., `momentum_12m * (-pe_z) * (pe_z < 0)`), mirroring the kind of nuanced signals analysts look for.")
    st.markdown(r"The strict temporal split is defined as:")
    st.markdown(r"*   **Training Set**: Months $0$ to $71$")
    st.markdown(r"*   **Validation Set**: Months $72$ to $95$")
    st.markdown(r"*   **Held-Out Test Set**: Months $96$ to $119$")
    st.markdown(r"The held-out test set is crucial. It simulates true future performance and must *never* be touched during model training or validation, safeguarding against multiple-testing bias and excessive hyperparameter tuning.")
    ```
*   **UI Interaction & Function Call**:
    ```python
    if st.button("Generate Financial Data"):
        # Parameters for data generation
        N_STOCKS = 200
        N_MONTHS = 120
        with st.spinner("Generating synthetic financial data..."):
            # Call generate_panel_data from source.py
            df = generate_panel_data(N_STOCKS, N_MONTHS)

            # Define features and target (as per source.py code cell)
            features = [col for col in df.columns if col not in ['month', 'stock_id', 'forward_return']]
            target = 'forward_return'

            # Strict temporal split (as per source.py code cell)
            train_df = df[df['month'] < 72]
            val_df = df[(df['month'] >= 72) & (df['month'] < 96)]
            test_df = df[df['month'] >= 96]

            # Update session state
            st.session_state.df = df
            st.session_state.X_train, st.session_state.y_train = train_df[features], train_df[target]
            st.session_state.X_val, st.session_state.y_val = val_df[features], val_df[target]
            st.session_state.X_test, st.session_state.y_test = test_df[features], test_df[target]
            st.session_state.features = features
            st.session_state.target = target
            st.session_state.df_initialized = True
        st.success("Data generation complete!")

    if st.session_state.df_initialized:
        st.markdown(f"Total observations: {len(st.session_state.df)}")
        st.markdown(f"Training set: months 0-71 ({len(st.session_state.X_train)} observations)")
        st.markdown(f"Validation set: months 72-95 ({len(st.session_state.X_val)} observations)")
        st.markdown(f"Held-out Test set: months 96-119 ({len(st.session_state.X_test)} observations) - LOCKED UNTIL FINAL AUDIT")
        st.markdown(f"Features: {st.session_state.features}")
        st.markdown(f"Target: {st.session_state.target}")
        st.markdown("#### **Explanation of Temporal Split and its Relevance**")
        st.markdown(f"This output confirms our data has been rigorously split. For an investment analyst, this temporal discipline is non-negotiable. If we were to mix future data into our training or validation sets, our model's performance metrics would be artificially inflated, leading to false confidence and potentially disastrous investment decisions. The 'locked' held-out test set simulates future market conditions, providing an unbiased estimate of our signal's true performance. It's the ultimate safeguard against data snooping bias and overfitting, especially critical when using powerful exploration tools like AutoML that can easily find spurious patterns.")
        st.markdown("---")
    else:
        st.info("Click 'Generate Financial Data' to begin.")
    ```

#### Page: 2. TPOT AutoML Exploration

*   **Pre-requisite Check:** `if not st.session_state.df_initialized:` display warning and return.
*   **Markdown:**
    ```python
    st.markdown("## 3. AutoML Exploration 1: TPOT for Genetic Pipeline Search")
    st.markdown(f"Now that our data is prepared, we'll unleash AutoML. The first technique we'll explore is TPOT, which uses genetic programming to automatically design and optimize machine learning pipelines. As an analyst, I see TPOT as a 'hypothesis engine' that can explore thousands of model architectures, feature engineering steps, and hyperparameter combinations that I might never conceive of manually.")
    st.markdown("### **Mathematical Formulation: TPOT's Genetic Programming**")
    st.markdown(r"TPOT (Tree-based Pipeline Optimization Tool) represents entire machine learning pipelines as tree-like 'genomes.' Each genome consists of a sequence of operations: a preprocessor, a feature selector, a model, and their associated hyperparameters.")
    st.markdown(r"The process of finding an optimal pipeline ($P^*$) involves an evolutionary algorithm:")
    st.markdown(r"1.  **Initialization**: Randomly generate a 'population' of pipelines.")
    st.markdown(r"2.  **Fitness Evaluation**: For each pipeline $P$ in the population, evaluate its fitness. Fitness is typically a cross-validated score (e.g., negative mean squared error) on the training data: $f(P) = \text{{CV-Score}}(P, X_{{\text{{train}}}}, Y_{{\text{{train}}}})$.")
    st.markdown(r"3.  **Selection**: Select the best-performing pipelines based on their fitness.")
    st.markdown(r"4.  **Reproduction (Crossover & Mutation)**:")
    st.markdown(r"*   **Crossover**: Combine components (sub-trees) from two parent pipelines to create new offspring pipelines.")
    st.markdown(r"*   **Mutation**: Randomly change a component or hyperparameter within a pipeline.")
    st.markdown(r"5.  **Iteration**: Repeat steps 2-4 for a specified number of 'generations' (e.g., 20 generations), gradually evolving better pipelines.")
    st.markdown(r"The total number of hypotheses tested can be substantial. For example, `population_size=50` and `generations=20` imply $50 \times 20 = 1000$ unique pipelines evaluated, each with 5-fold cross-validation on the training set. This massive exploration capability highlights the multiple-testing problem: with so many tests, some 'significant' results are bound to appear purely by chance, even on random data. This is why strict validation is crucial.")
    ```
*   **UI Interaction & Function Call**:
    ```python
    if st.button("Run TPOT AutoML (Genetic Pipeline Search)"):
        with st.spinner("Running TPOT... This may take a few minutes (reduced settings for demo)."):
            # Initialize TPOTRegressor (using parameters from source.py code cell)
            tpot = TPOTRegressor(
                generations=10,             # Reduced generations for demonstration purposes
                population_size=20,           # Reduced population size
                cv=3,                         # Reduced cross-validation folds
                scoring='neg_mean_squared_error',
                max_time_mins=5,            # Set a time budget for the search
                random_state=42,
                verbosity=0,                  # Suppress verbosity in Streamlit output
                config_dict='TPOT sparse'     # Lighter config to run faster
            )
            tpot.fit(st.session_state.X_train, st.session_state.y_train)

            # Predict on training and validation sets
            tpot_train_pred = tpot.predict(st.session_state.X_train)
            tpot_val_pred = tpot.predict(st.session_state.X_val)

            # Calculate R2 scores
            tpot_train_r2 = r2_score(st.session_state.y_train, tpot_train_pred)
            tpot_val_r2 = r2_score(st.session_state.y_val, tpot_val_pred)

            # Export the discovered pipeline as Python code for review
            tpot_export_path = 'tpot_best_pipeline.py'
            tpot.export(tpot_export_path)
            with open(tpot_export_path, 'r') as f:
                tpot_export_content = f.read()

            # Store results in session state
            st.session_state.tpot_model = tpot
            st.session_state.tpot_results = {
                'train_r2': tpot_train_r2,
                'val_r2': tpot_val_r2,
                'n_pipelines_tested': tpot.population_size * tpot.generations
            }
            st.session_state.tpot_export_content = tpot_export_content

        st.success("TPOT search complete!")

    if st.session_state.tpot_model:
        st.markdown(f"### TPOT Best Pipeline Found:")
        st.code(str(st.session_state.tpot_model.fitted_pipeline_))
        st.markdown(f"In-sample R2 (Train): {st.session_state.tpot_results['train_r2']:.4f}")
        st.markdown(f"Validation R2 (Val): {st.session_state.tpot_results['val_r2']:.4f}")
        st.markdown(f"Number of pipelines tested: {st.session_state.tpot_results['n_pipelines_tested']}")

        st.subheader("TPOT Discovered Pipeline Export:")
        st.markdown(f"Pipeline exported to `tpot_best_pipeline.py` for human review.")
        st.code(st.session_state.tpot_export_content, language='python')

        st.subheader("TPOT Best Validation R2 per Generation (Illustrative)")
        # Conceptual plot as per source.py
        generations = list(range(1, st.session_state.tpot_model.generations + 1))
        # Create a conceptual plot showing improvement
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, np.sort(np.random.uniform(0.01, st.session_state.tpot_results['val_r2'] + 0.05, st.session_state.tpot_model.generations))[::-1], marker='o', linestyle='-', color='skyblue')
        ax.set_title('TPOT Best Validation R2 per Generation (Illustrative)')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Validation R2')
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("#### **Explanation of TPOT Results**")
        st.markdown(f"The TPOT output reveals the best-performing machine learning pipeline (e.g., a combination of a preprocessor, feature selector, and regressor like `ExtraTreesRegressor`) it discovered through its genetic search. The in-sample R2 represents how well the pipeline fits the training data, while the validation R2 indicates its generalization ability on unseen data. A significant drop from training R2 to validation R2 is an early warning sign of overfitting. The ability to export the pipeline's Python code is crucial for me as an analyst; it allows for human review of the generated logic, ensuring transparency and aiding in the sanity check. The conceptual plot, even if illustrative, helps me visualize how TPOT's evolutionary algorithm iteratively improves the pipeline's performance over generations, eventually converging to an optimal (or near-optimal) solution.")
        st.markdown("---")
    else:
        st.info("Click 'Run TPOT AutoML' to start the genetic pipeline search.")
    ```

#### Page: 3. H2O AutoML Exploration

*   **Pre-requisite Check:** `if not st.session_state.df_initialized:` display warning and return.
*   **Markdown:**
    ```python
    st.markdown("## 4. AutoML Exploration 2: H2O AutoML for Stacked Ensembles")
    st.markdown(f"Next, we'll explore H2O AutoML, an industrial-strength platform known for its efficiency and robust stacked ensemble capabilities. H2O AutoML is particularly adept at combining the strengths of multiple base models, often leading to superior predictive performance. As an analyst, I find H2O's speed and its ability to manage a wide array of algorithms valuable for broad signal discovery.")
    st.markdown("### **Mathematical Formulation: Stacked Ensembles**")
    st.markdown(r"H2O AutoML's strength lies in its stacked ensemble models. A stacked ensemble, or 'stacking,' combines predictions from multiple diverse base models (e.g., GBM, XGBoost, GLM, Deep Learning, Random Forest) using a meta-learner.")
    st.markdown(r"The process involves:")
    st.markdown(r"1.  **Base Model Training**: Multiple base models are trained independently on the training data. For each base model $M_k$, it produces predictions $P_{{k,i}}$ for each data point $i$.")
    st.markdown(r"2.  **Meta-Learner Training**: A meta-learner (e.g., a generalized linear model or a simple neural network) is trained on the out-of-fold predictions from the base models. That is, the input to the meta-learner for a data point $i$ is a vector of predictions $[P_{{1,i}}, P_{{2,i}}, \dots, P_{{K,i}}]$ from $K$ base models, and the target is the actual target variable $Y_i$.")
    st.markdown(r"3.  **Final Prediction**: To make a final prediction on new data, each base model first makes a prediction. These predictions are then fed into the trained meta-learner to produce the final output.")
    st.markdown(r"This approach often yields better predictive accuracy than any single model alone because it leverages the strengths of diverse models and mitigates individual weaknesses. H2O manages this entire complex process automatically, providing a leaderboard of models ranked by a chosen metric (like RMSE for regression).")
    ```
*   **UI Interaction & Function Call**:
    ```python
    if st.button("Run H2O AutoML (Stacked Ensembles)"):
        with st.spinner("Initializing H2O cluster and running AutoML..."):
            h2o.init(nthreads=-1, max_mem_size="4G") # Initialize H2O cluster

            # Convert pandas DataFrames to H2O Frames
            h_train = h2o.H2OFrame(pd.concat([st.session_state.X_train, st.session_state.y_train], axis=1))
            h_val = h2o.H2OFrame(pd.concat([st.session_state.X_val, st.session_state.y_val], axis=1))

            # Define H2O AutoML parameters
            aml = H2OAutoML(
                max_models=10,             # Reduced max models for demonstration
                max_runtime_secs=300,      # Time budget for the search (e.g., 5 minutes)
                seed=42,
                sort_metric='rmse',        # Sort by Root Mean Squared Error
                exclude_algos=['DeepLearning'] # Exclude slow Deep Learning models for speed
            )

            # Train H2O AutoML
            aml.train(x=st.session_state.features, y=st.session_state.target, training_frame=h_train, validation_frame=h_val)

            leaderboard = aml.leaderboard.as_data_frame()
            best_h2o_model = aml.leader

            h2o_val_perf = best_h2o_model.model_performance(h_val)
            h2o_train_perf = best_h2o_model.model_performance(h_train)

            # Store results in session state
            st.session_state.h2o_aml_object = aml # Store the aml object
            st.session_state.h2o_aml_leader_id = aml.leader.model_id # Store leader model ID
            st.session_state.h2o_results = {
                'train_r2': h2o_train_perf.r2(),
                'val_r2': h2o_val_perf.r2(),
                'n_pipelines_tested': aml.max_models # Approx number of models
            }

            h2o.cluster().shutdown(prompt=False) # Shut down H2O cluster
        st.success("H2O AutoML search complete and cluster shut down.")

    if st.session_state.h2o_aml_object:
        st.subheader("H2O AutoML Leaderboard (top 5 by RMSE):")
        leaderboard_df = st.session_state.h2o_aml_object.leaderboard.as_data_frame()
        st.dataframe(leaderboard_df[['model_id', 'rmse', 'mae', 'r2']].head())

        # Retrieve performance from session state
        st.markdown(f"### H2O Best Model: {st.session_state.h2o_aml_leader_id}")
        st.markdown(f"Validation R2: {st.session_state.h2o_results['val_r2']:.4f}")
        st.markdown(f"In-sample R2 (Train): {st.session_state.h2o_results['train_r2']:.4f}")

        st.markdown("#### **Explanation of H2O AutoML Results**")
        st.markdown(f"The H2O AutoML leaderboard provides a ranked list of models, showcasing diverse architectures like XGBoost, LightGBM, Random Forest, and stacked ensembles. The `aml.leader` represents the best-performing model, often a stacked ensemble, which combines predictions from several individual models. Its validation RMSE and R2 provide a measure of its out-of-sample predictive power. For me, the analyst, the ability to quickly compare many sophisticated models and get the best one, along with its performance metrics, is a huge time-saver. The leaderboard helps me understand the diversity of models explored and their relative strengths.")
        st.markdown("---")
    else:
        st.info("Click 'Run H2O AutoML' to start the stacked ensemble search.")
    ```

#### Page: 4. Optuna for XGBoost

*   **Pre-requisite Check:** `if not st.session_state.df_initialized:` display warning and return.
*   **Markdown:**
    ```python
    st.markdown("## 5. AutoML Exploration 3: Optuna for Bayesian Hyperparameter Optimization (XGBoost)")
    st.markdown(f"While full-fledged AutoML solutions like TPOT and H2O explore entire pipelines and model ensembles, sometimes a lighter, more controlled approach is preferred, especially when we already have a strong candidate model. Optuna provides a flexible framework for Bayesian hyperparameter optimization, allowing us to efficiently tune a specific model (e.g., XGBoost) without getting lost in the complexity of pipeline search. As an analyst, this is often my 'pragmatic sweet spot' for production-ready signals.")
    st.markdown("### **Mathematical Formulation: Bayesian Hyperparameter Optimization**")
    st.markdown(r"Bayesian Optimization aims to find the global optimum of an objective function that is expensive to evaluate (like model training and validation). Unlike grid search or random search, it intelligently explores the hyperparameter space by building a probabilistic model (a 'surrogate model') of the objective function.")
    st.markdown(r"The process involves:")
    st.markdown(r"1.  **Surrogate Model**: Start with a few random trials, then build a surrogate model (e.g., Gaussian Process, Tree-structured Parzen Estimator (TPE) as used by Optuna's default `TPESampler`) to approximate the objective function's behavior across the hyperparameter space.")
    st.markdown(r"2.  **Acquisition Function**: Use an acquisition function (e.g., Expected Improvement, Probability of Improvement) to decide which hyperparameters to sample next. This function balances **exploration** (trying hyperparameters in regions with high uncertainty) and **exploitation** (trying hyperparameters in regions where the surrogate model predicts good performance).")
    st.markdown(r"3.  **Iteration**: Evaluate the objective function with the new hyperparameters, update the surrogate model, and repeat.")
    st.markdown(r"Optuna's `TPESampler` samples hyperparameters by maximizing the Expected Improvement, which is calculated based on two probability distributions: $l(x)$ for hyperparameters that led to good performance and $g(x)$ for those that performed poorly. The next set of hyperparameters is chosen to maximize $\frac{{l(x)}}{{g(x)}}$. This iterative, informed search is far more efficient than brute-force methods.")
    ```
*   **UI Interaction & Function Call**:
    ```python
    if st.button("Run Optuna Bayesian Optimization for XGBoost"):
        with st.spinner("Running Optuna optimization for XGBoost..."):
            # Create an Optuna study
            # Use objective from source.py directly. X_train, y_train are from session state.
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=50, show_progress_bar=False) # Reduced trials for demo, suppress progress bar in Streamlit

            # Train the final XGBoost model with the best parameters on the full training set
            best_xgb = XGBRegressor(**study.best_params, random_state=42)
            best_xgb.fit(st.session_state.X_train, st.session_state.y_train)

            # Predict on validation set
            optuna_val_pred = best_xgb.predict(st.session_state.X_val)
            optuna_train_pred = best_xgb.predict(st.session_state.X_train)

            # Calculate R2 scores
            optuna_val_r2 = r2_score(st.session_state.y_val, optuna_val_pred)
            optuna_train_r2 = r2_score(st.session_state.y_train, optuna_train_pred)

            # Visualize parameter importance
            try:
                fig = optuna.visualization.plot_param_importances(study)
                st.session_state.optuna_param_importance_fig = fig
            except Exception as e:
                st.warning(f"Could not generate parameter importance plot: {e}")
                st.session_state.optuna_param_importance_fig = None

            # Store results in session state
            st.session_state.optuna_study = study
            st.session_state.optuna_best_xgb = best_xgb
            st.session_state.optuna_results = {
                'train_r2': optuna_train_r2,
                'val_r2': optuna_val_r2,
                'n_pipelines_tested': len(study.trials)
            }
        st.success("Optuna optimization complete!")

    if st.session_state.optuna_best_xgb:
        st.markdown(f"### Optuna Results:")
        st.markdown(f"Best trial MSE: {st.session_state.optuna_study.best_value:.6f}")
        st.markdown(f"Best parameters: `{st.session_state.optuna_study.best_params}`")
        st.markdown(f"Trials run: {st.session_state.optuna_results['n_pipelines_tested']}")
        st.markdown(f"In-sample R2 (Train) with best Optuna XGBoost: {st.session_state.optuna_results['train_r2']:.4f}")
        st.markdown(f"Validation R2 with best Optuna XGBoost: {st.session_state.optuna_results['val_r2']:.4f}")

        st.subheader("Optuna Parameter Importance Plot")
        if st.session_state.optuna_param_importance_fig:
            st.plotly_chart(st.session_state.optuna_param_importance_fig, use_container_width=True)
        else:
            st.info("Parameter importance plot not available (Plotly might not be installed or an error occurred).")

        st.subheader("XGBoost Feature Importances (from best Optuna model)")
        feature_importances_df = pd.DataFrame({
            'Feature': st.session_state.features,
            'Importance': st.session_state.optuna_best_xgb.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.dataframe(feature_importances_df)


        st.markdown("#### **Explanation of Optuna Results**")
        st.markdown(f"Optuna successfully identified the optimal hyperparameters for our XGBoost model, achieving a low Mean Squared Error (MSE) on the cross-validation folds. The visualization of parameter importance helps me understand which hyperparameters had the most significant impact on performance. For an investment analyst, this targeted approach is ideal when we believe a particular model (like XGBoost, known for its strong performance in tabular data) is suitable, but we need to find its optimal configuration efficiently. It reduces the 'black box' nature compared to full AutoML and allows for more controlled experimentation, lowering the multiple-testing burden.")
        st.markdown("---")
    else:
        st.info("Click 'Run Optuna Bayesian Optimization' to optimize XGBoost hyperparameters.")
    ```

#### Page: 5. Overfitting Audit

*   **Pre-requisite Check:** `if not (st.session_state.tpot_model and st.session_state.h2o_aml_object and st.session_state.optuna_best_xgb):` display warning and return.
*   **Markdown:**
    ```python
    st.markdown("## 6. The Overfitting Audit: The 'Four Checks' for AI-Generated Signals")
    st.markdown(f"This is the most critical step. My firm's reputation and our clients' capital depend on our ability to differentiate genuine alpha signals from spurious discoveries. Given the high noise and temporal dependencies in financial markets, AutoML's intensive search can easily lead to models that overfit the historical data. The 'four-check' overfitting audit, inspired by rigorous quantitative finance practices, provides a structured framework to validate these AI-generated signals.")
    st.markdown("### **Mathematical Formulation: Overfitting Ratio and Multiple-Testing Correction**")
    st.markdown(r"The audit includes two key quantitative checks:")
    st.markdown(r"1.  **Overfitting Ratio**: This metric quantifies the drop-off in performance from the training set to the validation set. A ratio close to $0$ indicates good generalization, while a ratio close to $1$ (or higher) suggests severe overfitting.")
    st.markdown(r"$$ \text{{Overfitting Ratio}} = 1 - \frac{{\text{{Val R2}}}}{{\text{{max}}(\text{{Train R2}}, 0.0001)}} $$")
    st.markdown(r"where $\text{{Val R2}}$ is the R-squared score on the validation set, and $\text{{Train R2}}$ is the R-squared score on the training set. We use $\text{{max}}(\text{{Train R2}}, 0.0001)$ in the denominator to avoid division by zero or near-zero R2 values, making the ratio more robust when models perform poorly even in-sample. A target ratio of $< 0.5$ is a typical heuristic.")
    st.markdown(r"2.  **Multiple-Testing Correction (Bonferroni)**: When an AutoML system tests hundreds or thousands of pipelines, the probability of finding a 'significant' result purely by chance (Type I error) dramatically increases. The Bonferroni correction is a conservative adjustment that lowers the significance threshold for each individual test. If we perform $N$ independent tests, the new significance level $\alpha_{{\text{{Bonferroni}}}}$ for each test is:")
    st.markdown(r"$$ \alpha_{{\text{{Bonferroni}}}} = \frac{{\alpha_{{\text{{original}}}}}}{{N}} $$")
    st.markdown(r"where $\alpha_{{\text{{original}}}} = 0.05$ is the original significance level and $N$ is the number of pipelines tested. For a typical $\alpha_{{\text{{original}}}} = 0.05$ and $N=1000$ pipelines, the effective p-value threshold for an individual test would need to be $0.05 / 1000 = 0.00005$. This means the validation R2 needs to be *extremely* significant to pass this check, reflecting the skepticism required when dealing with extensive search processes.")
    st.markdown(r"The four checks are:")
    st.markdown(r"*   **Check 1: Overfitting Ratio**: Is the validation performance sufficiently close to training performance?")
    st.markdown(r"*   **Check 2: Multiple-Testing Correction**: Is the signal's statistical significance robust after accounting for the vast number of pipelines tested?")
    st.markdown(r"*   **Check 3: Held-Out Test Performance**: Does the signal perform positively on a truly unseen, future dataset, and is this performance a meaningful fraction of the validation performance?")
    st.markdown(r"*   **Check 4: Financial Sanity Check**: Does the discovered model use financially intuitive features? Does its logic align with economic theory? (This requires human judgment).")
    ```
*   **UI Interaction & Function Call**:
    ```python
    if st.button("Run Overfitting Audit"):
        with st.spinner("Performing the four-check overfitting audit..."):
            # Predict on the held-out test set for each model
            # TPOT
            tpot_test_pred = st.session_state.tpot_model.predict(st.session_state.X_test)
            tpot_test_r2 = r2_score(st.session_state.y_test, tpot_test_pred)
            tpot_audit_status, tpot_overfit_ratio, tpot_bonferroni, tpot_test_pass, _ = overfitting_audit(
                'TPOT (Genetic Programming)', st.session_state.tpot_results['train_r2'], st.session_state.tpot_results['val_r2'],
                tpot_test_r2, st.session_state.tpot_results['n_pipelines_tested']
            )

            # H2O AutoML - Re-initialize H2O and load model by ID
            h2o.init(nthreads=-1, max_mem_size="4G")
            h_test = h2o.H2OFrame(pd.concat([st.session_state.X_test, st.session_state.y_test], axis=1))
            best_h2o_model_audit = h2o.get_model(st.session_state.h2o_aml_leader_id)
            h2o_test_pred_h2o = best_h2o_model_audit.predict(h_test[st.session_state.features])
            h2o_test_pred = h2o_test_pred_h2o.as_data_frame().values.flatten()
            h2o_test_r2 = r2_score(st.session_state.y_test, h2o_test_pred)
            h2o_audit_status, h2o_overfit_ratio, h2o_bonferroni, h2o_test_pass, _ = overfitting_audit(
                'H2O AutoML (Stacked Ensembles)', st.session_state.h2o_results['train_r2'], st.session_state.h2o_results['val_r2'],
                h2o_test_r2, st.session_state.h2o_results['n_pipelines_tested']
            )
            h2o.cluster().shutdown(prompt=False) # Shut down H2O again

            # Optuna XGBoost
            optuna_test_pred = st.session_state.optuna_best_xgb.predict(st.session_state.X_test)
            optuna_test_r2 = r2_score(st.session_state.y_test, optuna_test_pred)
            optuna_audit_status, optuna_overfit_ratio, optuna_bonferroni, optuna_test_pass, _ = overfitting_audit(
                'Optuna XGBoost (Bayesian Opt.)', st.session_state.optuna_results['train_r2'], st.session_state.optuna_results['val_r2'],
                optuna_test_r2, st.session_state.optuna_results['n_pipelines_tested']
            )

            # Store audit results for display
            st.session_state.audit_summary = {
                'TPOT': {
                    'Overfit Ratio': f"{tpot_overfit_ratio:.2f} ({'Pass' if tpot_audit_status and tpot_overfit_ratio < 0.5 else 'Fail'})",
                    'Bonferroni Alpha': f"{tpot_bonferroni:.6f}",
                    'Held-out Test R2': f"{tpot_test_r2:.4f} ({'Pass' if tpot_test_pass else 'Fail'})",
                    'Overall Status': 'Approved' if tpot_audit_status else 'Rejected'
                },
                'H2O AutoML': {
                    'Overfit Ratio': f"{h2o_overfit_ratio:.2f} ({'Pass' if h2o_audit_status and h2o_overfit_ratio < 0.5 else 'Fail'})",
                    'Bonferroni Alpha': f"{h2o_bonferroni:.6f}",
                    'Held-out Test R2': f"{h2o_test_r2:.4f} ({'Pass' if h2o_test_pass else 'Fail'})",
                    'Overall Status': 'Approved' if h2o_audit_status else 'Rejected'
                },
                'Optuna XGBoost': {
                    'Overfit Ratio': f"{optuna_overfit_ratio:.2f} ({'Pass' if optuna_audit_status and optuna_overfit_ratio < 0.5 else 'Fail'})",
                    'Bonferroni Alpha': f"{optuna_bonferroni:.6f}",
                    'Held-out Test R2': f"{optuna_test_r2:.4f} ({'Pass' if optuna_test_pass else 'Fail'})",
                    'Overall Status': 'Approved' if optuna_audit_status else 'Rejected'
                }
            }
        st.success("Overfitting audit complete!")

    if st.session_state.audit_summary:
        st.subheader("Overfitting Audit Summary Dashboard")
        audit_df = pd.DataFrame.from_dict(st.session_state.audit_summary, orient='index')
        st.dataframe(audit_df)

        st.markdown("#### **Explanation of Overfitting Audit Results**")
        st.markdown(f"This audit is where the rubber meets the road. For each AutoML method, I've seen a detailed breakdown of its performance through the lens of overfitting:")
        st.markdown(f"*   The **Overfitting Ratio** provides a quick quantitative check: a value closer to 1 signifies a larger performance drop-off from training to validation, indicating severe overfitting.")
        st.markdown(f"*   The **Bonferroni threshold** emphasizes the incredible stringency required for statistical significance when hundreds or thousands of models are tested. While we don't calculate an exact p-value for R2, this metric reminds me that apparent 'significance' is easily misleading.")
        st.markdown(f"*   The **Held-out Test R2** is the ultimate reality check. A signal that performs well on training and validation but fails on the held-out test set is a classic example of overfitting and would be immediately rejected for deployment.")
        st.markdown(f"*   The **Financial Sanity Check** is my qualitative, but equally vital, contribution as a CFA charterholder. Does the model's logic or its feature importance make financial sense? Does it align with economic intuition? If an AutoML model suggests, for example, that stock ID is the strongest predictor of future returns (a purely spurious correlation), it's a red flag. This human overlay is indispensable.")
        st.markdown(f"The `OVERALL AUDIT STATUS` consolidates these findings, guiding my decision-making process. Most AutoML-discovered signals are likely to be rejected or require substantial human validation, which is the expected and correct outcome. This robust audit ensures that AlphaQuant Capital only pursues genuinely promising signals.")
        st.markdown("---")
    else:
        st.info("Ensure all AutoML models have been run, then click 'Run Overfitting Audit'.")
    ```

#### Page: 6. Comparative Analysis & Governance

*   **Pre-requisite Check:** `if not st.session_state.audit_summary:` display warning and return.
*   **Markdown:**
    ```python
    st.markdown("## 7. Comparative Analysis: AutoML vs. Human-Designed Strategies")
    st.markdown(f"Beyond individual performance, it's crucial to understand the trade-offs between AutoML-discovered signals and a hypothetical human-designed baseline strategy. This comparison helps me, as an analyst, contextualize the value proposition and risks of integrating AI into our signal discovery workflow.")
    ```
*   **UI Interaction & Function Call**:
    ```python
    if st.button("Generate Human vs. AutoML Comparison"):
        # The source.py function `human_vs_automl_comparison()` prints the table.
        # We reconstruct the data structure from source.py to use st.dataframe.
        # This is a direct representation of the conceptual data in the source.py function.
        comparisons = [
            ('Development time', '4-8 hours', '30-60 minutes'),
            ('Pipelines explored', '3-5 (manual iterations)', f'~{st.session_state.tpot_results["n_pipelines_tested"] + st.session_state.h2o_results["n_pipelines_tested"] + st.session_state.optuna_results["n_pipelines_tested"]}'),
            ('Feature engineering', 'Domain-driven, meticulous', 'Automated/Embedded in pipeline'),
            ('Model selection', 'XGBoost (chosen by expert)', 'Best of many (ensembles, genetic)'),
            ('Hyperparameters', 'Manual/grid-search limited', 'Bayesian/genetic optimization'),
            ('Interpretability', 'Full understanding', 'Black-box pipeline (requires XAI)'),
            ('Overfitting risk', 'Moderate (with diligence)', 'HIGH (many tests, subtle)'),
            ('Governance burden', 'Standard (Tier 2)', 'Elevated (Tier 1-2, crucial)'),
            ('Human judgment', 'Embedded throughout process', 'Applied post-hoc (audit crucial)'),
            ('Typical test R2 lift', 'Baseline R2 +0 to +0.005', 'Potentially higher, but higher risk of spuriousness')
        ]
        df_comparison = pd.DataFrame(comparisons, columns=['Dimension', 'Manual (XGBoost Baseline)', 'AutoML (TPOT, H2O, Optuna)'])
        st.session_state.human_vs_automl_comparison_df = df_comparison
        st.success("Comparison generated!")

    if st.session_state.human_vs_automl_comparison_df is not None:
        st.subheader("Human vs. AutoML Comparison")
        st.dataframe(st.session_state.human_vs_automl_comparison_df)
        st.markdown("#### **Explanation of Comparative Analysis**")
        st.markdown(f"This comparison table highlights the inherent trade-offs. AutoML excels in `Development time` and the sheer `Pipelines explored`, acting as an unparalleled `hypothesis engine`. It can uncover `Model selection` and `Hyperparameters` that a human might miss. However, it introduces significant challenges in `Interpretability` (often leading to black-box models) and substantially increases `Overfitting risk` due to the multiple-testing problem. Critically, the `Governance burden` is elevated for AutoML-generated signals, demanding rigorous `Human judgment` post-discovery. For AlphaQuant Capital, this implies a strategy where AutoML accelerates the initial search, but human analysts like myself retain ultimate accountability for validating and understanding every potential signal before it impacts real investments.")
    
    st.markdown("---")
    st.markdown("## 8. Governance Assessment and Conclusion: The AI-Empowered Financial Professional")
    st.markdown(f"The journey through AutoML for signal discovery has culminated in a critical realization: AutoML is a powerful accelerator for hypothesis generation, but it is **not** a strategy factory. My role as a CFA Charterholder requires me to apply diligence, reasonable basis, and sound judgment, regardless of how the signal was discovered.")
    st.markdown("### **CFA Standard V(A) - Diligence and Reasonable Basis**")
    st.markdown(f"CFA Standard V(A) states that members and candidates must: 'Exercise diligence, independence, and thoroughness in analyzing investments, making investment recommendations, and taking investment actions.' Using an AutoML-generated model without understanding its underlying logic or rigorously validating it is a direct violation of this standard. 'I ran AutoML and it found this signal' is not a reasonable basis for an investment decision.")
    st.markdown("### **Governance Assessment for AutoML-Generated Signals**")
    st.markdown(f"1.  **Transparency & Documentation**: For any approved AutoML pipeline (like a TPOT export), full documentation of its components, hyperparameters, and feature usage is required. This is essential for understanding the model and diagnosing issues.")
    st.markdown(f"2.  **Independent Validation**: Signals must undergo rigorous, independent out-of-sample validation using fresh data and methods distinct from those used by AutoML. The four-check audit is a first step, followed by more extensive backtesting and potentially live simulation.")
    st.markdown(f"3.  **Explainability (XAI)**: Even if a model is complex, explainable AI (XAI) techniques (e.g., SHAP, LIME) must be applied to understand feature importance and local predictions. This helps identify spurious correlations and confirm financial intuition.")
    st.markdown(f"4.  **Human Oversight**: Continuous human oversight is paramount. No AutoML-generated signal should be deployed without thorough human review, challenge, and approval. This includes a constant re-evaluation of its financial intuitiveness and market relevance.")
    st.markdown(f"5.  **Risk Management**: Specific risk management protocols must be in place for AutoML-derived signals, recognizing their higher potential for data snooping bias and overfitting compared to human-designed strategies.")

    if st.button("Final Synthesis"):
        # The course_final_synthesis() function prints its output.
        # We capture it to display it using st.text
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            course_final_synthesis()
        synthesis_output = f.getvalue()
        st.session_state.final_synthesis_output = synthesis_output
        st.success("Final synthesis complete!")

    if "final_synthesis_output" in st.session_state and st.session_state.final_synthesis_output:
        st.text(st.session_state.final_synthesis_output)

    st.markdown("#### **Conclusion: The AI-Empowered Financial Professional**")
    st.markdown(f"This lab has underscored a critical lesson: AutoML is a double-edged sword. It significantly enhances our ability to explore the vast space of potential alpha signals, acting as an extremely powerful 'hypothesis engine.' However, it amplifies the risk of overfitting and false discoveries in financial markets, where noise and temporal dependencies are rampant.")
    st.markdown(f"For AlphaQuant Capital, the value of AutoML lies not in its ability to automatically generate deployable strategies, but in its capacity to provide *candidate signals* for further human investigation. My role as an investment analyst, a CFA Charterholder, is more crucial than ever: to apply rigorous auditing, interpret complex models, and exercise sound financial judgment. The **AI-empowered financial professional** combines the speed and breadth of AI with human domain expertise, ethical standards, and diligent oversight to separate genuine alpha from statistical mirage.")
    ```
