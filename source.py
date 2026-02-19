import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
from h2o.automl import H2OAutoML
# from tpot import TPOTRegressor # TPOT is commented out in the original notebook, so it's excluded here.
import optuna
from xgboost import XGBRegressor
import warnings

# Suppress warnings for cleaner output in a notebook environment
warnings.filterwarnings('ignore')

def set_global_settings():
    """Sets global aesthetic settings for plots and suppresses warnings."""
    sns.set_theme(style="whitegrid")
    warnings.filterwarnings('ignore')

def generate_panel_data(n_stocks: int, n_months: int) -> pd.DataFrame:
    """
    Generates synthetic panel data for n_stocks over n_months.
    Features and forward returns are designed to mimic financial data complexity.

    Args:
        n_stocks (int): Number of synthetic stocks to generate.
        n_months (int): Number of synthetic months to generate data for.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic financial panel data.
    """
    np.random.seed(42) # for reproducibility
    records = []
    for month in range(n_months):
        for stock_id in range(n_stocks):
            # Generate synthetic features with some financial intuition in mind
            pe_z = np.random.randn() # Price-to-earnings Z-score
            pb_z = np.random.randn() # Price-to-book Z-score
            roe = np.random.randn() * 0.3 + 0.05 # Return on equity, usually positive
            earnings_growth = np.random.randn() * 0.1 + 0.02 # Earnings growth
            momentum_12m = np.random.randn() * 0.5 # 12-month momentum
            momentum_1m = np.random.randn() * 0.1 # 1-month momentum
            volatility = np.abs(np.random.randn()) * 0.03 + 0.01 # Volatility, always positive
            analyst_revisions = np.random.randn() * 0.2 # Analyst revisions
            log_mcap = np.random.randn() * 2 + 7 # Log Market Cap, typically larger values
            short_interest = np.random.randn() * 0.05 + 0.01 # Short interest, usually positive

            # True return: linear + nonlinear + noise (mimicking some alpha logic)
            # This complex formula is designed to ensure there's a signal to find,
            # but also non-linearities and interactions that AutoML might uncover.
            ret = (
                (0.003 * (-pe_z)) +
                (0.004 * momentum_12m) +
                (0.002 * roe) +
                (0.001 * log_mcap) +
                (0.002 * analyst_revisions) +
                (0.003 * momentum_12m * (-pe_z) * (pe_z < 0)) + # Non-linear interaction
                (0.002 * volatility * (momentum_12m < 0)) +     # Non-linear interaction
                (np.random.randn() * 0.06)                       # Noise
            )

            records.append({
                'month': month,
                'stock_id': stock_id,
                'pe_z': pe_z,
                'pb_z': pb_z,
                'roe': roe,
                'earnings_growth': earnings_growth,
                'momentum_12m': momentum_12m,
                'momentum_1m': momentum_1m,
                'volatility': volatility,
                'analyst_revisions': analyst_revisions,
                'log_mcap': log_mcap,
                'short_interest': short_interest,
                'forward_return': ret,
            })
    return pd.DataFrame(records)

def split_panel_data(df: pd.DataFrame, features: list, target: str,
                     train_months_end: int, val_months_end: int) -> tuple:
    """
    Splits the panel data into training, validation, and test sets based on months.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): List of feature column names.
        target (str): Name of the target column.
        train_months_end (int): The exclusive upper bound for training months.
        val_months_end (int): The exclusive upper bound for validation months.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test) DataFrames/Series.
    """
    train_df = df[df['month'] < train_months_end]
    val_df = df[(df['month'] >= train_months_end) & (df['month'] < val_months_end)]
    test_df = df[df['month'] >= val_months_end]

    print(f"Total observations: {len(df)}")
    print(f"Training set: months 0-{train_months_end-1} ({len(train_df)} observations)")
    print(f"Validation set: months {train_months_end}-{val_months_end-1} ({len(val_df)} observations)")
    print(f"Held-out Test set: months {val_months_end}-{df['month'].max()} ({len(test_df)} observations) - LOCKED UNTIL FINAL AUDIT")
    print(f"Features: {features}")
    print(f"Target: {target}")

    return (train_df[features], train_df[target],
            val_df[features], val_df[target],
            test_df[features], test_df[target])

def run_h2o_automl_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            X_test: pd.DataFrame, features: list, target: str,
                            max_models: int = 10, max_runtime_secs: int = 300) -> dict:
    """
    Runs H2O AutoML to find the best model for the given data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        X_test (pd.DataFrame): Test features (for final evaluation).
        features (list): List of feature names.
        target (str): Name of the target variable.
        max_models (int): Maximum number of models to build in AutoML.
        max_runtime_secs (int): Maximum time in seconds for the AutoML run.

    Returns:
        dict: A dictionary containing 'train_r2', 'val_r2', 'test_r2', and 'n_pipelines_tested'.
    """
    print("\nInitializing H2O cluster...")
    try:
        # Initialize H2O cluster; nthreads=-1 uses all cores, max_mem_size can be adjusted
        h2o.init(nthreads=-1, max_mem_size="4G")
    except Exception as e:
        print(f"H2O initialization failed: {e}")
        print("Attempting to shut down any existing cluster and retry...")
        h2o.cluster().shutdown(prompt=False)
        h2o.init(nthreads=-1, max_mem_size="4G")


    # Convert pandas DataFrames to H2O Frames
    h_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    h_val = h2o.H2OFrame(pd.concat([X_val, y_val], axis=1))
    h_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

    # Define H2O AutoML parameters
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=42,
        sort_metric='rmse',
        exclude_algos=['DeepLearning']
    )

    print(f"Starting H2O AutoML stacked ensemble search with max_models={max_models}, max_runtime_secs={max_runtime_secs}...")
    # Train H2O AutoML
    aml.train(x=features, y=target, training_frame=h_train, validation_frame=h_val)

    print("\nH2O AutoML search complete. Displaying leaderboard.")
    leaderboard = aml.leaderboard.as_data_frame()
    print("H2O AutoML LEADERBOARD (top 5 by RMSE):")
    print(leaderboard[['model_id', 'rmse', 'mae']].head())

    best_h2o_model = aml.leader

    # Predict on all sets
    train_pred_h2o = best_h2o_model.predict(h_train[features])
    val_pred_h2o = best_h2o_model.predict(h_val[features])
    test_pred_h2o = best_h2o_model.predict(h_test[features])

    train_pred = train_pred_h2o.as_data_frame().values.flatten()
    val_pred = val_pred_h2o.as_data_frame().values.flatten()
    test_pred = test_pred_h2o.as_data_frame().values.flatten()

    # Calculate R2 scores
    h2o_train_r2 = r2_score(y_train, train_pred)
    h2o_val_r2 = r2_score(y_val, val_pred)
    h2o_test_r2 = r2_score(y_test, test_pred)

    print(f"\nH2O BEST MODEL: {best_h2o_model.model_id}")
    print(f"In-sample R2 (Train): {h2o_train_r2:.4f}")
    print(f"Validation R2: {h2o_val_r2:.4f}")
    print(f"Held-out Test R2: {h2o_test_r2:.4f}")

    h2o_results = {
        'train_r2': h2o_train_r2,
        'val_r2': h2o_val_r2,
        'test_r2': h2o_test_r2,
        'n_pipelines_tested': len(leaderboard) # Actual number of models on leaderboard
    }

    print("\nShutting down H2O cluster...")
    h2o.cluster().shutdown(prompt=False)

    return h2o_results

def objective_xgboost(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Optuna objective function: finds optimal hyperparameters for XGBoost.
    Uses TimeSeriesSplit for temporal cross-validation.

    Args:
        trial (optuna.Trial): A trial object from Optuna.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        float: The mean squared error across time series cross-validation folds.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'random_state': 42
    }

    model = XGBRegressor(**params)

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for tr_idx, va_idx in tscv.split(X_train):
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[va_idx])
        scores.append(mean_squared_error(y_train.iloc[va_idx], pred))

    return np.mean(scores)

def run_optuna_xgboost_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                X_test: pd.DataFrame, n_trials: int = 50) -> dict:
    """
    Runs Optuna for hyperparameter optimization of an XGBoost Regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        X_test (pd.DataFrame): Test features (for final evaluation).
        n_trials (int): Number of optimization trials for Optuna.

    Returns:
        dict: A dictionary containing 'train_r2', 'val_r2', 'test_r2', and 'n_pipelines_tested'.
    """
    print("\nStarting Optuna Bayesian hyperparameter optimization for XGBoost...")

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_xgboost(trial, X_train, y_train),
                   n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna optimization complete.")
    print("\nOPTUNA RESULTS:")
    print(f"Best trial MSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Trials run: {len(study.trials)}")

    # Train the final XGBoost model with the best parameters on the full training set
    best_xgb = XGBRegressor(**study.best_params, random_state=42)
    best_xgb.fit(X_train, y_train)

    # Predict on all sets
    train_pred = best_xgb.predict(X_train)
    val_pred = best_xgb.predict(X_val)
    test_pred = best_xgb.predict(X_test)

    # Calculate R2 scores
    optuna_train_r2 = r2_score(y_train, train_pred)
    optuna_val_r2 = r2_score(y_val, val_pred)
    optuna_test_r2 = r2_score(y_test, test_pred)

    print(f"In-sample R2 (Train) with best Optuna XGBoost: {optuna_train_r2:.4f}")
    print(f"Validation R2 with best Optuna XGBoost: {optuna_val_r2:.4f}")
    print(f"Held-out Test R2 with best Optuna XGBoost: {optuna_test_r2:.4f}")

    # Visualize parameter importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        # fig.show() # Plotly figure, might not display in all environments.
        print("\nOptuna parameter importance plot generated (requires plotly).")
    except ImportError:
        print("\nInstall plotly to visualize parameter importances (e.g., !pip install plotly).")
    except Exception as e:
        print(f"\nCould not generate parameter importance plot: {e}")

    optuna_results = {
        'train_r2': optuna_train_r2,
        'val_r2': optuna_val_r2,
        'test_r2': optuna_test_r2,
        'n_pipelines_tested': len(study.trials)
    }
    return optuna_results

def overfitting_audit(model_name: str, train_r2: float, val_r2: float, test_r2: float,
                      n_pipelines_tested: int, features_used: list = None) -> tuple:
    """
    Performs a four-check overfitting audit for AutoML-discovered signals.

    Args:
        model_name (str): Name of the model being audited.
        train_r2 (float): R2 score on the training set.
        val_r2 (float): R2 score on the validation set.
        test_r2 (float): R2 score on the held-out test set.
        n_pipelines_tested (int): Number of unique pipelines/models explored.
        features_used (list, optional): List of features used by the model. Defaults to None.

    Returns:
        tuple: (overall_pass, overfit_ratio, bonferroni_alpha, test_pass, financial_sanity_pass)
    """
    print(f"\n===== OVERFITTING AUDIT: {model_name} ====")
    print("=" * 60)

    overall_pass = True

    # Check 1: Overfitting ratio
    if train_r2 > 0: # Avoid division by zero
        overfit_ratio = 1 - (val_r2 / train_r2)
    else:
        overfit_ratio = 1.0 # If train R2 is zero or negative, severe overfitting or no signal

    print("\nCHECK 1 - Overfitting ratio:")
    print(f"  Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")
    print(f"  Ratio (1 - val/train): {overfit_ratio:.2f}")
    overfit_pass = (overfit_ratio < 0.5) # A heuristic threshold
    print(f"  {'PASS (< 0.5)' if overfit_pass else 'FAIL (>= 0.5: severe overfitting)'}")
    if not overfit_pass: overall_pass = False

    # Check 2: Multiple-testing correction (Bonferroni)
    alpha_original = 0.05
    bonferroni_alpha = alpha_original / n_pipelines_tested
    print("\nCHECK 2 - Multiple-testing correction (Bonferroni):")
    print(f"  Pipelines tested: {n_pipelines_tested}")
    print(f"  Bonferroni threshold (for p-value): {bonferroni_alpha:.6f}")
    print(f"  (This check is conceptual; real p-value calculation for R2 is complex, but it highlights the stringency needed.)")
    bonferroni_pass = True # This check is qualitative/conceptual for this exercise.

    # Check 3: Test set (held-out future data)
    print("\nCHECK 3 - Held-out test performance:")
    print(f"  Test R2: {test_r2:.4f}")
    # Pass if test R2 is positive AND (val R2 is non-positive OR test R2 is at least 50% of positive val R2)
    test_pass = (test_r2 > 0.0 and (val_r2 <= 0 or test_r2 >= val_r2 * 0.5))
    print(f"  {'PASS: Test R2 positive and >= 50% of val R2 (if val R2 > 0)' if test_pass else 'FAIL: Test R2 collapsed or negative'}")
    if not test_pass: overall_pass = False

    # Check 4: Does it make financial sense? (Requires human judgment)
    print("\nCHECK 4 - Financial sanity (requires human judgment):")
    print(f"  Does the model use sensible features? [HUMAN REVIEW REQUIRED]")
    if features_used:
        print(f"  Features identified/used by model: {features_used} [HUMAN REVIEW REQUIRED]")
    else:
        print(f"  Feature importances not directly available for generic pipelines, but conceptually important. [HUMAN REVIEW REQUIRED]")
    print(f"  Would you bet your own money on this signal? [HUMAN REVIEW REQUIRED]")
    financial_sanity_pass = '[HUMAN REVIEW REQUIRED]' # This is a placeholder for analyst decision

    print(f"\nOVERALL AUDIT STATUS: {'CONDITIONALLY APPROVED (pending human review)' if overall_pass else 'REJECTED (overfitting detected / poor out-of-sample performance)'}")
    print("=" * 60)
    return overall_pass, overfit_ratio, bonferroni_alpha, test_pass, financial_sanity_pass

def human_vs_automl_comparison(h2o_results: dict, optuna_results: dict):
    """
    Compares conceptual human-designed XGBoost to AutoML discoveries across various dimensions.

    Args:
        h2o_results (dict): Results from the H2O AutoML pipeline.
        optuna_results (dict): Results from the Optuna XGBoost pipeline.
    """
    print("\n\n===== HUMAN vs. AutoML COMPARISON ====")
    print("=" * 65)
    print(f"{'Dimension':<30s}{'Manual (XGBoost Baseline)':>30s}{'AutoML (H2O, Optuna)':>30s}")
    print("-" * 90)

    total_pipelines_tested = h2o_results['n_pipelines_tested'] + optuna_results['n_pipelines_tested']

    comparisons = [
        ('Development time', '4-8 hours', '30-60 minutes'),
        ('Pipelines explored', '3-5 (manual iterations)', f'~{total_pipelines_tested}'),
        ('Feature engineering', 'Domain-driven, meticulous', 'Automated/Embedded in pipeline'),
        ('Model selection', 'XGBoost (chosen by expert)', 'Best of many (ensembles, genetic)'),
        ('Hyperparameters', 'Manual/grid-search limited', 'Bayesian/genetic optimization'),
        ('Interpretability', 'Full understanding', 'Black-box pipeline (requires XAI)'),
        ('Overfitting risk', 'Moderate (with diligence)', 'HIGH (many tests, subtle)'),
        ('Governance burden', 'Standard (Tier 2)', 'Elevated (Tier 1-2, crucial)'),
        ('Human judgment', 'Embedded throughout process', 'Applied post-hoc (audit crucial)'),
        ('Typical test R2 lift', 'Baseline R2 +0 to +0.005', 'Potentially higher, but higher risk of spuriousness')
    ]

    for dim, manual, auto in comparisons:
        print(f"{dim:<30s}{manual:>30s}{auto:>30s}")

    print("\nVERDICT:")
    print("AutoML saves significant time and explores a vastly larger solution space, potentially uncovering non-obvious signals.")
    print("However, its primary value is HYPOTHESIS GENERATION, not automatic strategy production.")
    print("The human analyst remains essential for rigorous validation, interpretation, and ultimate judgment.")
    print("=" * 90)

def course_final_synthesis():
    """Summarizes the key takeaways regarding AutoML and human oversight in finance."""
    print("\n\n===== COURSE SYNTHESIS: THE COMPLETE AI TOOLKIT ====")
    print("=" * 60)
    print("\n60 case studies. 5 days. 20 topics. One message:")
    print("\nAI is the most powerful tool financial professionals have ever had.")
    print("But a tool is only as good as the professional who wields it—with competence, integrity, diligence, and responsibility.")
    print("\nBuild with confidence. Govern with rigor.")
    print("Deploy with humility. Serve clients with integrity.")
    print("\nThe course is complete. The work begins now, empowering the AI-enabled financial professional.")
    print("=" * 60)

def run_full_automl_analysis(n_stocks: int = 200, n_months: int = 120,
                             train_months_end: int = 72, val_months_end: int = 96,
                             h2o_max_models: int = 10, h2o_max_runtime_secs: int = 300,
                             optuna_n_trials: int = 50):
    """
    Orchestrates the entire AutoML analysis pipeline from data generation to audit and synthesis.

    Args:
        n_stocks (int): Number of synthetic stocks.
        n_months (int): Number of synthetic months.
        train_months_end (int): Exclusive upper bound for training months.
        val_months_end (int): Exclusive upper bound for validation months.
        h2o_max_models (int): Max models for H2O AutoML.
        h2o_max_runtime_secs (int): Max runtime for H2O AutoML.
        optuna_n_trials (int): Number of trials for Optuna XGBoost.
    """
    set_global_settings()

    print("Generating synthetic financial data...")
    df = generate_panel_data(n_stocks, n_months)

    features = [col for col in df.columns if col not in ['month', 'stock_id', 'forward_return']]
    target = 'forward_return'

    print("\nSplitting data into training, validation, and test sets...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_panel_data(
        df, features, target, train_months_end, val_months_end
    )

    # --- Run H2O AutoML ---
    h2o_results = run_h2o_automl_pipeline(
        X_train, y_train, X_val, y_val, X_test, features, target,
        max_models=h2o_max_models, max_runtime_secs=h2o_max_runtime_secs
    )

    # --- Run Optuna XGBoost ---
    optuna_results = run_optuna_xgboost_pipeline(
        X_train, y_train, X_val, y_val, X_test, n_trials=optuna_n_trials
    )

    # --- Perform Overfitting Audit ---
    print("\nRunning the four-check overfitting audit on all AutoML-discovered signals...\n")

    h2o_audit_status, h2o_overfit_ratio, h2o_bonferroni, h2o_test_pass, _ = overfitting_audit(
        'H2O AutoML (Stacked Ensembles)', h2o_results['train_r2'], h2o_results['val_r2'],
        h2o_results['test_r2'], h2o_results['n_pipelines_tested'], features_used=features # H2O uses all features
    )

    optuna_audit_status, optuna_overfit_ratio, optuna_bonferroni, optuna_test_pass, _ = overfitting_audit(
        'Optuna XGBoost (Bayesian Opt.)', optuna_results['train_r2'], optuna_results['val_r2'],
        optuna_results['test_r2'], optuna_results['n_pipelines_tested'], features_used=features # XGBoost uses all features
    )

    # Store audit results for comparison table (can be printed directly or returned)
    audit_summary = {
        'H2O AutoML': {
            'Train R2': f"{h2o_results['train_r2']:.4f}",
            'Validation R2': f"{h2o_results['val_r2']:.4f}",
            'Held-out Test R2': f"{h2o_results['test_r2']:.4f}",
            'Overfit Ratio': f"{h2o_overfit_ratio:.2f} ({'Pass' if h2o_audit_status and h2o_overfit_ratio < 0.5 else 'Fail'})",
            'Bonferroni Alpha': f"{h2o_bonferroni:.6f}",
            'Test Set Check': f"{'Pass' if h2o_test_pass else 'Fail'}",
            'Overall Status': 'Approved' if h2o_audit_status else 'Rejected'
        },
        'Optuna XGBoost': {
            'Train R2': f"{optuna_results['train_r2']:.4f}",
            'Validation R2': f"{optuna_results['val_r2']:.4f}",
            'Held-out Test R2': f"{optuna_results['test_r2']:.4f}",
            'Overfit Ratio': f"{optuna_overfit_ratio:.2f} ({'Pass' if optuna_audit_status and optuna_overfit_ratio < 0.5 else 'Fail'})",
            'Bonferroni Alpha': f"{optuna_bonferroni:.6f}",
            'Test Set Check': f"{'Pass' if optuna_test_pass else 'Fail'}",
            'Overall Status': 'Approved' if optuna_audit_status else 'Rejected'
        }
    }

    print("\n----- AUDIT SUMMARY TABLE -----")
    for model, results in audit_summary.items():
        print(f"\nModel: {model}")
        for key, value in results.items():
            print(f"  {key:<20}: {value}")
    print("------------------------------")


    # --- Comparison and Synthesis ---
    human_vs_automl_comparison(h2o_results, optuna_results)
    course_final_synthesis()

if __name__ == "__main__":
    # Example usage when running the script directly
    N_STOCKS = 200
    N_MONTHS = 120
    TRAIN_MONTHS_END = 72 # months 0-71
    VAL_MONTHS_END = 96   # months 72-95, test is 96-119

    H2O_MAX_MODELS = 10
    H2O_MAX_RUNTIME_SECS = 300 # 5 minutes

    OPTUNA_N_TRIALS = 50

    run_full_automl_analysis(
        n_stocks=N_STOCKS,
        n_months=N_MONTHS,
        train_months_end=TRAIN_MONTHS_END,
        val_months_end=VAL_MONTHS_END,
        h2o_max_models=H2O_MAX_MODELS,
        h2o_max_runtime_secs=H2O_MAX_RUNTIME_SECS,
        optuna_n_trials=OPTUNA_N_TRIALS
    )
