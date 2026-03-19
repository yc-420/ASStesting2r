import os
import warnings
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore") #ignore
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "garments_worker_productivity.csv")
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")

QUARTER_CATS = ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
DEPARTMENT_CATS = ["finishing", "sewing"]
DAY_CATS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]

st.set_page_config(page_title="Garment Worker Productivity Dashboard", layout="wide")


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


@st.cache_data
def load_raw_data():
    original_df = pd.read_csv(DATA_PATH)
    original_missing_wip = int(original_df["wip"].isna().sum())

    df = original_df.copy()
    df["department"] = (
        df["department"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"sweing": "sewing"})
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df["date"].dt.day_name()
    df["wip"] = df["wip"].fillna(0)

    return df, original_missing_wip


@st.cache_data
def build_model_dataframe():
    df, _ = load_raw_data()
    model_df = df.copy()

    model_df["quarter"] = pd.Categorical(model_df["quarter"], categories=QUARTER_CATS)
    model_df["department"] = pd.Categorical(model_df["department"], categories=DEPARTMENT_CATS)
    model_df["day"] = pd.Categorical(model_df["day"], categories=DAY_CATS)

    model_df = pd.get_dummies(
        model_df,
        columns=["quarter", "department", "day"],
        drop_first=True,
    )
    model_df = model_df.drop(columns=["date"])
    model_df.columns = model_df.columns.str.strip()
    return model_df


@st.cache_data
def get_column_details():
    return pd.DataFrame([
        ["date", "Production date", "datetime"],
        ["quarter", "Production quarter", "categorical"],
        ["department", "Department name", "categorical"],
        ["team", "Team number", "numeric"],
        ["targeted_productivity", "Target productivity rate", "numeric"],
        ["smv", "Standard Minute Value", "numeric"],
        ["wip", "Work in progress", "numeric"],
        ["over_time", "Overtime minutes", "numeric"],
        ["incentive", "Incentive amount", "numeric"],
        ["idle_time", "Idle time", "numeric"],
        ["idle_men", "Number of idle workers", "numeric"],
        ["no_of_style_change", "Count of style changes", "numeric"],
        ["no_of_workers", "Number of workers", "numeric"],
        ["actual_productivity", "Actual productivity achieved", "target"],
        ["day", "Day name derived from date", "derived categorical"],
    ], columns=["Feature", "Description", "Type"])


@st.cache_resource
def train_and_evaluate_models():
    model_df = build_model_dataframe()
    X = model_df.drop("actual_productivity", axis=1)
    y = model_df["actual_productivity"]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    predictions: Dict[str, np.ndarray] = {}
    best_models = {}

    models = {
        "Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
    }

    for model_name, model in models.items():
        model.fit(Xtrain, ytrain)
        pred = model.predict(Xtest)
        mae, rmse, r2 = evaluate_model(ytest, pred)
        cv_rmse = -cross_val_score(
            model, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error"
        ).mean()
        cv_r2 = cross_val_score(model, Xtrain, ytrain, cv=5, scoring="r2").mean()

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "CV_RMSE": cv_rmse,
            "CV_R2": cv_r2,
            "Best Parameters": "Mean strategy" if model_name == "Baseline" else "Default",
        })
        predictions[model_name] = pred
        best_models[model_name] = model

    ridge_grid = GridSearchCV(
        Ridge(),
        {"alpha": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    ridge_grid.fit(Xtrain, ytrain)
    best_ridge = ridge_grid.best_estimator_
    pred_ridge = best_ridge.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_ridge)
    cv_rmse = -cross_val_score(
        best_ridge, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error"
    ).mean()
    cv_r2 = cross_val_score(best_ridge, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Ridge Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": str(ridge_grid.best_params_),
    })
    predictions["Ridge Regression"] = pred_ridge
    best_models["Ridge Regression"] = best_ridge

    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    dt_grid.fit(Xtrain, ytrain)
    best_dt = dt_grid.best_estimator_
    pred_dt = best_dt.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_dt)
    cv_rmse = -cross_val_score(
        best_dt, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error"
    ).mean()
    cv_r2 = cross_val_score(best_dt, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Decision Tree",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": str(dt_grid.best_params_),
    })
    predictions["Decision Tree"] = pred_dt
    best_models["Decision Tree"] = best_dt

    best_rf = None
    if os.path.exists(RF_MODEL_PATH):
        try:
            best_rf = joblib.load(RF_MODEL_PATH)
            best_rf.fit(Xtrain, ytrain)
            rf_best_params = str({
                k: best_rf.get_params()[k]
                for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
                if k in best_rf.get_params()
            })
        except Exception:
            best_rf = None

    if best_rf is None:
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        rf_grid.fit(Xtrain, ytrain)
        best_rf = rf_grid.best_estimator_
        rf_best_params = str(rf_grid.best_params_)

    pred_rf = best_rf.predict(Xtest)
    mae, rmse, r2 = evaluate_model(ytest, pred_rf)
    cv_rmse = -cross_val_score(
        best_rf, Xtrain, ytrain, cv=5, scoring="neg_root_mean_squared_error"
    ).mean()
    cv_r2 = cross_val_score(best_rf, Xtrain, ytrain, cv=5, scoring="r2").mean()
    results.append({
        "Model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "CV_R2": cv_r2,
        "Best Parameters": rf_best_params,
    })
    predictions["Random Forest"] = pred_rf
    best_models["Random Forest"] = best_rf

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    return {
        "Xtrain": Xtrain,
        "Xtest": Xtest,
        "ytrain": ytrain,
        "ytest": ytest,
        "results_df": results_df,
        "predictions": predictions,
        "best_models": best_models,
        "feature_columns": list(X.columns),
    }


def prepare_prediction_input(input_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = input_df.copy()
    df["department"] = (
        df["department"].astype(str).str.strip().str.lower().replace({"sweing": "sewing"})
    )
    df["quarter"] = pd.Categorical(df["quarter"], categories=QUARTER_CATS)
    df["department"] = pd.Categorical(df["department"], categories=DEPARTMENT_CATS)
    df["day"] = pd.Categorical(df["day"], categories=DAY_CATS)
    df["wip"] = df["wip"].fillna(0)

    df = pd.get_dummies(df, columns=["quarter", "department", "day"], drop_first=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]
    return df


def get_prediction_status(gap: float):
    if gap >= 0.05:
        return "Likely to exceed target", "success"
    elif gap >= 0:
        return "Likely to meet target", "success"
    elif gap >= -0.05:
        return "Slightly below target", "warning"
    else:
        return "Significantly below target", "error"


raw_df, original_missing_wip = load_raw_data()
model_bundle = train_and_evaluate_models()
results_df = model_bundle["results_df"]
feature_cols = model_bundle["feature_columns"]
best_models = model_bundle["best_models"]
best_model_row = results_df.sort_values("RMSE").iloc[0]

st.title("Garment Worker Productivity Dashboard")
st.caption(
    "BMDS2003 Data Science Project — EDA, model comparison, single prediction, and batch prediction"
)

menu = st.radio(
    "Navigation",
    [
        "Overview",
        "Data Exploration",
        "Model Performance",
        "Single Prediction",
        "Batch Prediction",
        "About",
    ],
    horizontal=True,
)

if menu == "Overview":
    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Original Missing WIP", original_missing_wip)
    c4.metric("Best Model", best_model_row["Model"])

    st.subheader("Business Objective")
    st.write(
        "This project predicts actual productivity of garment factory teams so that production managers "
        "can estimate likely performance, detect possible underachievement, and improve workforce or production planning decisions."
    )

    st.subheader("Why this prototype matters")
    st.info(
        "This dashboard helps users understand the dataset, compare multiple machine learning models, "
        "and generate productivity predictions for both single and batch records."
    )

    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Column Details")
    st.dataframe(get_column_details(), use_container_width=True)

    st.subheader("Data Preparation Summary")
    st.markdown(
        "- Corrected `sweing` to `sewing`.\n"
        "- Converted `date` to datetime and derived `day` from date.\n"
        "- Filled missing `wip` values with 0.\n"
        "- Applied one-hot encoding to categorical features for modelling.\n"
        "- Used a fixed category mapping so single and batch predictions remain consistent with training."
    )

elif menu == "Data Exploration":
    st.header("Data Exploration")

    eda_df = raw_df.copy()

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            dept_filter = st.selectbox(
                "Department",
                ["All"] + sorted(eda_df["department"].dropna().unique().tolist()),
            )
        with f2:
            quarter_filter = st.selectbox(
                "Quarter",
                ["All"] + sorted(eda_df["quarter"].dropna().unique().tolist()),
            )
        with f3:
            day_filter = st.selectbox(
                "Day",
                ["All"] + DAY_CATS,
            )

    filtered_df = eda_df.copy()
    if dept_filter != "All":
        filtered_df = filtered_df[filtered_df["department"] == dept_filter]
    if quarter_filter != "All":
        filtered_df = filtered_df[filtered_df["quarter"] == quarter_filter]
    if day_filter != "All":
        filtered_df = filtered_df[filtered_df["day"] == day_filter]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered Records", len(filtered_df))
    m2.metric("Average Productivity", f"{filtered_df['actual_productivity'].mean():.3f}")
    m3.metric("Average Target Productivity", f"{filtered_df['targeted_productivity'].mean():.3f}")

    st.subheader("1. Distribution of the Target Variable")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(filtered_df["actual_productivity"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Actual Productivity")
        ax.set_xlabel("Actual Productivity")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.caption(
            "Interpretation: This histogram shows the overall distribution of actual productivity. "
            "Most observations are concentrated around the middle to higher productivity range, "
            "indicating that many teams achieve moderate to strong performance."
        )

    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=filtered_df["actual_productivity"], ax=ax)
        ax.set_title("Boxplot of Actual Productivity")
        ax.set_xlabel("Actual Productivity")
        st.pyplot(fig)
        st.caption(
            "Interpretation: This boxplot summarizes the spread of actual productivity and highlights outliers. "
            "It helps identify whether unusually low or high productivity values exist in the dataset."
        )

    st.subheader("2. Relationship with Key Numeric Variables")
    c3, c4 = st.columns(2)

    with c3:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            x="targeted_productivity",
            y="actual_productivity",
            data=filtered_df,
            alpha=0.65,
            ax=ax,
        )
        ax.set_title("Targeted Productivity vs Actual Productivity")
        ax.set_xlabel("Targeted Productivity")
        ax.set_ylabel("Actual Productivity")
        st.pyplot(fig)
        st.caption(
            "Interpretation: This scatter plot shows the relationship between targeted productivity and actual productivity. "
            "A general upward tendency suggests that higher targets are often associated with higher actual output, "
            "although other operational factors still influence performance."
        )

    with c4:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            x="over_time",
            y="actual_productivity",
            data=filtered_df,
            alpha=0.65,
            ax=ax,
        )
        ax.set_title("Over Time vs Actual Productivity")
        ax.set_xlabel("Over Time")
        ax.set_ylabel("Actual Productivity")
        st.pyplot(fig)
        st.caption(
            "Interpretation: This chart illustrates how overtime relates to actual productivity. "
            "It helps evaluate whether additional working time improves productivity consistently "
            "or whether excessive overtime may produce weaker returns."
        )

    c5, c6 = st.columns(2)

    with c5:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            x="no_of_workers",
            y="actual_productivity",
            data=filtered_df,
            alpha=0.65,
            ax=ax,
        )
        ax.set_title("Number of Workers vs Actual Productivity")
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Actual Productivity")
        st.pyplot(fig)
        st.caption(
            "Interpretation: This plot examines the relationship between team size and actual productivity. "
            "The clustered pattern suggests that increasing the number of workers does not always guarantee higher productivity, "
            "possibly due to coordination challenges or diminishing returns."
        )

    with c6:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="department", y="actual_productivity", data=filtered_df, ax=ax)
        ax.set_title("Actual Productivity by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Actual Productivity")
        st.pyplot(fig)
        st.caption(
            "Interpretation: This boxplot compares actual productivity across departments. "
            "Differences between departments may reflect variation in workflow, task type, or production efficiency."
        )

    st.subheader("3. Categorical and Structural Insight")
    c7, c8 = st.columns(2)

    with c7:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="quarter", y="actual_productivity", data=filtered_df, ax=ax)
        ax.set_title("Actual Productivity by Quarter")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Actual Productivity")
        st.pyplot(fig)
        st.caption(
            "Interpretation: This chart compares productivity across quarters. "
            "It helps identify whether productivity differs by production period and whether time-based operational factors may matter."
        )

    with c8:
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        st.caption(
            "Interpretation: The correlation heatmap shows the strength and direction of relationships among numerical variables. "
            "It supports feature selection and helps identify variables that are more strongly related to actual productivity."
        )

    team_avg = filtered_df.groupby("team")["actual_productivity"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    team_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Actual Productivity by Team")
    ax.set_xlabel("Team")
    ax.set_ylabel("Average Actual Productivity")
    st.pyplot(fig)
    st.caption(
        "Interpretation: This bar chart compares the average actual productivity of each team. "
        "It highlights performance differences across teams and may support benchmarking or management review."
    )

elif menu == "Model Performance":
    st.header("Model Performance")

    st.success(
        f"Best Performing Model: {best_model_row['Model']} "
        f"(RMSE = {best_model_row['RMSE']:.4f}, R² = {best_model_row['R2']:.4f})"
    )

    st.info(
        "All models are compared here because this is the most appropriate section for benchmarking and evaluation."
    )

    display_df = results_df.copy()
    for col in ["MAE", "RMSE", "R2", "CV_RMSE", "CV_R2"]:
        display_df[col] = display_df[col].round(4)
    st.dataframe(display_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=results_df, x="Model", y="RMSE", ax=ax)
        ax.set_title("RMSE Comparison Across Models")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE (lower is better)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        st.caption(
            "Interpretation: Lower RMSE indicates smaller prediction error. "
            "Therefore, models with lower RMSE provide more accurate productivity predictions."
        )

    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=results_df, x="Model", y="R2", ax=ax)
        ax.set_title("R² Comparison Across Models")
        ax.set_xlabel("Model")
        ax.set_ylabel("R² (higher is better)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        st.caption(
            "Interpretation: Higher R² indicates that the model explains more variation in actual productivity. "
            "This helps assess overall goodness of fit."
        )

    rf_model = best_models["Random Forest"]
    if hasattr(rf_model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": model_bundle["Xtrain"].columns,
            "Importance": rf_model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=fi, x="Importance", y="Feature", ax=ax)
        ax.set_title("Top 15 Feature Importances (Random Forest)")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)
        st.caption(
            "Interpretation: Features with higher importance contribute more strongly to Random Forest predictions. "
            "These variables are most influential in productivity forecasting."
        )

    selected_model = st.selectbox(
        "Choose a model for detailed diagnostic plots",
        [m for m in model_bundle["predictions"].keys() if m != "Baseline"],
        index=3,
    )

    ytest = model_bundle["ytest"]
    ypred = model_bundle["predictions"][selected_model]

    d1, d2 = st.columns(2)
    with d1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(ytest, ypred, alpha=0.7)
        min_v = min(float(np.min(ytest)), float(np.min(ypred)))
        max_v = max(float(np.max(ytest)), float(np.max(ypred)))
        ax.plot([min_v, max_v], [min_v, max_v], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted - {selected_model}")
        st.pyplot(fig)
        st.caption(
            "Interpretation: Points closer to the diagonal line indicate more accurate predictions. "
            "A large distance from the line indicates higher prediction error."
        )

    with d2:
        residuals = ytest - ypred
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=ypred, y=residuals, ax=ax)
        ax.axhline(0, linestyle="--", color="red")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"Residual Plot - {selected_model}")
        st.pyplot(fig)
        st.caption(
            "Interpretation: A well-fitted model shows residuals randomly scattered around zero. "
            "Strong visible patterns suggest that the model may still miss some structure in the data."
        )

elif menu == "Single Prediction":
    st.header("Single Prediction")
    st.write("Enter production information to estimate actual productivity.")

    submitted = False

    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        with st.form("single_prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                team = st.number_input("Team", min_value=1, max_value=50, value=8)
                targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.80, 0.01)
                smv = st.number_input("SMV", min_value=0.0, value=26.16)
                wip = st.number_input("WIP", min_value=0.0, value=1108.0)
            with c2:
                over_time = st.number_input("Over Time", min_value=0.0, value=7080.0)
                incentive = st.number_input("Incentive", min_value=0.0, value=98.0)
                idle_time = st.number_input("Idle Time", min_value=0.0, value=0.0)
                idle_men = st.number_input("Idle Men", min_value=0.0, value=0.0)
            with c3:
                no_of_style_change = st.number_input("No. of Style Change", min_value=0, value=0)
                no_of_workers = st.number_input("No. of Workers", min_value=1.0, value=59.0)
                quarter = st.selectbox("Quarter", QUARTER_CATS)
                department = st.selectbox("Department", DEPARTMENT_CATS)
                day = st.selectbox("Day", DAY_CATS)

            model_choice = st.selectbox(
                "Primary Model for Prediction",
                ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest"],
                index=3,
            )
            submitted = st.form_submit_button("Generate Prediction")

    with right_col:
        st.subheader("Prediction Result")
        st.caption(
            "The input selections are shown on the left, while the prediction output is shown on the right "
            "to make the interface easier to explain during presentation."
        )

        if submitted:
            raw = {
                "team": team,
                "targeted_productivity": targeted_productivity,
                "smv": smv,
                "wip": wip,
                "over_time": over_time,
                "incentive": incentive,
                "idle_time": idle_time,
                "idle_men": idle_men,
                "no_of_style_change": no_of_style_change,
                "no_of_workers": no_of_workers,
                "quarter": quarter,
                "department": department,
                "day": day,
            }

            pred_input = prepare_prediction_input(pd.DataFrame([raw]), feature_cols)
            model = best_models[model_choice]
            pred = float(model.predict(pred_input)[0])
            gap = pred - targeted_productivity
            status_text, status_type = get_prediction_status(gap)

            summary_df = pd.DataFrame({
                "Metric": ["Selected Model", "Target Productivity", "Predicted Productivity", "Gap to Target"],
                "Value": [
                    model_choice,
                    f"{targeted_productivity:.3f}",
                    f"{pred:.3f}",
                    f"{gap:.3f}",
                ],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            if status_type == "success":
                st.success(f"Status: {status_text}")
            elif status_type == "warning":
                st.warning(f"Status: {status_text}")
            else:
                st.error(f"Status: {status_text}")

            if gap >= 0:
                st.markdown(
                    "**Interpretation:** Based on the selected production conditions, "
                    "the team is likely to meet or exceed the target productivity."
                )
                st.markdown(
                    "**Managerial implication:** Current operating conditions appear sufficient, "
                    "although managers should still monitor consistency and operational stability."
                )
            else:
                st.markdown(
                    "**Interpretation:** Based on the selected production conditions, "
                    "the team may fall below the target productivity."
                )
                st.markdown(
                    "**Managerial implication:** Managers may need to review staffing efficiency, workload balance, "
                    "or other production conditions to improve expected performance."
                )

            st.info(
                "This prediction should be used as decision support rather than a guaranteed outcome, "
                "because actual productivity can still be affected by factors not included in the dataset."
            )

            all_preds = {}
            for model_name in ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest"]:
                all_preds[model_name] = float(best_models[model_name].predict(pred_input)[0])

            compare_df = pd.DataFrame({
                "Model": list(all_preds.keys()),
                "Predicted Productivity": list(all_preds.values()),
            }).sort_values("Predicted Productivity", ascending=False)
            compare_df["Predicted Productivity"] = compare_df["Predicted Productivity"].round(4)

            st.subheader("All Model Predictions for This Input")
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            st.caption(
                "This table compares predictions from all trained models for the same input record. "
                "It supports deeper analysis without making the main result panel confusing."
            )

        else:
            st.write("Submit the form to generate a prediction result.")

elif menu == "Batch Prediction":
    st.header("Batch Prediction")
    st.write(
        "Upload a CSV file with multiple production records to generate productivity predictions."
    )

    template_df = pd.DataFrame([{
        "team": 8,
        "targeted_productivity": 0.80,
        "smv": 26.16,
        "wip": 1108,
        "over_time": 7080,
        "incentive": 98,
        "idle_time": 0,
        "idle_men": 0,
        "no_of_style_change": 0,
        "no_of_workers": 59,
        "quarter": "Quarter1",
        "department": "sewing",
        "day": "Monday",
    }])

    st.subheader("Sample Input Format")
    st.dataframe(template_df, use_container_width=True)
    st.download_button(
        "Download Sample Template",
        template_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    model_choice = st.selectbox(
        "Model for Batch Prediction",
        ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest"],
        index=3,
    )

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        batch_df.columns = batch_df.columns.str.strip()

        if "day" not in batch_df.columns and "date" in batch_df.columns:
            batch_df["date"] = pd.to_datetime(batch_df["date"], errors="coerce")
            batch_df["day"] = batch_df["date"].dt.day_name()

        keep_cols = [
            "team", "targeted_productivity", "smv", "wip", "over_time", "incentive",
            "idle_time", "idle_men", "no_of_style_change", "no_of_workers",
            "quarter", "department", "day",
        ]
        missing_cols = [c for c in keep_cols if c not in batch_df.columns]

        st.subheader("Uploaded Data Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Please use the sample template shown above.")
        else:
            pred_input = prepare_prediction_input(batch_df[keep_cols].copy(), feature_cols)
            model = best_models[model_choice]
            batch_df["predicted_actual_productivity"] = model.predict(pred_input)
            batch_df["predicted_actual_productivity"] = batch_df["predicted_actual_productivity"].round(4)

            st.subheader("Prediction Results")
            st.dataframe(batch_df.head(20), use_container_width=True)
            st.caption(
                "This batch module uses the same feature mapping as the training stage, "
                "so category encoding remains consistent for uploaded files."
            )

            st.download_button(
                "Download Results CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="batch_prediction_results.csv",
                mime="text/csv",
            )

elif menu == "About":
    st.header("About This Project")
    st.markdown(
        """
        ### Garment Worker Productivity Dashboard

        This dashboard was developed for the **BMDS2003 Data Science** assignment.

        **Project objective**  
        Predict actual productivity of garment factory teams using operational variables such as team size,
        overtime, incentive, work-in-progress, department, and quarter.

        **Techniques used**  
        - Data cleaning and preprocessing
        - Exploratory data analysis (EDA)
        - Dummy Regressor baseline
        - Linear Regression
        - Ridge Regression
        - Decision Tree Regressor
        - Random Forest Regressor
        - Model evaluation using MAE, RMSE, and R²
        - 5-Fold Cross Validation
        - Hyperparameter tuning using GridSearchCV

        **CRISP-DM flow**  
        Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment

        **Business value**  
        The model helps managers estimate whether a production setup is likely to meet its target productivity,
        supporting more informed staffing and production planning decisions.
        """
    )
