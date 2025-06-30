import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import datetime
import uuid
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Omniscience")

# Create storage directories if they don't exist
STORAGE_DIR = Path("./omniscience_data")
DATASETS_DIR = STORAGE_DIR / "datasets"
MODELS_DIR = STORAGE_DIR / "models"
RESULTS_DIR = STORAGE_DIR / "results"

for directory in [STORAGE_DIR, DATASETS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

st.title("Omniscience: Elite Sports Betting Analysis")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.data_info = None
    st.session_state.upload_status = None
    st.session_state.dataset_id = None
    st.session_state.saved_datasets = []
    st.session_state.analysis_results = {}

# Load saved datasets list
def load_saved_datasets():
    datasets = []
    for file_path in DATASETS_DIR.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                metadata = json.load(f)["metadata"]
                datasets.append({
                    "id": metadata["id"],
                    "name": metadata["name"],
                    "rows": metadata["rows"],
                    "date": metadata["date"],
                    "path": str(file_path)
                })
        except Exception as e:
            st.error(f"Error loading dataset {file_path}: {e}")
    return sorted(datasets, key=lambda x: x["date"], reverse=True)

# Save dataset with metadata
def save_dataset(data, name=None):
    if data is None:
        return None

    # Generate unique ID if not already assigned
    dataset_id = st.session_state.dataset_id or str(uuid.uuid4())

    # Create metadata
    timestamp = datetime.datetime.now().isoformat()
    if not name:
        name = f"Dataset_{timestamp}"

    metadata = {
        "id": dataset_id,
        "name": name,
        "rows": len(data),
        "columns": list(data.columns),
        "date": timestamp
    }

    # Save data with metadata
    file_path = DATASETS_DIR / f"{dataset_id}.json"
    with open(file_path, "w") as f:
        json.dump({
            "metadata": metadata,
            "data": data.to_dict(orient="records")
        }, f)

    st.session_state.dataset_id = dataset_id
    return dataset_id

# Load dataset by ID
def load_dataset(dataset_id):
    file_path = DATASETS_DIR / f"{dataset_id}.json"
    try:
        with open(file_path, "r") as f:
            dataset = json.load(f)
            data = pd.DataFrame(dataset["data"])
            metadata = dataset["metadata"]
            return data, metadata
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None

# Clean data function
@st.cache_data
def clean_data(df):
    """Clean data with progress tracking and optimizations"""
    try:
        # Make a copy to avoid modifying the original
        cleaned = df.copy()
        notes = []

        # Basic cleaning (fast operations first)
        cleaned.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in cleaned.columns]
        notes.append("✓ Standardized column names")

        # Remove duplicates (fast operation)
        initial_rows = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        dupes_removed = initial_rows - len(cleaned)
        if dupes_removed > 0:
            notes.append(f"✓ Removed {dupes_removed} duplicate rows")

        # Handle missing values (medium speed)
        for col in cleaned.columns:
            if cleaned[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                else:
                    cleaned[col] = cleaned[col].fillna('')
        notes.append("✓ Filled missing values")

        # Convert date columns
        date_cols = [col for col in cleaned.columns if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
        for col in date_cols:
            try:
                cleaned[col] = pd.to_datetime(cleaned[col])
                notes.append(f"✓ Converted {col} to datetime")
            except:
                pass

        # Return with basic info
        return cleaned, "Data cleaned successfully: " + ", ".join(notes)
    except Exception as e:
        return df, f"Basic cleaning only: {str(e)}"

# Generate basic insights from data
def generate_insights(data):
    insights = {}

    # Basic statistics
    insights["basic_stats"] = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        insights["basic_stats"]["numeric"] = data[numeric_cols].describe().to_dict()

    # Correlation analysis
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        insights["correlations"] = corr_matrix.to_dict()

        # Find highest correlations
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Only include strong correlations
                    high_corrs.append((col1, col2, corr))

        high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        insights["high_correlations"] = high_corrs

    # Time series analysis if date columns exist
    date_cols = [col for col in data.columns if pd.api.types.is_datetime64_dtype(data[col])]
    if date_cols and numeric_cols:
        insights["time_series"] = {}
        date_col = date_cols[0]  # Use the first date column

        # Group by date and calculate stats
        try:
            time_grouped = data.groupby(pd.Grouper(key=date_col, freq='D')).mean()
            insights["time_series"]["daily_means"] = time_grouped[numeric_cols[:3]].to_dict()  # First 3 numeric columns
        except:
            pass

    # Categorical analysis
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        insights["categorical"] = {}
        for col in cat_cols[:5]:  # Limit to first 5 categorical columns
            value_counts = data[col].value_counts().head(10).to_dict()
            insights["categorical"][col] = value_counts

    return insights

# Sidebar for data management
with st.sidebar:
    st.header("Data Management")

    # Load saved datasets
    st.session_state.saved_datasets = load_saved_datasets()

    # Option to select existing dataset
    if st.session_state.saved_datasets:
        st.subheader("Load Saved Dataset")
        dataset_options = {f"{d['name']} ({d['rows']} rows, {d['date'][:10]})": d["id"] for d in st.session_state.saved_datasets}
        selected_dataset = st.selectbox("Select a dataset", options=list(dataset_options.keys()))

        if st.button("Load Selected Dataset"):
            dataset_id = dataset_options[selected_dataset]
            data, metadata = load_dataset(dataset_id)
            if data is not None:
                st.session_state.data = data
                st.session_state.data_info = {
                    "rows": len(data),
                    "columns": len(data.columns),
                    "name": metadata["name"]
                }
                st.session_state.dataset_id = dataset_id
                st.session_state.upload_status = "complete"
                st.success(f"Loaded dataset: {metadata['name']}")
                st.rerun()

    # Upload new dataset
    st.subheader("Upload New Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    dataset_name = st.text_input("Dataset name (optional)")

    if uploaded_file and st.button("Process Dataset"):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Load data
            status_text.text("Loading data...")
            progress_bar.progress(25)
            data = pd.read_csv(uploaded_file)

            # Step 2: Basic validation
            status_text.text("Validating data...")
            progress_bar.progress(50)
            if len(data) == 0:
                st.warning("The uploaded CSV appears to be empty.")
            else:
                # Step 3: Clean data
                status_text.text("Cleaning data...")
                progress_bar.progress(75)
                cleaned_data, cleaning_msg = clean_data(data)

                # Step 4: Save and store in session state
                status_text.text("Saving data...")
                progress_bar.progress(90)

                # Save dataset
                dataset_id = save_dataset(cleaned_data, name=dataset_name or uploaded_file.name)

                # Update session state
                st.session_state.data = cleaned_data
                st.session_state.data_info = {
                    "rows": len(cleaned_data),
                    "columns": len(cleaned_data.columns),
                    "name": dataset_name or uploaded_file.name,
                    "cleaning_msg": cleaning_msg
                }
                st.session_state.dataset_id = dataset_id
                st.session_state.upload_status = "complete"

                progress_bar.progress(100)
                status_text.empty()
                st.success(f"✅ Successfully processed and saved {len(cleaned_data)} rows of data!")
                time.sleep(1)
                st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Main content area - Data Analysis and Insights
if st.session_state.data is not None and st.session_state.upload_status == "complete":
    st.header(f"Dataset: {st.session_state.data_info.get('name', 'Unnamed')}")
    st.info(f"{st.session_state.data_info['rows']} rows, {st.session_state.data_info['columns']} columns")

    # Data preview
    with st.expander("Data Preview", expanded=True):
        st.dataframe(st.session_state.data.head(10))
        if "cleaning_msg" in st.session_state.data_info:
            st.info(st.session_state.data_info["cleaning_msg"])

    # Data analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Stats", "Correlations", "Visualizations", "Predictions"])

    with tab1:
        st.subheader("Basic Statistics")

        # Column selection for stats
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect("Select columns for statistics", numeric_cols, default=numeric_cols[:5])

            if selected_cols:
                st.write("Descriptive Statistics:")
                st.dataframe(st.session_state.data[selected_cols].describe())

                # Show missing values
                missing = st.session_state.data[selected_cols].isna().sum()
                if missing.sum() > 0:
                    st.write("Missing Values:")
                    st.dataframe(missing)
        else:
            st.write("No numeric columns found in the dataset.")

    with tab2:
        st.subheader("Correlation Analysis")

        # Column selection for correlation
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            selected_cols = st.multiselect("Select columns for correlation", numeric_cols, default=numeric_cols[:5], key="corr_cols")

            if len(selected_cols) > 1:
                corr_matrix = st.session_state.data[selected_cols].corr()

                # Display correlation matrix
                st.write("Correlation Matrix:")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

                # Find and display highest correlations
                high_corrs = []
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        col1 = selected_cols[i]
                        col2 = selected_cols[j]
                        corr = corr_matrix.loc[col1, col2]
                        high_corrs.append((col1, col2, corr))

                high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

                st.write("Strongest Correlations:")
                for col1, col2, corr in high_corrs[:10]:  # Show top 10
                    st.write(f"{col1} vs {col2}: {corr:.3f}")

                # Correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
        else:
            st.write("Need at least 2 numeric columns for correlation analysis.")

    with tab3:
        st.subheader("Data Visualizations")

        # Column selection for visualization
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
        date_cols = [col for col in st.session_state.data.columns if pd.api.types.is_datetime64_dtype(st.session_state.data[col])]

        # Visualization type selector
        viz_type = st.selectbox("Select visualization type",
                               ["Distribution", "Scatter Plot", "Time Series", "Bar Chart"])

        if viz_type == "Distribution":
            if numeric_cols:
                col = st.selectbox("Select column", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(st.session_state.data[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

                # Show basic stats
                st.write(f"Mean: {st.session_state.data[col].mean():.3f}")
                st.write(f"Median: {st.session_state.data[col].median():.3f}")
                st.write(f"Std Dev: {st.session_state.data[col].std():.3f}")
            else:
                st.write("No numeric columns available for distribution plot.")

        elif viz_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Select X-axis", numeric_cols)
                col2 = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

                # Optional color by category
                color_by = None
                if cat_cols:
                    use_color = st.checkbox("Color by category")
                    if use_color:
                        color_by = st.selectbox("Select category for color", cat_cols)

                fig, ax = plt.subplots(figsize=(10, 6))
                if color_by:
                    scatter = sns.scatterplot(data=st.session_state.data, x=col1, y=col2, hue=color_by, ax=ax)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    scatter = sns.scatterplot(data=st.session_state.data, x=col1, y=col2, ax=ax)

                ax.set_title(f"{col2} vs {col1}")
                st.pyplot(fig)

                # Show correlation
                corr = st.session_state.data[[col1, col2]].corr().iloc[0, 1]
                st.write(f"Correlation: {corr:.3f}")
            else:
                st.write("Need at least 2 numeric columns for scatter plot.")

        elif viz_type == "Time Series":
            if date_cols and numeric_cols:
                date_col = st.selectbox("Select date column", date_cols)
                value_col = st.selectbox("Select value column", numeric_cols)

                # Group by date and calculate mean
                try:
                    # Make a copy to avoid SettingWithCopyWarning
                    plot_data = st.session_state.data[[date_col, value_col]].copy()
                    plot_data = plot_data.sort_values(by=date_col)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(plot_data[date_col], plot_data[value_col])
                    ax.set_title(f"{value_col} over Time")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Show trend
                    if len(plot_data) > 1:
                        first_val = plot_data[value_col].iloc[0]
                        last_val = plot_data[value_col].iloc[-1]
                        change = last_val - first_val
                        pct_change = (change / first_val) * 100 if first_val != 0 else float('inf')
                        st.write(f"Change: {change:.3f} ({pct_change:.2f}%)")
                except Exception as e:
                    st.error(f"Error creating time series: {e}")
            else:
                st.write("Need date and numeric columns for time series plot.")

        elif viz_type == "Bar Chart":
            if cat_cols and numeric_cols:
                cat_col = st.selectbox("Select category column", cat_cols)
                value_col = st.selectbox("Select value column", numeric_cols, key="bar_value")

                # Group by category and calculate mean
                grouped = st.session_state.data.groupby(cat_col)[value_col].mean().sort_values(ascending=False)

                # Limit to top N categories
                top_n = st.slider("Show top N categories", 5, 20, 10)
                grouped = grouped.head(top_n)

                fig, ax = plt.subplots(figsize=(12, 6))
                grouped.plot(kind='bar', ax=ax)
                ax.set_title(f"Average {value_col} by {cat_col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("Need categorical and numeric columns for bar chart.")

    with tab4:
        st.subheader("Predictions & Analysis")

        # Simple prediction options
        st.write("Select prediction type:")
        pred_type = st.radio("Prediction Type", ["Trend Analysis", "Simple Regression", "Win Probability"])

        if pred_type == "Trend Analysis":
            # Trend analysis on numeric columns
            if numeric_cols and date_cols:
                date_col = st.selectbox("Select date column for trend", date_cols)
                value_col = st.selectbox("Select value column for trend", numeric_cols, key="trend_value")

                try:
                    # Make a copy and sort by date
                    trend_data = st.session_state.data[[date_col, value_col]].copy().dropna()
                    trend_data = trend_data.sort_values(by=date_col)

                    # Calculate simple moving average
                    window = st.slider("Moving average window", 2, 20, 5)
                    if len(trend_data) >= window:
                        trend_data['MA'] = trend_data[value_col].rolling(window=window).mean()

                        # Plot original and trend
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(trend_data[date_col], trend_data[value_col], label='Original')
                        ax.plot(trend_data[date_col], trend_data['MA'], label=f'Moving Avg ({window})', linewidth=2)
                        ax.set_title(f"Trend Analysis: {value_col}")
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Calculate trend direction
                        if len(trend_data) > window*2:
                            first_half = trend_data['MA'].dropna().iloc[:len(trend_data)//2].mean()
                            second_half = trend_data['MA'].dropna().iloc[len(trend_data)//2:].mean()
                            trend_pct = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0

                            if trend_pct > 5:
                                st.success(f"Strong upward trend: +{trend_pct:.2f}%")
                            elif trend_pct > 0:
                                st.info(f"Slight upward trend: +{trend_pct:.2f}%")
                            elif trend_pct > -5:
                                st.info(f"Slight downward trend: {trend_pct:.2f}%")
                            else:
                                st.error(f"Strong downward trend: {trend_pct:.2f}%")
                    else:
                        st.warning(f"Need at least {window} data points for moving average.")
                except Exception as e:
                    st.error(f"Error in trend analysis: {e}")
            else:
                st.write("Need date and numeric columns for trend analysis.")

        elif pred_type == "Simple Regression":
            # Simple linear regression between two variables
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X variable (predictor)", numeric_cols)
                y_col = st.selectbox("Select Y variable (target)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

                try:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import train_test_split

                    # Prepare data
                    reg_data = st.session_state.data[[x_col, y_col]].dropna()
                    X = reg_data[x_col].values.reshape(-1, 1)
                    y = reg_data[y_col].values

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Fit model
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Plot results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(X_test, y_test, color='blue', label='Actual')
                    ax.scatter(X_test, y_pred, color='red', label='Predicted')
                    ax.plot([X.min(), X.max()], [model.predict([[X.min()]]), model.predict([[X.max()]])], color='green', label='Regression Line')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"Linear Regression: {y_col} vs {x_col}")
                    plt.legend()
                    st.pyplot(fig)

                    # Show model details
                    st.write(f"Coefficient: {model.coef_[0]:.4f}")
                    st.write(f"Intercept: {model.intercept_:.4f}")
                    st.write(f"Equation: {y_col} = {model.coef_[0]:.4f} × {x_col} + {model.intercept_:.4f}")

                    # Calculate R-squared
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R-squared: {r2:.4f}")

                    # Prediction input
                    st.subheader("Make a prediction")
                    x_input = st.number_input(f"Enter {x_col} value:", value=float(X.mean()))
                    prediction = model.predict([[x_input]])[0]
                    st.write(f"Predicted {y_col}: {prediction:.4f}")

                except Exception as e:
                    st.error(f"Error in regression analysis: {e}")
            else:
                st.write("Need at least 2 numeric columns for regression analysis.")

        elif pred_type == "Win Probability":
            # Win probability analysis
            st.write("Win probability analysis based on historical data")

            # Check if we have team/player columns
            potential_team_cols = [col for col in st.session_state.data.columns if 'team' in col.lower() or 'player' in col.lower() or 'name' in col.lower()]
            potential_outcome_cols = [col for col in st.session_state.data.columns if 'win' in col.lower() or 'loss' in col.lower() or 'result' in col.lower() or 'outcome' in col.lower()]

            if potential_team_cols and (numeric_cols or potential_outcome_cols):
                team_col = st.selectbox("Select team/player column", potential_team_cols)

                # Determine outcome column
                if potential_outcome_cols:
                    outcome_col = st.selectbox("Select outcome column", potential_outcome_cols)
                    outcome_type = "categorical"
                else:
                    outcome_col = st.selectbox("Select numeric outcome column", numeric_cols)
                    threshold = st.number_input("Threshold for success (values above this are considered wins)", value=0.5)
                    outcome_type = "numeric"

                try:
                    # Calculate win rates
                    if outcome_type == "categorical":
                        # For categorical outcomes
                        win_values = st.multiselect("Select values that count as 'win'",
                                                   st.session_state.data[outcome_col].unique(),
                                                   default=[val for val in st.session_state.data[outcome_col].unique() if 'win' in str(val).lower()])

                        win_rates = {}
                        for team in st.session_state.data[team_col].unique():
                            team_data = st.session_state.data[st.session_state.data[team_col] == team]
                            if len(team_data) > 0:
                                wins = team_data[team_data[outcome_col].isin(win_values)]
                                win_rate = len(wins) / len(team_data)
                                win_rates[team] = (win_rate, len(team_data))
                    else:
                        # For numeric outcomes
                        win_rates = {}
                        for team in st.session_state.data[team_col].unique():
                            team_data = st.session_state.data[st.session_state.data[team_col] == team]
                            if len(team_data) > 0:
                                wins = team_data[team_data[outcome_col] > threshold]
                                win_rate = len(wins) / len(team_data)
                                win_rates[team] = (win_rate, len(team_data))

                    # Sort by win rate
                    sorted_rates = sorted(win_rates.items(), key=lambda x: x[1][0], reverse=True)

                    # Display results
                    st.subheader("Win Probabilities")

                    # Create DataFrame for display
                    win_df = pd.DataFrame([
                        {"Team/Player": team, "Win Rate": f"{rate*100:.1f}%", "Sample Size": count}
                        for team, (rate, count) in sorted_rates
                    ])

                    st.dataframe(win_df)

                    # Plot win rates
                    fig, ax = plt.subplots(figsize=(12, 6))
                    teams = [team for team, _ in sorted_rates[:15]]  # Top 15 teams
                    rates = [rate for _, (rate, _) in sorted_rates[:15]]

                    bars = ax.bar(teams, rates)
                    ax.set_title("Win Probabilities")
                    ax.set_ylabel("Win Rate")
                    ax.set_ylim(0, 1)
                    plt.xticks(rotation=45, ha='right')

                    # Add percentage labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.0%}', ha='center', va='bottom')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Matchup predictor
                    st.subheader("Matchup Predictor")
                    team1 = st.selectbox("Select Team/Player 1", list(win_rates.keys()))
                    team2 = st.selectbox("Select Team/Player 2", list(win_rates.keys()), index=1 if len(win_rates) > 1 else 0)

                    if team1 != team2:
                        # Simple probability calculation based on win rates
                        team1_rate = win_rates[team1][0]
                        team2_rate = win_rates[team2][0]

                        # Calculate relative strength
                        total = team1_rate + team2_rate
                        if total > 0:
                            team1_prob = team1_rate / total
                            team2_prob = team2_rate / total
                        else:
                            team1_prob = team2_prob = 0.5

                        st.write(f"**Estimated win probability:**")
                        st.write(f"{team1}: {team1_prob:.1%}, {team2}: {team2_prob:.1%}")
                except Exception as e:
                    st.error(f"Error calculating win probabilities: {e}")
            else:
                st.write("Need appropriate team/player and outcome columns for win probability analysis.")
