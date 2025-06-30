import streamlit as st
import openai
import pandas as pd
import os
import numpy as np
import re

st.title("Omniscience: Sports Betting Chatbot")

# Load your OpenAI API key from Streamlit secrets or environment variable
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found! Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# Comprehensive data cleaning function
def clean_data(df):
    """Clean and prepare the dataframe for analysis with comprehensive cleaning."""
    try:
        # Make a copy to avoid modifying the original
        cleaned = df.copy()
        cleaning_notes = []
        
        # 1. Standardize column names
        original_cols = list(cleaned.columns)
        cleaned.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in cleaned.columns]
        cleaning_notes.append("✓ Standardized column names")
        
        # 2. Handle missing values
        missing_before = cleaned.isna().sum().sum()
        
        # Identify numeric and string columns
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        string_cols = cleaned.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values in numeric columns
        for col in numeric_cols:
            if cleaned[col].isna().sum() > 0:
                median_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median_val)
        
        # Handle missing values in string columns
        for col in string_cols:
            cleaned[col] = cleaned[col].fillna('')
        
        missing_after = cleaned.isna().sum().sum()
        cleaning_notes.append(f"✓ Filled {missing_before - missing_after} missing values")
        
        # 3. Remove duplicate rows
        initial_rows = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        dupes_removed = initial_rows - len(cleaned)
        if dupes_removed > 0:
            cleaning_notes.append(f"✓ Removed {dupes_removed} duplicate rows")
        
        # 4. Convert date columns to datetime
        date_cols = [col for col in cleaned.columns if 'date' in col or 'time' in col or 'day' in col]
        date_converted = 0
        for col in date_cols:
            if col in string_cols:
                try:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors='coerce')
                    date_converted += 1
                except:
                    pass  # Skip if conversion fails
        
        if date_converted > 0:
            cleaning_notes.append(f"✓ Converted {date_converted} date columns")
        
        # 5. Handle outliers in numeric columns
        outliers_handled = 0
        for col in numeric_cols:
            # Skip columns with too few values or all same value
            if cleaned[col].nunique() <= 1 or len(cleaned[col].dropna()) < 5:
                continue
                
            # Cap outliers at 3 standard deviations
            mean, std = cleaned[col].mean(), cleaned[col].std()
            if std > 0:  # Avoid division by zero
                outlier_count = ((cleaned[col] < mean - 3*std) | (cleaned[col] > mean + 3*std)).sum()
                if outlier_count > 0:
                    cleaned[col] = cleaned[col].clip(mean - 3*std, mean + 3*std)
                    outliers_handled += outlier_count
        
        if outliers_handled > 0:
            cleaning_notes.append(f"✓ Handled {outliers_handled} outliers")
        
        # 6. Team name standardization
        team_mapping = {
            # MLB
            "NY Yankees": "New York Yankees", "NYY": "New York Yankees", "Yankees": "New York Yankees",
            "NY Mets": "New York Mets", "NYM": "New York Mets", "Mets": "New York Mets",
            "LA Dodgers": "Los Angeles Dodgers", "LAD": "Los Angeles Dodgers", "Dodgers": "Los Angeles Dodgers",
            "LA Angels": "Los Angeles Angels", "LAA": "Los Angeles Angels", "Angels": "Los Angeles Angels",
            "SF Giants": "San Francisco Giants", "SFG": "San Francisco Giants", "Giants": "San Francisco Giants",
            "CHI Cubs": "Chicago Cubs", "CHC": "Chicago Cubs", "Cubs": "Chicago Cubs",
            "CHI White Sox": "Chicago White Sox", "CWS": "Chicago White Sox", "White Sox": "Chicago White Sox",
            
            # NBA
            "LA Lakers": "Los Angeles Lakers", "LAL": "Los Angeles Lakers", "Lakers": "Los Angeles Lakers",
            "LA Clippers": "Los Angeles Clippers", "LAC": "Los Angeles Clippers", "Clippers": "Los Angeles Clippers",
            "GS Warriors": "Golden State Warriors", "GSW": "Golden State Warriors", "Warriors": "Golden State Warriors",
            "NY Knicks": "New York Knicks", "NYK": "New York Knicks", "Knicks": "New York Knicks",
            
            # NFL
            "NY Giants": "New York Giants", "NYG": "New York Giants", "Giants": "New York Giants",
            "NY Jets": "New York Jets", "NYJ": "New York Jets", "Jets": "New York Jets",
            "LA Rams": "Los Angeles Rams", "LAR": "Los Angeles Rams", "Rams": "Los Angeles Rams",
            "LA Chargers": "Los Angeles Chargers", "LAC": "Los Angeles Chargers", "Chargers": "Los Angeles Chargers",
        }
        
        team_cols = [col for col in cleaned.columns if 'team' in col or 'opponent' in col]
        teams_standardized = 0
        
        for col in team_cols:
            if col in string_cols:
                # Count replacements before making them
                teams_standardized += cleaned[col].isin(team_mapping.keys()).sum()
                cleaned[col] = cleaned[col].replace(team_mapping)
        
        if teams_standardized > 0:
            cleaning_notes.append(f"✓ Standardized {teams_standardized} team names")
        
        # 7. Data type conversion for percentages and other formats
        pct_cols = [col for col in cleaned.columns if 'pct' in col or 'percentage' in col or '%' in col]
        pct_converted = 0
        
        for col in pct_cols:
            if col in string_cols:
                try:
                    # Remove % sign and convert to float
                    cleaned[col] = cleaned[col].str.rstrip('%').astype('float') / 100
                    pct_converted += 1
                except:
                    pass  # Skip if conversion fails
        
        if pct_converted > 0:
            cleaning_notes.append(f"✓ Converted {pct_converted} percentage columns")
        
        # 8. Extract numeric values from string columns that might contain them
        numeric_extracted = 0
        for col in string_cols:
            # Skip columns that are likely categorical
            if cleaned[col].nunique() < 10 or 'name' in col or 'team' in col:
                continue
                
            # Check if column contains numeric patterns
            if cleaned[col].str.contains(r'\d+\.?\d*').any():
                try:
                    # Extract numeric values using regex
                    new_col_name = f"{col}_numeric"
                    cleaned[new_col_name] = cleaned[col].str.extract(r'(\d+\.?\d*)').astype(float)
                    numeric_extracted += 1
                except:
                    pass  # Skip if extraction fails
        
        if numeric_extracted > 0:
            cleaning_notes.append(f"✓ Extracted numeric values from {numeric_extracted} text columns")
        
        # Return the cleaned dataframe and a summary message
        summary = "Data cleaning complete:\n" + "\n".join(cleaning_notes)
        return cleaned, summary
        
    except Exception as e:
        return df, f"Basic cleaning only: {str(e)}"

# Improved CSV upload with error handling and cleaning
uploaded_file = st.file_uploader("Upload your matchup data (CSV)", type="csv")
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        if len(data) > 0:
            st.success(f"Successfully loaded CSV with {len(data)} rows and {len(data.columns)} columns!")
            
            # Clean the data
            with st.spinner("Cleaning data..."):
                cleaned_data, cleaning_msg = clean_data(data)
                st.info(cleaning_msg)
                data = cleaned_data
            
            # Show data preview with tabs for original and cleaned
            tab1, tab2 = st.tabs(["Cleaned Data", "Original Data"])
            with tab1:
                st.write("Cleaned Data Preview:", data.head())
                st.write(f"Shape: {data.shape}")
            with tab2:
                original_data = pd.read_csv(uploaded_file)
                st.write("Original Data:", original_data.head())
                st.write(f"Shape: {original_data.shape}")
                
            # Show column info
            if st.checkbox("Show column information"):
                col_info = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes,
                    'Non-Null Count': data.count(),
                    'Unique Values': [data[col].nunique() for col in data.columns]
                })
                st.write("Column Information:", col_info)
        else:
            st.warning("The uploaded CSV appears to be empty.")
            data = None
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        data = None
else:
    data = None
    st.info("Upload a CSV file to get predictions based on your data.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask Omniscience anything about today's games, matchups, or props:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Simple prediction logic (replace with your model)
    if data is not None:
        try:
            # Just use the first row as an example (replace with your actual prediction logic)
            prediction = data.iloc[0].to_dict()
            insight = f"Based on your data, here's what I found: {prediction}"
        except Exception as e:
            insight = f"I couldn't make a prediction from your data: {e}"
    else:
        insight = "Please upload data for specific predictions. What would you like to know about sports betting?"

    # Prepare messages for OpenAI
    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    messages.append({"role": "system", "content": "You are Omniscience, an elite sports betting AI assistant. Provide insightful, accurate betting advice."})
    messages.append({"role": "assistant", "content": insight})

    # Call OpenAI Chat API (new syntax)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"I'm sorry, I encountered an error: {e}"})

# Display chat history
for msg in st.session_state.messages:
    st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")
