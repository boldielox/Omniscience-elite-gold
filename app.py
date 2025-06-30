import streamlit as st
import openai
import pandas as pd
import os
import json
import datetime
import uuid
import time
from pathlib import Path

st.set_page_config(layout="wide", page_title="Omniscience")

# Create storage directories if they don't exist
STORAGE_DIR = Path("./omniscience_data")
DATASETS_DIR = STORAGE_DIR / "datasets"
LEARNING_DIR = STORAGE_DIR / "learning"
FEEDBACK_DIR = STORAGE_DIR / "feedback"

for directory in [STORAGE_DIR, DATASETS_DIR, LEARNING_DIR, FEEDBACK_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

st.title("Omniscience: Elite Sports Betting AI")

# Load your OpenAI API key from Streamlit secrets or environment variable
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found! Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.data_info = None
    st.session_state.upload_status = None
    st.session_state.dataset_id = None
    st.session_state.saved_datasets = []
    st.session_state.messages = []
    st.session_state.feedback = {}

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

# Save chat history and feedback
def save_learning_data():
    if not st.session_state.messages:
        return
    
    # Generate unique conversation ID if needed
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    # Prepare data to save
    learning_data = {
        "id": st.session_state.conversation_id,
        "date": datetime.datetime.now().isoformat(),
        "messages": st.session_state.messages,
        "dataset_id": st.session_state.dataset_id,
        "feedback": st.session_state.feedback
    }
    
    # Save to file
    file_path = LEARNING_DIR / f"{st.session_state.conversation_id}.json"
    with open(file_path, "w") as f:
        json.dump(learning_data, f)

# Clean data function
@st.cache_data
def clean_data(df):
    """Clean data with progress tracking and optimizations"""
    try:
        # Make a copy to avoid modifying the original
        cleaned = df.copy()
        notes = []
        
        # Basic cleaning (fast operations first)
        cleaned.columns = [col.lower().replace(' ', '_') for col in cleaned.columns]
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
        
        # Return with basic info
        return cleaned, "Data cleaned successfully: " + ", ".join(notes)
    except Exception as e:
        return df, f"Basic cleaning only: {str(e)}"

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
                st.rerun()  # Using st.rerun() instead of experimental_rerun
    
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
                st.rerun()  # Using st.rerun() instead of experimental_rerun
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Main content area
col1, col2 = st.columns([2, 1])

# Display data if loaded
with col1:
    if st.session_state.data is not None and st.session_state.upload_status == "complete":
        st.subheader(f"Current Dataset: {st.session_state.data_info.get('name', 'Unnamed')}")
        st.info(f"{st.session_state.data_info['rows']} rows, {st.session_state.data_info['columns']} columns")
        
        with st.expander("View Data Preview"):
            st.dataframe(st.session_state.data.head())
            if "cleaning_msg" in st.session_state.data_info:
                st.info(st.session_state.data_info["cleaning_msg"])

# Chat interface
st.header("Chat with Omniscience")

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        st.write(content)
        
        # Add feedback buttons for assistant messages
        if role == "assistant" and i not in st.session_state.feedback:
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("👍", key=f"like_{i}"):
                    st.session_state.feedback[i] = "positive"
                    save_learning_data()
                    st.rerun()  # Using st.rerun() instead of experimental_rerun
            with col2:
                if st.button("👎", key=f"dislike_{i}"):
                    st.session_state.feedback[i] = "negative"
                    # Show correction input
                    correction = st.text_area("What was wrong? How should it be corrected?", key=f"correction_{i}")
                    if st.button("Submit Correction", key=f"submit_{i}"):
                        st.session_state.feedback[i] = {"rating": "negative", "correction": correction}
                        save_learning_data()
                        st.rerun()  # Using st.rerun() instead of experimental_rerun

# Chat input with explicit send button
st.write("Ask about sports betting, games, or predictions:")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_area("Your message:", key="user_message", height=100)
with col2:
    send_button = st.button("Send", key="send_message")

if send_button and user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate context from data if available
    if st.session_state.data is not None:
        data_info = f"User has uploaded sports data '{st.session_state.data_info.get('name', 'Unnamed')}' with {st.session_state.data_info['rows']} rows."
        # Get a sample of data to provide context
        try:
            sample = st.session_state.data.iloc[0].to_dict()
            data_info += f" Sample data includes: {str(sample)[:200]}..."
        except:
            pass
    else:
        data_info = "No data has been uploaded yet."
    
    # Load previous learning data for context
    learning_context = ""
    if st.session_state.feedback:
        corrections = [f"Previous correction: {fb['correction']}" for i, fb in st.session_state.feedback.items() if isinstance(fb, dict) and "correction" in fb]
        if corrections:
            learning_context = " ".join(corrections)
    
    # Prepare messages for OpenAI
    system_message = f"You are Omniscience, an elite sports betting AI assistant. {data_info}"
    if learning_context:
        system_message += f" IMPORTANT - Learn from these previous corrections: {learning_context}"
    
    messages = [{"role": "system", "content": system_message}]
    messages.extend(st.session_state.messages)
    
    # Call OpenAI Chat API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Save the conversation after each interaction
                save_learning_data()
                
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"I'm sorry, I encountered an error: {e}"})
    
    # Clear the input box after sending
    st.session_state.user_message = ""
    st.rerun()  # Using st.rerun() instead of experimental_rerun

# Add reset button
if st.button("Start New Conversation"):
    # Save current conversation before clearing
    save_learning_data()
    # Reset conversation but keep dataset
    st.session_state.messages = []
    st.session_state.feedback = {}
    st.session_state.conversation_id = str(uuid.uuid4())
    st.rerun()  # Using st.rerun() instead of experimental_rerun
