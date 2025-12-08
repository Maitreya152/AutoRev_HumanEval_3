import streamlit as st
import pandas as pd
import json
import os
import random
import base64
import re
from pathlib import Path
from datetime import datetime

# --- Configuration & Constants ---
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_CSV_PATH = BASE_DIR / "results.csv"

# Source A: COLM
COLM_JSON_DIR = BASE_DIR / "data_colm"
COLM_PDF_DIR = BASE_DIR / "pdfs_colm"

# Source B: NeurIPS
NEURIPS_JSON_DIR = BASE_DIR / "data_neurips"
NEURIPS_PDF_DIR = BASE_DIR / "pdfs_neurips"

# --- Page Configuration ---
st.set_page_config(
    page_title="Human Evaluation for Peer Reviews",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for PDF layout & UI ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    iframe {width: 100%; height: 80vh;}
    /* Highlight the download button area slightly */
    .stDownloadButton {text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def display_pdf(file_path):
    """
    Provides a Download button and attempts to embed the PDF.
    """
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
            
        file_name = os.path.basename(file_path)

        # --- Option 1: Download Button (Robust for Large Files) ---
        st.download_button(
            label=f"📥 Download {file_name}",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True  # Makes the button span the column width
        )
        
        # --- Option 2: Embed (Visual Preview) ---
        # Note: Very large base64 strings can still freeze some browsers. 
        # You could add a size check here (e.g., if len(pdf_bytes) < 5MB) 
        # but the download button above serves as the fallback.
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"PDF file not found at: {file_path}")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")

def parse_review(review_text):
    """
    Parses a review string into a dictionary of sections and bullet points.
    """
    if not isinstance(review_text, str):
        return {
            "Summary": ["Review not available."], "Strengths": [], "Weaknesses": [], "Questions": []
        }

    sections = { "Summary": [], "Strengths": [], "Weaknesses": [], "Questions": [] }
    
    # Regex parsing
    summary_match = re.search(r'\*\*Summary\*\*(.*?)(?=\*\*Strengths\*\*|\Z)', review_text, re.DOTALL)
    strengths_match = re.search(r'\*\*Strengths\*\*(.*?)(?=\*\*Weaknesses\*\*|\Z)', review_text, re.DOTALL)
    weaknesses_match = re.search(r'\*\*Weaknesses\*\*(.*?)(?=\*\*Questions\*\*|\Z)', review_text, re.DOTALL)
    questions_match = re.search(r'\*\*Questions\*\*(.*)', review_text, re.DOTALL)

    if summary_match and summary_match.group(1).strip():
        sections["Summary"].append(summary_match.group(1).strip())

    for match, section_name in [(strengths_match, "Strengths"), (weaknesses_match, "Weaknesses"), (questions_match, "Questions")]:
        if match:
            content = match.group(1).strip()
            raw_points = content.split('\n-')
            cleaned_points = []
            for point in raw_points:
                cleaned = point.strip().lstrip('-').strip()
                if cleaned:
                    cleaned_points.append(cleaned)
            sections[section_name] = cleaned_points
            
    return sections

@st.cache_data
def load_global_data():
    """Loads metadata and JSONs."""
    data_store = {
        "users": pd.DataFrame(),
        "mapping": pd.DataFrame(),
        "colm": {"5_3": {}, "5_5": {}},
        "neurips": {"5_3": {}, "5_5": {}}
    }
    
    try:
        # 1. Metadata
        if (DATA_DIR / "user.csv").exists():
            data_store["users"] = pd.read_csv(DATA_DIR / "user.csv")
            data_store["users"].columns = data_store["users"].columns.str.strip()
            
        if (DATA_DIR / "mapping.csv").exists():
            data_store["mapping"] = pd.read_csv(DATA_DIR / "mapping.csv")
            data_store["mapping"].columns = data_store["mapping"].columns.str.strip()

        # 2. COLM
        p_c_53 = COLM_JSON_DIR / "inference_new_papers_5_3.json"
        p_c_55 = COLM_JSON_DIR / "inference_new_papers_5_5.json"
        if p_c_53.exists():
            with open(p_c_53, 'r', encoding='utf-8') as f: data_store["colm"]["5_3"] = json.load(f)
        if p_c_55.exists():
            with open(p_c_55, 'r', encoding='utf-8') as f: data_store["colm"]["5_5"] = json.load(f)

        # 3. NeurIPS
        p_n_53 = NEURIPS_JSON_DIR / "inference_new_papers_5_3.json"
        p_n_55 = NEURIPS_JSON_DIR / "inference_new_papers_5_5.json"
        if p_n_53.exists():
            with open(p_n_53, 'r', encoding='utf-8') as f: data_store["neurips"]["5_3"] = json.load(f)
        if p_n_55.exists():
            with open(p_n_55, 'r', encoding='utf-8') as f: data_store["neurips"]["5_5"] = json.load(f)
                
    except Exception as e:
        st.error(f"Critical Error loading data: {e}")
        
    return data_store

def get_paper_details(paper_id, data_store):
    """Locates paper in COLM or NeurIPS."""
    # Check COLM
    if paper_id in data_store["colm"]["5_3"] or paper_id in data_store["colm"]["5_5"]:
        return COLM_PDF_DIR / f"{paper_id}.pdf", data_store["colm"]["5_3"].get(paper_id), data_store["colm"]["5_5"].get(paper_id), "COLM"

    # Check NeurIPS
    if paper_id in data_store["neurips"]["5_3"] or paper_id in data_store["neurips"]["5_5"]:
        return NEURIPS_PDF_DIR / f"{paper_id}.pdf", data_store["neurips"]["5_3"].get(paper_id), data_store["neurips"]["5_5"].get(paper_id), "NeurIPS"
        
    return None, None, None, None

def save_results(new_records):
    """Saves to CSV."""
    df = pd.DataFrame(new_records)
    if not os.path.exists(RESULTS_CSV_PATH):
        df.to_csv(RESULTS_CSV_PATH, index=False)
    else:
        df.to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)

# --- Grading Constants ---
GRADING_OPTIONS = [
    "Select...",
    "Completely Disagree",
    "Mostly Disagree",
    "Mostly Agree",
    "Completely Agree"
]

# --- Main App Logic ---
st.sidebar.title("Review Setup")
global_data = load_global_data()
user_df = global_data["users"]
mapping_df = global_data["mapping"]

selected_paper_id = None
current_user_id = None

if not user_df.empty:
    user_names = user_df['Name'].tolist()
    selected_name = st.sidebar.selectbox("Select User", ["Select..."] + user_names)

    if selected_name != "Select...":
        current_user_id = user_df[user_df['Name'] == selected_name]['User'].iloc[0]
        user_mapping = mapping_df[mapping_df['user'] == current_user_id]
        
        if not user_mapping.empty:
            p1 = user_mapping.iloc[0]['paper_1']
            p2 = user_mapping.iloc[0]['paper_2']
            st.sidebar.markdown("---")
            st.sidebar.subheader("Assignments")
            selected_paper_id = st.sidebar.radio("Choose Paper", [p1, p2])
        else:
            st.sidebar.error("No papers assigned.")

if current_user_id and selected_paper_id:
    pdf_path, review_53, review_55, source = get_paper_details(selected_paper_id, global_data)
    
    if not source:
        st.error(f"Paper ID '{selected_paper_id}' not found.")
    else:
        st.title(f"Evaluating: {selected_paper_id} ({source})")
        
        # --- 1. Top Section: PDF Viewer ---
        st.subheader("Original Paper")
        if pdf_path.exists():
            display_pdf(pdf_path)
        else:
            st.warning(f"PDF missing at {pdf_path}")
        
        st.markdown("---")

        # --- 2. Bottom Section: Review Evaluation ---
        st.subheader("AutoRev Generated Review")
        
        reviews_map = {'5_3': review_53, '5_5': review_55}
        
        if not review_53 and not review_55:
            st.error("JSON entries empty.")
        else:
            # Blind A/B
            session_key = f"order_{selected_paper_id}"
            if session_key not in st.session_state:
                opts = ['5_3', '5_5']
                random.shuffle(opts)
                st.session_state[session_key] = opts
            
            review_order = st.session_state[session_key]
            
            tab_a, tab_b = st.tabs(["Review Set A", "Review Set B"])
            
            for idx, (tab, r_type) in enumerate(zip([tab_a, tab_b], review_order)):
                with tab:
                    raw_data = reviews_map.get(r_type)
                    
                    # Extract string
                    review_text = ""
                    if isinstance(raw_data, dict):
                        review_text = raw_data.get("inference_review", "") or raw_data.get("prediction", "")
                    elif isinstance(raw_data, str):
                        review_text = raw_data
                    
                    parsed_data = parse_review(review_text)

                    # --- Evaluation Form ---
                    with st.container(border=True):
                        # 1. Summary
                        st.markdown("##### Summary")
                        s_list = parsed_data.get("Summary", [])
                        s_text = s_list[0] if s_list else "No summary found."
                        st.markdown(s_text)
                        
                        s_key = f"{selected_paper_id}_{r_type}_Summary"
                        st.selectbox(
                            "Is this summary accurate?",
                            options=GRADING_OPTIONS,
                            key=s_key
                        )
                        st.divider()

                        # 2. Points
                        for section in ["Strengths", "Weaknesses", "Questions"]:
                            points = parsed_data.get(section, [])
                            if points:
                                st.markdown(f"##### {section}")
                                for i, pt in enumerate(points):
                                    # Create columns: Left (Text), Right (Dropdown)
                                    c_text, c_input = st.columns([3, 1])
                                    
                                    with c_text:
                                        st.markdown(f"- {pt}")
                                    
                                    with c_input:
                                        p_key = f"{selected_paper_id}_{r_type}_{section}_{i}"
                                        st.selectbox(
                                            "Rate Point",
                                            options=GRADING_OPTIONS,
                                            key=p_key,
                                            label_visibility="collapsed"
                                        )
                                
                                if section != "Questions":
                                    st.divider()

            st.markdown("---")
            if st.button("Submit Evaluation", type="primary"):
                records = []
                valid = True
                
                for r_type in ['5_3', '5_5']:
                    # Re-parse to get text
                    raw_sub = reviews_map.get(r_type)
                    txt_sub = raw_sub.get("inference_review", "") if isinstance(raw_sub, dict) else raw_sub
                    parsed_sub = parse_review(txt_sub)

                    # Check Summary
                    s_val = st.session_state.get(f"{selected_paper_id}_{r_type}_Summary")
                    if not s_val or s_val == "Select...":
                        valid = False
                        break
                    
                    records.append({
                        "timestamp": datetime.now().isoformat(),
                        "user": current_user_id,
                        "paper_id": selected_paper_id,
                        "review_type": r_type,
                        "section": "Summary",
                        "point_index": 0,
                        "point_text": parsed_sub.get("Summary", [""])[0],
                        "rating": s_val
                    })
                    
                    for sec in ["Strengths", "Weaknesses", "Questions"]:
                        for i, txt in enumerate(parsed_sub.get(sec, [])):
                            p_key = f"{selected_paper_id}_{r_type}_{sec}_{i}"
                            p_val = st.session_state.get(p_key)
                            
                            if not p_val or p_val == "Select...":
                                valid = False
                                break

                            records.append({
                                "timestamp": datetime.now().isoformat(),
                                "user": current_user_id,
                                "paper_id": selected_paper_id,
                                "review_type": r_type,
                                "section": sec,
                                "point_index": i,
                                "point_text": txt,
                                "rating": p_val
                            })
                    if not valid: break
                
                if valid:
                    save_results(records)
                    st.success("Saved successfully!")
                else:
                    st.error("Please ensure all fields (Summary and Points) are rated for both Review sets.")
else:
    st.info("👈 Select a user from the sidebar to start.")