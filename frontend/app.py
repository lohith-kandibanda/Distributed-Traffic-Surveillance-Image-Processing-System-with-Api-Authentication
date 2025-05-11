import streamlit as st
import requests
import time
import json
from PIL import Image
from io import BytesIO
from pathlib import Path
import os

API_URL = "http://api_server:8000"
LOCAL_ANNOTATED_DIR = "./static/annotated"

st.set_page_config(page_title="🚦 Distributed Traffic System", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #1f6feb;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.5em;
        }
        .stDownloadButton > button {
            background-color: #238636;
            color: white;
            border-radius: 8px;
        }
        .stFileUploader {
            background: #161b22;
            padding: 1em;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='color:#58a6ff;'>🚦 Distributed Traffic Processing System</h1>
<p>Upload a traffic surveillance <b>video or image</b>, and the system will:</p>
<ul>
  <li>🔍 Detect vehicles and their types</li>
  <li>🔤 Extract license plates</li>
  <li>🪖 Detect helmet violations and associate them with vehicle plates</li>
  <li>📎 Return an annotated result with download support</li>
</ul>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a Traffic Video/Image", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])

if uploaded_file and st.button("🚀 Submit File"):
    st.info("⏳ Uploading file... Please wait.")
    file_ext = uploaded_file.name.split(".")[-1].lower()
    content_type = "video/mp4" if file_ext in ["mp4", "avi", "mov", "mkv"] else "image/jpeg"
    files = {"file": (uploaded_file.name, uploaded_file, content_type)}
    headers = {"X-API-Key": "traffic123"}

    try:
        response = requests.post(f"{API_URL}/upload/", files=files, headers=headers)
        response.raise_for_status()
        task_id = response.json()["task_id"]
        st.success(f"✅ Task ID `{task_id}` submitted.")

        retries = 0
        max_retries = 120
        result = None

        with st.spinner("🔄 Processing..."):
            while retries < max_retries:
                res = requests.get(f"{API_URL}/result/{task_id}")
                res_data = res.json()
                if res_data["status"] == "done":
                    result = json.loads(res_data["result"]) if isinstance(res_data["result"], str) else res_data["result"]
                    st.session_state.result = result
                    break
                elif res_data["status"] == "not_found":
                    st.error("❌ Task Not Found.")
                    break
                time.sleep(5)
                retries += 1

        if not result:
            st.error("⏱️ Timeout: Task took too long. Please try again.")

    except Exception as e:
        st.error(f"🚫 Error: {e}")

if "result" in st.session_state:
    result = st.session_state.result
    st.markdown("""<h2 style='color:#8b949e;'>📊 Summary:</h2>""", unsafe_allow_html=True)
    st.write(f"**Total Frames Processed**: {result.get('total_frames_processed', 0)}")
    total_unique_vehicles = sum(result.get("vehicle_types", {}).values())
    st.write(f"**Total Unique Vehicles Detected**: {total_unique_vehicles}")
    st.write(f"**Unique License Plates Found**: {len(result.get('license_plates', []))}")
    st.write(f"**Helmet Violations**: {len(result.get('helmet_violations', []))}")

    st.markdown("""<h3 style='color:#d2a8ff;'>🚗 Vehicle Types:</h3>""", unsafe_allow_html=True)
    for vtype, count in result.get("vehicle_types", {}).items():
        st.markdown(f"<span style='color:#58a6ff'>• {vtype}:</span> {count}", unsafe_allow_html=True)

    st.markdown("""<h3 style='color:#ff7b72;'>🔴 Helmet Violations:</h3>""", unsafe_allow_html=True)
    helmet_violations = result.get("helmet_violations", [])
    if "show_helmet" not in st.session_state:
        st.session_state.show_helmet = False
    st.session_state.show_helmet = st.checkbox("👷 Show Only Helmet Violations", value=st.session_state.show_helmet)
    if st.session_state.show_helmet:
        if helmet_violations:
            for v in helmet_violations:
                st.write(f"- ❌ No Helmet | Plate: {v.get('plate')} | Box: {v.get('bbox')}")
        else:
            st.success("✅ No Helmet Violations Detected")
    else:
        st.write(f"Total: {len(helmet_violations)} (Toggle above to filter list)")

    st.markdown("""<h3 style='color:#79c0ff;'>📎 Annotated Output:</h3>""", unsafe_allow_html=True)
    annotated_url = result.get("annotated_url", "")
    file_name = Path(annotated_url).name
    local_file_path = os.path.join(LOCAL_ANNOTATED_DIR, file_name)

    if result["type"] == "image":
        try:
            image_res = requests.get(annotated_url)
            image = Image.open(BytesIO(image_res.content))
            st.image(image, caption="🖼️ Annotated Image", use_container_width=True)
        except Exception:
            st.warning("⚠️ Unable to load image preview.")
    else:
        st.video(annotated_url)

    if os.path.exists(local_file_path):
        with open(local_file_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Annotated Output",
                data=f,
                file_name=file_name,
                mime="video/mp4" if file_name.endswith(".mp4") else "image/jpeg"
            )
    else:
        st.warning("⚠️ Annotated file not found on server.")
