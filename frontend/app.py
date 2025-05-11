import streamlit as st
import requests
import time
import json
from PIL import Image

API_URL = "http://api_server:8000"

st.set_page_config(page_title="Distributed Traffic System", layout="centered")
st.title("🚦 Distributed Traffic Processing System")

st.markdown("""
Upload a traffic surveillance **video or image**, and the system will:
- Detect vehicles and their types
- Extract license plates
- Detect helmet violations and associate with vehicle plates
""")

uploaded_file = st.file_uploader("Upload a Traffic Video/Image (.mp4/.jpg/.png)", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Submit File"):
        st.success("Uploading file... Please wait.")

        file_ext = uploaded_file.name.split(".")[-1].lower()
        content_type = "video/mp4" if file_ext in ["mp4", "avi", "mov", "mkv"] else "image/jpeg"

        files = {"file": (uploaded_file.name, uploaded_file, content_type)}
        headers = {"X-API-Key": "traffic123"}

        try:
            response = requests.post(f"{API_URL}/upload/", files=files, headers=headers)
            response.raise_for_status()
            task_id = response.json()["task_id"]

            st.info(f"File uploaded! Task ID: {task_id}")
            st.info("Processing... This may take a minute ⏳")

            while True:
                res = requests.get(f"{API_URL}/result/{task_id}")
                res_data = res.json()

                if res_data["status"] == "done":
                    st.success("✅ Processing Complete!")
                    try:
                        result = json.loads(res_data["result"]) if isinstance(res_data["result"], str) else res_data["result"]
                    except Exception as e:
                        st.error(f"❌ Error parsing result JSON: {e}")
                        break
                    break
                elif res_data["status"] == "not_found":
                    st.error("❌ Task Not Found.")
                    break
                else:
                    st.info("⏳ Still Processing...")
                    time.sleep(5)

            if not result:
                st.warning("⚠️ No result data found.")
                st.stop()

            st.subheader("📊 Summary:")
            st.write(f"**Total Frames Processed**: {result.get('total_frames_processed', 0)}")
            st.write(f"**Total Vehicles Detected**: {result.get('vehicle_count', 0)}")

            license_plates = result.get("license_plates", [])
            unique_plates = set(p.get("plate_text", "") for p in license_plates if isinstance(p, dict))
            st.write(f"**Unique License Plates Found**: {len(unique_plates)}")

            helmet_violations = result.get("helmet_violations", [])
            st.write(f"**Helmet Violations**: {len(helmet_violations)}")

            st.subheader("🚗 Vehicle Types:")
            for v in result.get("vehicles", []):
                if isinstance(v, dict):
                    st.write(f"- {v.get('type', 'Unknown')} at Box {v.get('bbox', [])}")

            st.subheader("🔵 License Plates:")
            for plate in license_plates:
                if isinstance(plate, dict):
                    st.write(f"- {plate.get('plate_text', 'Unknown')} (Conf: {plate.get('confidence', '?')})")

            st.subheader("🔴 Helmet Violations with Plate:")
            if helmet_violations:
                for item in helmet_violations:
                    if isinstance(item, dict):
                        plate_text = item.get("plate_text", "Unknown")
                        bbox = item.get("bbox", [])
                        st.write(f"- ❌ No Helmet | Plate: {plate_text} | Box: {bbox}")
                    else:
                        st.write(f"- ❌ No Helmet | {item}")
            else:
                st.info("✅ No Helmet Violations Detected!")

            if "annotated_video_url" in result:
                st.subheader("📹 Annotated Result Video:")
                st.video(result["annotated_video_url"])

        except Exception as e:
            st.error(f"🚫 Error: {e}")
