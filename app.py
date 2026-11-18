import json
import sqlite3
import datetime
import io
import torch
import pandas as pd
import streamlit as st

from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.config import *
from src.model import build_model
from src.utils import load_checkpoint
from src.gradcam import GradCAM, create_overlay


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Krishi Drishti – Rice Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Load CSS
# ============================================================

def load_css():
    css_path = Path("assets/css/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ============================================================
# App State
# ============================================================

if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = None

# ============================================================
# SQLite – Database Setup
# ============================================================

DB_PATH = PROJECT_ROOT / "predictions.db"


def init_db():
    """Create the predictions table if it does not exist."""
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            disease     TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            filename    TEXT
        )
    """)
    con.commit()
    con.close()


def save_prediction(disease: str, confidence: float, filename: str = ""):
    """Insert one prediction record."""
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute(
        "INSERT INTO predictions (timestamp, disease, confidence, filename) VALUES (?,?,?,?)",
        (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            disease,
            round(confidence * 100, 2),
            filename,
        ),
    )
    con.commit()
    con.close()


def fetch_predictions(limit: int = 50):
    """Return the most recent prediction records as a DataFrame."""
    con = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT id, timestamp, disease, confidence, filename "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        con,
        params=(limit,),
    )
    con.close()
    return df


# Initialise the database on startup
init_db()


# ============================================================
# Load Model (cached)
# ============================================================

@st.cache_resource
def load_model():
    model = build_model(pretrained=False)
    model = load_checkpoint(model, MODEL_PATH / "best_model.pth", DEVICE)
    model.eval()
    return model


model = load_model()


# ============================================================
# Load Knowledge Base (cached)
# ============================================================

@st.cache_data
def load_database():
    with open(
        PROJECT_ROOT / "knowledge_base" / "rice_disease_info.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


disease_db = load_database()


# ============================================================
# Image Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🌾</div>
        <div>
            <div class="sidebar-title">Krishi Drishti</div>
            <div class="sidebar-subtitle">Smart Crop Health</div>
            <div class="sidebar-subtitle-hindi">स्मार्ट फसल स्वास्थ्य</div>
        </div>
    </div>
    """)

    st.divider()

    st.markdown(
        '<div class="sidebar-section-title">Navigation / नेविगेशन</div>',
        unsafe_allow_html=True,
    )

    if st.button("🏠  Home / होम", use_container_width=True, key="nav_home"):
        st.session_state.selected_crop = None
        st.rerun()

    if st.button(
        "🌱  Crop Diagnosis / फसल जांच",
        use_container_width=True,
        key="nav_diagnosis",
    ):
        st.session_state.selected_crop = "Rice"
        st.rerun()

    st.button(
        "📖  Disease Guide / रोग गाइड",
        use_container_width=True,
        key="nav_disease",
    )

    if st.button(
        "📊  Prediction History / इतिहास",
        use_container_width=True,
        key="nav_history",
    ):
        st.session_state.selected_crop = "History"
        st.rerun()

    st.button(
        "ℹ️  About / हमारे बारे में",
        use_container_width=True,
        key="nav_about",
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-section-title">Language / भाषा</div>',
        unsafe_allow_html=True,
    )

    st.radio(
        "Language",
        ["English + हिंदी"],
        label_visibility="collapsed",
        key="language",
    )

    st.divider()

    st.html("""
    <div class="sidebar-status">
        <div class="status-label">CURRENTLY AVAILABLE</div>
        <div class="status-crop">🌾 Rice / चावल</div>
        <div class="status-text">Disease detection available</div>
        <div class="status-text-hindi">बीमारी की पहचान उपलब्ध है</div>
    </div>
    """)

    # Model info (always visible in sidebar)
    st.divider()
    st.markdown('<div class="sidebar-section-title">Model Info</div>', unsafe_allow_html=True)
    st.write(f"**Architecture:** {MODEL_NAME}")
    st.write(f"**Classes:** {NUM_CLASSES}")
    st.write(f"**Image Size:** {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write(f"**Device:** {DEVICE}")
    st.write("**Framework:** PyTorch + timm")
    st.write("**Explainability:** Grad-CAM")


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.selected_crop is None:

    st.html("""
    <div class="krishi-hero">
        <div class="krishi-logo">🌾</div>
        <h1>Krishi Drishti</h1>
        <p class="krishi-tagline">
            Smart Crop Health Assistant
            <br>
            <span>स्मार्ट फसल स्वास्थ्य सहायक</span>
        </p>
        <p class="krishi-hero-desc">
            AI-powered rice leaf disease detection using EfficientNet-B0.
            Upload a photo of your rice leaf and get an instant diagnosis.
        </p>
    </div>
    """)

    st.html("""
    <div class="crop-heading">
        <h2>Select your crop / अपनी फसल चुनें</h2>
    </div>
    """)

    crop1, crop2, crop3, crop4 = st.columns(4)

    with crop1:
        st.markdown("""
        <div class="crop-card available">
            <div class="crop-icon">🌾</div>
            <h3>Rice</h3>
            <p>चावल</p>
            <span class="available-badge">✓ Available</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 Start Diagnosis", key="rice_btn", use_container_width=True):
            st.session_state.selected_crop = "Rice"
            st.rerun()

    with crop2:
        st.markdown("""
        <div class="crop-card">
            <div class="crop-icon">🍅</div>
            <h3>Tomato</h3>
            <p>टमाटर</p>
            <span class="coming-badge">Coming Soon</span>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="tomato_btn", disabled=True, use_container_width=True)

    with crop3:
        st.markdown("""
        <div class="crop-card">
            <div class="crop-icon">🍎</div>
            <h3>Apple</h3>
            <p>सेब</p>
            <span class="coming-badge">Coming Soon</span>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="apple_btn", disabled=True, use_container_width=True)

    with crop4:
        st.markdown("""
        <div class="crop-card">
            <div class="crop-icon">🎋</div>
            <h3>Sugarcane</h3>
            <p>गन्ना</p>
            <span class="coming-badge">Coming Soon</span>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="sugarcane_btn", disabled=True, use_container_width=True)

    # Feature overview
    st.markdown("---")
    st.html("""
    <div class="features-row">
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Detection</div>
            <div class="feature-desc">EfficientNet-B0 model trained on rice leaf images</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Grad-CAM</div>
            <div class="feature-desc">Visual explanation showing which leaf areas triggered the prediction</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📖</div>
            <div class="feature-title">Advisory</div>
            <div class="feature-desc">Disease description, symptoms, treatment and prevention</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">History</div>
            <div class="feature-desc">SQLite-backed prediction history and analytics</div>
        </div>
    </div>
    """)


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================

elif st.session_state.selected_crop == "History":

    st.html("""
    <div class="diagnosis-header">
        <div class="diagnosis-icon">📊</div>
        <div>
            <h1>Prediction History</h1>
            <p>पिछली भविष्यवाणियों का इतिहास</p>
        </div>
    </div>
    """)

    history_df = fetch_predictions(limit=100)

    if history_df.empty:
        st.info("No predictions saved yet. Upload a rice leaf image to get started! / अभी तक कोई भविष्यवाणी सहेजी नहीं गई।")
    else:
        # Analytics summary
        total = len(history_df)
        most_common = history_df["disease"].mode()[0]
        avg_conf = history_df["confidence"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Diagnoses / कुल जांच", total)
        c2.metric("Most Common Disease", most_common)
        c3.metric("Average Confidence / औसत भरोसा", f"{avg_conf:.1f}%")

        st.markdown("---")

        # Disease distribution bar chart
        st.subheader("Disease Distribution / बीमारी वितरण")
        disease_counts = history_df["disease"].value_counts().reset_index()
        disease_counts.columns = ["Disease", "Count"]
        st.bar_chart(disease_counts.set_index("Disease"))

        st.markdown("---")
        st.subheader("Recent Predictions / हाल की भविष्यवाणियाँ")
        st.dataframe(
            history_df.rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "disease": "Disease",
                "confidence": "Confidence (%)",
                "filename": "File",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RICE DIAGNOSIS PAGE
# ============================================================

elif st.session_state.selected_crop == "Rice":

    # ----------------------------------------------------------
    # Page Header
    # ----------------------------------------------------------

    st.html("""
    <div class="diagnosis-header">
        <div class="diagnosis-icon">🌾</div>
        <div>
            <h1>Rice Disease Diagnosis</h1>
            <p>चावल की बीमारी की पहचान</p>
            <span>
                Upload a clear photo of your rice leaf
                <br>
                चावल की पत्ती की साफ फोटो अपलोड करें
            </span>
        </div>
    </div>
    """)

    # ----------------------------------------------------------
    # Upload Section
    # ----------------------------------------------------------

    st.markdown("""
    <div class="upload-intro">
        <h2>📷 Upload Leaf Image / पत्ती की फोटो अपलोड करें</h2>
        <p>
            Take a clear, well-lit photo of the affected rice leaf.<br>
            प्रभावित चावल की पत्ती की साफ और अच्छी रोशनी वाली फोटो लें।
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your rice leaf image / चावल की पत्ती की फोटो",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    # Supported formats hint
    st.markdown(
        '<p class="upload-format-hint">✅ Supported formats: JPG, JPEG, PNG</p>',
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------
    # Prediction – only after upload
    # ----------------------------------------------------------

    if uploaded_file is not None:

        # Validate the image is readable
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error(
                "❌ The uploaded file could not be read as an image. "
                "Please upload a valid JPG or PNG file.\n\n"
                "अपलोड की गई फ़ाइल को इमेज के रूप में नहीं पढ़ा जा सका। "
                "कृपया एक वैध JPG या PNG फ़ाइल अपलोड करें।"
            )
            st.stop()

        # ---- Inference ----
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence_tensor, predicted_tensor = torch.max(probabilities, 1)

        predicted_class = CLASS_NAMES[predicted_tensor.item()]
        confidence = confidence_tensor.item()

        # ---- Grad-CAM ----
        gradcam = GradCAM(model, model.blocks[-1][0].bn3)
        cam = gradcam.generate(input_tensor, predicted_tensor.item())
        heatmap, overlay = create_overlay(image, cam)

        # ---- Disease Info from knowledge base ----
        info = disease_db[predicted_class]

        # ---- Save to SQLite ----
        save_prediction(
            disease=predicted_class,
            confidence=confidence,
            filename=uploaded_file.name,
        )

        # ==================================================
        # Prediction Result Header
        # ==================================================

        st.html("""
        <div class="result-header">
            <div>
                <div class="result-eyebrow">AI DIAGNOSIS / AI जांच</div>
                <h2>Prediction Result / भविष्यवाणी का परिणाम</h2>
                <p>आपकी फसल की पत्ती का AI द्वारा विश्लेषण किया गया है</p>
            </div>
        </div>
        """)

        # ==================================================
        # Main Result Card
        # ==================================================

        st.html(f"""
        <div class="diagnosis-result-card">
            <div class="result-status">
                <span class="status-dot"></span>
                Disease Detected / बीमारी की पहचान हुई
            </div>
            <div class="result-disease">
                <div>
                    <div class="result-label">Predicted Disease / संभावित बीमारी</div>
                    <div class="result-disease-name">{predicted_class}</div>
                </div>
                <div class="confidence-box">
                    <div class="result-label">Confidence / भरोसे का स्तर</div>
                    <div class="confidence-value">{confidence * 100:.2f}%</div>
                </div>
            </div>
        </div>
        """)

        # Confidence progress bar
        st.progress(
            float(confidence),
            text=f"AI Confidence / AI भरोसा: {confidence * 100:.2f}%",
        )

        if confidence < 0.70:
            st.warning(
                "⚠️ Low confidence prediction. Please upload a clearer, well-lit image of the affected leaf. "
                "कम भरोसा: कृपया पत्ती की एक साफ और अच्छी रोशनी वाली फोटो अपलोड करें।"
            )

        # ==================================================
        # AI Explanation – Grad-CAM
        # ==================================================

        st.html("""
        <div class="section-heading">
            <div class="section-icon">🔍</div>
            <div>
                <h2>AI Explanation / AI विश्लेषण</h2>
                <p>AI ने पत्ती के किन हिस्सों पर ध्यान दिया — Grad-CAM द्वारा</p>
            </div>
        </div>
        """)

        img1, img2, img3 = st.columns(3)

        with img1:
            st.html("""
            <div class="image-card-title">
                Original Image
                <span>मूल फोटो</span>
            </div>
            """)
            st.image(image, use_container_width=True)

        with img2:
            st.html("""
            <div class="image-card-title">
                Grad-CAM Heatmap
                <span>AI ध्यान क्षेत्र — गर्म रंग = अधिक महत्व</span>
            </div>
            """)
            st.image(heatmap, use_container_width=True, clamp=True)

        with img3:
            st.html("""
            <div class="image-card-title">
                AI Focus Overlay
                <span>AI द्वारा पहचाना गया क्षेत्र (मूल + हीटमैप)</span>
            </div>
            """)
            st.image(overlay, use_container_width=True)

        # ==================================================
        # Prediction Analysis – Probability Chart
        # ==================================================

        st.html("""
        <div class="section-heading probability-heading">
            <div class="section-icon">📊</div>
            <div>
                <h2>Prediction Analysis / पूर्वानुमान विश्लेषण</h2>
                <p>सभी संभावित बीमारियों की संभावना</p>
            </div>
        </div>
        """)

        prob_values = probabilities.squeeze().cpu().numpy() * 100
        prob_df = pd.DataFrame({
            "Disease": CLASS_NAMES,
            "Probability (%)": prob_values,
        })
        st.bar_chart(prob_df.set_index("Disease"))

        # ==================================================
        # Disease Advisory
        # ==================================================

        st.html("""
        <div class="section-heading advisory-heading">
            <div class="section-icon">📖</div>
            <div>
                <h2>Disease Guide / बीमारी की जानकारी</h2>
                <p>बीमारी के बारे में महत्वपूर्ण जानकारी</p>
            </div>
        </div>
        """)

        st.html(f"""
        <div class="disease-title-card">
            <div class="disease-title">{predicted_class}</div>
            <div class="disease-title-hindi">Detected Disease / पहचानी गई बीमारी</div>
        </div>
        """)

        with st.expander("📋 Description / विवरण", expanded=True):
            st.write(info["description"])

        with st.expander("🔎 Symptoms / लक्षण"):
            for symptom in info["symptoms"]:
                st.markdown(f"- {symptom}")

        with st.expander("🦠 Causes / कारण"):
            for cause in info["causes"]:
                st.markdown(f"- {cause}")

        with st.expander("💊 Treatment / उपचार"):
            for treatment in info["treatment"]:
                st.markdown(f"- {treatment}")

        with st.expander("🛡️ Prevention / बचाव"):
            for prevention in info["prevention"]:
                st.markdown(f"- {prevention}")

        # ==================================================
        # Final Diagnosis Summary
        # ==================================================

        st.html(f"""
        <div class="final-summary">
            <div class="summary-icon">🌾</div>
            <div>
                <h3>Diagnosis Summary / जांच का सारांश</h3>
                <p>
                    The AI model predicts
                    <strong>{predicted_class}</strong>
                    with
                    <strong>{confidence * 100:.2f}%</strong>
                    confidence.
                </p>
                <p class="summary-hindi">
                    AI मॉडल के अनुसार आपकी चावल की पत्ती में
                    <strong>{predicted_class}</strong>
                    की संभावना है।
                    भरोसे का स्तर: <strong>{confidence * 100:.2f}%</strong>
                </p>
            </div>
        </div>
        """)

        # ==================================================
        # Downloadable Report
        # ==================================================

        report_lines = [
            "==================================================",
            "Rice Leaf Disease Detection Report",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "==================================================",
            "",
            f"Predicted Disease : {predicted_class}",
            f"Confidence        : {confidence * 100:.2f}%",
            "",
            "--------------------------------------------------",
            "Description",
            "--------------------------------------------------",
            info["description"],
            "",
            "--------------------------------------------------",
            "Symptoms",
            "--------------------------------------------------",
        ]
        for s in info["symptoms"]:
            report_lines.append(f"  - {s}")
        report_lines += ["", "--------------------------------------------------", "Causes", "--------------------------------------------------"]
        for c in info["causes"]:
            report_lines.append(f"  - {c}")
        report_lines += ["", "--------------------------------------------------", "Treatment", "--------------------------------------------------"]
        for t in info["treatment"]:
            report_lines.append(f"  - {t}")
        report_lines += ["", "--------------------------------------------------", "Prevention", "--------------------------------------------------"]
        for p in info["prevention"]:
            report_lines.append(f"  - {p}")
        report_lines += [
            "",
            "==================================================",
            "Generated by Krishi Drishti",
            "EfficientNet-B0 + PyTorch + Streamlit",
            "==================================================",
        ]
        report = "\n".join(report_lines)

        st.download_button(
            label="📥 Download Prediction Report / रिपोर्ट डाउनलोड करें",
            data=report,
            file_name=f"{predicted_class}_Prediction_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown(
            '<p style="text-align:center;color:#6b776b;font-size:14px;">'
            "Krishi Drishti · EfficientNet-B0 · PyTorch · Streamlit"
            "</p>",
            unsafe_allow_html=True,
        )

    else:
        # No image uploaded yet — show a friendly placeholder
        st.html("""
        <div class="upload-placeholder">
            <div class="upload-camera">📷</div>
            <div class="upload-main-text">
                <strong>Upload a rice leaf image to begin</strong><br>
                <span>चावल की पत्ती की फोटो अपलोड करें और AI जांच शुरू करें</span>
            </div>
            <div class="upload-format">Supported: JPG · JPEG · PNG</div>
        </div>
        """)