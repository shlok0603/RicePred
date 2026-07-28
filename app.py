import json
import torch
import pandas as pd
import streamlit as st

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
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)
from pathlib import Path

def load_css():
    css_path = Path("assets/css/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# st.title("Rice Leaf Disease Detection using EfficientNet-B0")

# st.markdown(
# """
# Upload a **Rice Leaf Image** and the model will

# - Detect the disease
# - Show confidence score
# - Explain prediction using Grad-CAM
# - Display disease information
# """
# )

# st.divider()

st.markdown("""
<div class="hero">

<h1>🌾 Rice Leaf Disease Detection</h1>

<p class="subtitle">
AI-Powered Crop Disease Diagnosis using EfficientNet-B0 and Grad-CAM
</p>

<div class="hero-stats">

<div class="stat-card">
<h2>99.33%</h2>
<p>Accuracy</p>
</div>

<div class="stat-card">
<h2>4</h2>
<p>Diseases</p>
</div>

<div class="stat-card">
<h2>AI</h2>
<p>Explainable</p>
</div>

</div>

</div>
""", unsafe_allow_html=True)
st.markdown("### 📤 Upload a rice leaf image below to begin diagnosis")

# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    model = build_model(pretrained=False)

    model = load_checkpoint(
        model,
        MODEL_PATH / "best_model.pth",
        DEVICE
    )

    model.eval()

    return model


model = load_model()


# ============================================================
# Load Knowledge Base
# ============================================================

@st.cache_data
def load_database():

    with open(
        PROJECT_ROOT /
        "knowledge_base" /
        "rice_disease_info.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


disease_db = load_database()


# ============================================================
# Image Transform
# ============================================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )

])


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("# 🌾 Rice AI")

    st.caption("Deep Learning Powered Crop Diagnosis")

    st.divider()

    st.markdown("## 📊 Model")

    st.metric("Accuracy", "99.33%")

    st.metric("Diseases", "4")

    st.metric("Framework", "PyTorch")

    st.divider()

    st.markdown("## 🧠 Architecture")

    st.success("EfficientNet-B0")

    st.info("Grad-CAM Explainability")

    st.divider()

    st.markdown("## 🌱 Supported Diseases")

    st.markdown("""
- 🦠 Bacterial Blight
- 🍂 Blast
- 🟤 Brown Spot
- 🟡 Tungro
""")


# ============================================================
# Upload Image
# ============================================================

st.markdown("## 📤 Upload Image")

st.caption(
    "Supported formats: JPG • JPEG • PNG"
)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]

    confidence = confidence.item()


    # ========================================================
    # Grad-CAM
    # ========================================================

    gradcam = GradCAM(

        model,

        model.blocks[-1][0].bn3

    )

    cam = gradcam.generate(

        input_tensor,

        predicted.item()

    )

    heatmap, overlay = create_overlay(

        image,

        cam

    )


    # ========================================================
    # Display Images
    # ========================================================

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(

            image,

            caption="Original Image",

            use_container_width=True

        )

    with col2:

        st.image(

            heatmap,

            caption="Grad-CAM Heatmap",

            use_container_width=True,

            clamp=True

        )

    with col3:

        st.image(

            overlay,

            caption="Overlay",

            use_container_width=True

        )


    # ========================================================
    # Prediction
    # ========================================================

    st.success(f"Prediction : **{predicted_class}**")

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(

            "Confidence",

            f"{confidence*100:.2f}%"

        )

        st.progress(float(confidence))
        if confidence < 0.70:
            st.warning(
                "Low confidence prediction. "
                "Please upload a clearer image."
            )

    with metric2:

        st.metric(

            "Predicted Disease",

            predicted_class

        )


    # ========================================================
    # Probability Chart
    # ========================================================

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame({

        "Disease": CLASS_NAMES,

        "Probability (%)":
            probabilities.squeeze().cpu().numpy()*100

    })

    st.bar_chart(

        prob_df.set_index("Disease")

    )

    st.divider()
    # ============================================================
    # Disease Information
    # ============================================================

    info = disease_db[predicted_class]

    st.header("Disease Advisory System")

    st.markdown(
        f"""
        ### Detected Disease : **{predicted_class}**
        
        Below is detailed information about the detected disease,
        including its symptoms, causes, treatment, and prevention.
        """
    )

    # ============================================================
    # Description
    # ============================================================

    with st.expander("Description", expanded=True):
        st.write(info["description"])

    # ============================================================
    # Symptoms
    # ============================================================

    with st.expander("Symptoms"):
        for symptom in info["symptoms"]:
            st.markdown(f"- {symptom}")


    # ============================================================
    # Causes
    # ============================================================

    with st.expander("Causes"):
        for cause in info["causes"]:
            st.markdown(f"- {cause}")

    # ============================================================
    # Treatment
    # ============================================================

    with st.expander("Treatment"):
        for treatment in info["treatment"]:
            st.markdown(f"- {treatment}")


    # ============================================================
    # Prevention
    # ============================================================

    with st.expander("Prevention"):
        for prevention in info["prevention"]:
            st.markdown(f"- {prevention}")
    st.divider()


    # ============================================================
    # Summary Card
    # ============================================================
    st.success(
        f"""
        ## Diagnosis Summary

        **Disease:** {predicted_class}

        **Confidence:** {confidence*100:.2f}%

        The uploaded rice leaf is predicted as **{predicted_class}**.

        Grad-CAM highlights the leaf regions that contributed most to this prediction.

        The advisory section below provides recommended management practices.
        """
    )


        # ============================================================
    # Download Prediction Report
    # ============================================================
    
    report = f"""
    ==================================================
    Rice Leaf Disease Detection Report
    ==================================================
    
    Predicted Disease : {predicted_class}
    
    Confidence        : {confidence*100:.2f} %
    
    --------------------------------------------------
    
    Description
    
    {info['description']}
    
    --------------------------------------------------
    
    Symptoms
    
    """
    
    for symptom in info["symptoms"]:
        report += f"- {symptom}\n"
    
    report += "\nCauses\n\n"
    
    for cause in info["causes"]:
        report += f"- {cause}\n"
    
    report += "\nTreatment\n\n"
    
    for treatment in info["treatment"]:
        report += f"- {treatment}\n"
    
    report += "\nPrevention\n\n"
    
    for prevention in info["prevention"]:
        report += f"- {prevention}\n"
    
    report += """
    
    ==================================================
    Generated by
    
    Rice Leaf Disease Detection System
    EfficientNet-B0 + PyTorch + Streamlit
    ==================================================
    """
    
    st.download_button(
    
        label="Download Prediction Report",
    
        data=report,
    
        file_name=f"{predicted_class}_Prediction_Report.txt",
    
        mime="text/plain"
    
    )
    
    
    # ============================================================
    # Sidebar Information
    # ============================================================
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Model Information")
    
    st.sidebar.write(f"**Architecture:** {MODEL_NAME}")
    
    st.sidebar.write(f"**Classes:** {NUM_CLASSES}")
    
    st.sidebar.write(f"**Image Size:** {IMAGE_SIZE} × {IMAGE_SIZE}")
    
    st.sidebar.write(f"**Device:** {DEVICE}")
    
    st.sidebar.write("**Framework:** PyTorch")
    
    st.sidebar.write("**Explainability:** Grad-CAM")
    
    
    # ============================================================
    # Footer
    # ============================================================
    
    st.divider()
    
    st.markdown(
    """
    
    ### Rice Leaf Disease Detection System
    
    """
    )