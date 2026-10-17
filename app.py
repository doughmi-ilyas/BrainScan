from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import MODEL_PATH, predict_image


st.set_page_config(page_title="BrainScan AI", page_icon="🧠")
st.title("BrainScan AI")
st.write("Classification simple d'une image IRM cérébrale par CNN.")
st.warning("Cet outil est un support expérimental et ne remplace pas l'avis d'un professionnel de santé.")

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image sélectionnée", use_container_width=True)

    if not MODEL_PATH.exists():
        st.error("Le modèle n'existe pas encore. Lancez d'abord : python -m src.step2.train_model")
    elif st.button("Analyser l'image"):
        temporary_path = Path("uploaded_image.png")
        image.save(temporary_path)
        predicted_class, confidence = predict_image(temporary_path)
        temporary_path.unlink(missing_ok=True)

        st.success(f"Classe prédite : {predicted_class}")
        st.metric("Confiance du modèle", f"{confidence:.2%}")
        st.info("La confiance est une estimation du modèle, pas une certitude médicale.")
