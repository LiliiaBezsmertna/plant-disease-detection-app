import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Sidinställningar
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Laddar modellen
model = tf.keras.models.load_model("plant_disease_model.keras")

# Rubrik
st.title("🌿 Plant Disease Detection App")
st.markdown(
    "Ladda upp en bild av ett växtblad för att kontrollera om det är **friskt** eller **sjukt**."
)

st.info(
    "För bästa resultat: fotografera ett enskilt blad i närbild med enkel bakgrund och bra ljus."
)

# Sidebar
with st.sidebar:
    st.header("Om modellen")
    st.write("**Modell:** MobileNetV2 med transfer learning")
    st.write("**Bildstorlek:** 224 × 224")
    st.write("**Klasser:** Healthy / Disease")
    st.write("**Valideringsnoggrannhet:** ca 97–98 %")

    st.header("Affärsvärde")
    st.write(
        "Bildanalys kan hjälpa jordbruket att upptäcka växtsjukdomar tidigt, "
        "minska manuellt arbete och förbättra kvalitet och skörd."
    )

    st.header("Möjliga förbättringar")
    st.write("- fler sjukdomsklasser")
    st.write("- fler träningsbilder")
    st.write("- fine-tuning av basmodellen")
    st.write("- kamera/mobilintegration")

# Val av bildkälla
input_method = st.radio(
    "Välj bildkälla:",
    ["Ladda upp bild", "Ta foto med kamera"]
)

image = None

if input_method == "Ladda upp bild":
    uploaded_file = st.file_uploader(
        "Välj en bild...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_photo = st.camera_input("Ta ett foto")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")

# Bildanalys
if image is not None:
    # Centrerar bilden och tar bort onödig bakgrund
    width, height = image.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2

    image_cropped = image.crop((left, top, right, bottom))

    # Förbereder bilden för modellen
    image_resized = image_cropped.resize((224, 224))
    image_array = np.array(image_resized).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    # Prediktion
    prediction = model.predict(image_array, verbose=0)
    probability = prediction[0][0]

    if probability > 0.5:
        result_sv = "Friskt"
        result_en = "Healthy"
        confidence = float(probability)
        status_box = "success"
        explanation = (
            "Modellen bedömer att bladet ser friskt ut och inte visar tydliga tecken på sjukdom."
        )
    else:
        result_sv = "Sjukt"
        result_en = "Disease"
        confidence = float(1 - probability)
        status_box = "error"
        explanation = (
            "Modellen bedömer att bladet visar mönster som liknar sjukdom i träningsdatan."
        )

    # Layout med tre kolumner
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(image, caption="Originalbild", use_container_width=True)

    with col2:
        st.subheader("Analysresultat")

        if status_box == "success":
            st.success(f"**Resultat:** {result_sv} ({result_en})")
        else:
            st.error(f"**Resultat:** {result_sv} ({result_en})")

        st.write(f"**Säkerhet:** {confidence:.2%}")
        st.progress(confidence)

        st.markdown("**Tolkning:**")
        st.write(explanation)

        if confidence < 0.70:
            st.warning(
                "Modellen är inte helt säker. Resultatet bör tolkas försiktigt."
            )

    # Bonus: visar vilken del av bilden modellen analyserar
    st.markdown("### Bonus: Bilden som modellen analyserar")
    st.image(image_cropped, caption="Centrerad och beskuren bild", width=300)

st.markdown("---")
st.subheader("Hur fungerar appen?")
st.write(
    "Appen använder en förtränad CNN-modell (**MobileNetV2**) och transfer learning "
    "för att klassificera växtblad som friska eller sjuka."
)

st.subheader("Användningsområde")
st.write(
    "Lösningen kan användas inom smart agriculture, kvalitetskontroll och tidig "
    "upptäckt av växtsjukdomar."
)

# -------------------------------------------------------
# Affärsvärde, möjligheter och utmaningar med bildanalys
# -------------------------------------------------------

# Denna lösning visar hur bildanalys med hjälp av AI kan skapa affärsvärde
# inom exempelvis jordbruk, livsmedelsproduktion och kvalitetskontroll.
# Genom att automatiskt identifiera växtsjukdomar kan företag upptäcka problem
# i ett tidigt skede, minska manuellt arbete och optimera resursanvändningen.

# Ett viktigt affärsvärde är möjligheten att fatta snabbare och mer datadrivna beslut,
# vilket kan bidra till ökad produktivitet och minskade kostnader. Tekniken kan även
# integreras i mobila applikationer eller IoT-lösningar för realtidsövervakning.

# Samtidigt finns vissa utmaningar. Modellens noggrannhet beror starkt på kvaliteten
# och variationen i träningsdatan. Bilder från verkliga miljöer kan skilja sig från
# träningsdatasetet, vilket kan påverka resultatet negativt.

# Ur ett etiskt perspektiv är det viktigt att användaren förstår att modellen är ett
# beslutsstöd och inte en ersättning för expertbedömning. Felaktiga klassificeringar
# kan annars leda till felaktiga beslut.

# I framtiden kan modellen förbättras genom större dataset, fler sjukdomsklasser och
# bättre anpassning till verkliga användningsmiljöer.