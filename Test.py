import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from streamlit_drawable_canvas import st_canvas
import time
import base64
import os

def load_model():
    model = InceptionV3()
    return model


def preprocess_image(img):
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def predict_image(img, model):
    img = preprocess_image(img)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions


def recognize_uploaded_image():
    st.write("# Afbeelding Herkenning - Upload een Afbeelding")
    st.write('Upload hier uw afbeelding van het type jpg, jpeg of png om uw dier te herkennen.')
    # Afbeelding uploaden
    uploaded_file = st.file_uploader("Selecteer een afbeelding", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Afbeelding weergeven
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Laden van het model
        model = load_model()

        # ImageNet-classificatie uitvoeren
        predictions = predict_image(uploaded_file, model)

        # Resultaten weergeven
        st.write("### Voorspellingen:")
        for i, (imagenet_id, label, score) in enumerate(predictions[0]):
            if i == 0:
                st.write(f"**{i + 1}. {label} ({score * 100:.2f}%) - Dit is het voorspelde dier.**")
                animal_name = label.lower()
                animal_link = f"https://en.wikipedia.org/wiki/{animal_name}"
                st.markdown(f"[Meer over {label} op Wikipedia]({animal_link})", unsafe_allow_html=True)
            else:
                st.write(f"{i + 1}. {label} ({score * 100:.2f}%)")

        # Dierenherkenning
        top_prediction_score = predictions[0][0][2]
        if top_prediction_score > 0.5:
            st.success("Deze afbeelding bevat een dier!")
        else:
            st.warning("Geen dier gevonden in de afbeelding.")

def recognize_webcam_image():
    st.write("# Afbeelding Herkenning - Webcam")
    st.write("Hier kan u uw webcam gebruiken om een dier te herkennen.")
    # Gebruik de webcam
    cap = cv2.VideoCapture(0)

    if st.button("Maak foto"):
        progress_placeholder = st.empty()

        for i in range(5, 0, -1):
            progress_placeholder.text(f"Fototimer: {i} seconden")
            time.sleep(1)

        progress_placeholder.empty()

        ret, frame = cap.read()
        st.image(frame, channels="BGR")

        # Opslaan van de afbeelding
        cv2.imwrite("webcam_image.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Laden van het model
        model = load_model()

        # ImageNet-classificatie uitvoeren
        predictions = predict_image("webcam_image.jpg", model)

        # Resultaten weergeven
        st.write("### Voorspellingen:")
        for i, (imagenet_id, label, score) in enumerate(predictions[0]):
            if i == 0:
                st.write(f"**{i + 1}. {label} ({score * 100:.2f}%) - Dit is de voorspelde dier.**")
                animal_name = label.lower()
                animal_link = f"https://en.wikipedia.org/wiki/{animal_name}"
                st.markdown(f"[Meer over {label} op Wikipedia]({animal_link})", unsafe_allow_html=True)
            else:
                st.write(f"{i + 1}. {label} ({score * 100:.2f}%)")

        # Dierenherkenning
        top_prediction_score = predictions[0][0][2]
        if top_prediction_score > 0.5:
            st.success("Deze afbeelding bevat een dier!")
        else:
            st.warning("Geen dier gevonden in de afbeelding.")
    cap.release()

def recognize_drawing():
    st.write("# Afbeelding Herkenning - Tekenen en Herkennen")

    # Canvas om op te tekenen
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.3)",  # Oranje transparante kleur
        stroke_width=10,
        stroke_color="rgb(0, 0, 0)",
        background_color="#fff",
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Herken tekening"):
        if canvas_result.image_data is not None:
            # Convert canvas drawing to image
            drawn_image = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2RGB)

            # Resize de afbeelding naar (299, 299), wat het verwachte formaat is voor InceptionV3
            drawn_image = cv2.resize(drawn_image, (299, 299))

            # Convert the drawn image to a format expected by InceptionV3
            drawn_image = np.expand_dims(drawn_image, axis=0)
            drawn_image = preprocess_input(drawn_image)

            # Voer de getekende afbeelding in het model
            model = load_model()
            predictions = model.predict(drawn_image)

            # Resultaten weergeven
            st.write("### Top 5 Voorspellingen:")

            # Get the top 5 predictions
            top_predictions = decode_predictions(predictions, top=5)[0]

            # Display the top 5 predictions
            for i, (imagenet_id, label, score) in enumerate(top_predictions):
                st.write(f"{i + 1}. {label} ({score * 100:.2f}%)")

            # Dierenherkenning
            top_prediction_score = top_predictions[0][2]
            if top_prediction_score > 0.95:
                st.success("Deze getekende afbeelding bevat een dier!")
            else:
                st.warning("Geen dier gevonden in de getekende afbeelding.")
        else:
            st.warning("Er is geen tekening om te herkennen.")
def main():
    # Convert favicon image to Base64
    favicon_path = os.path.join(".", "favicon", "favicondisc.png")
    with open(favicon_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the page configuration including the favicon
    st.set_page_config(
        page_title="Afbeelding Herkenner",
        page_icon=f"data:image/png;base64,{encoded_image}",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.sidebar.title("Navigatie")
    app_mode = st.sidebar.radio("Selecteer een optie:", ["Home", "Upload Afbeelding", "Webcam", "Tekenen en Herkennen"])

    if app_mode == "Home":
        st.title("Afbeelding Herkenning App")
        st.write("1. **Upload Afbeelding:** Kies een afbeelding van je lokale opslag en ontdek welk dier erop staat.")
        st.write("2. **Webcam:** Maak een foto met je webcam en laat het model vertellen wat er te zien is.")
        st.write("3. **Tekenen en Herkennen:** Laat je creativiteit de vrije loop, teken iets en ontdek of het model het herkent.")
    elif app_mode == "Upload Afbeelding":
        recognize_uploaded_image()
    elif app_mode == "Webcam":
        recognize_webcam_image()
    elif app_mode == "Tekenen en Herkennen":
        recognize_drawing()


if __name__ == "__main__":
    main()