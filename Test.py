import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from streamlit_drawable_canvas import st_canvas

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
            st.write(f"{i + 1}. {label} ({score * 100:.2f}%)")

        # Dierenherkenning
        top_prediction_score = predictions[0][0][2]
        if top_prediction_score > 0.5:
            st.success("Deze afbeelding bevat een dier!")
        else:
            st.warning("Geen dier gevonden in de afbeelding.")

def recognize_webcam_image():
    st.write("# Afbeelding Herkenning - Webcam")

    # Gebruik de webcam
    cap = cv2.VideoCapture(0)

    if st.button("Maak foto"):
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
            st.write(f"{i + 1}. {label} ({score * 100:.2f}%)")

        # Dierenherkenning
        top_prediction_score = predictions[0][0][2]
        if top_prediction_score > 0.2:
            st.success("Deze afbeelding bevat een dier!")
        else:
            st.warning("Geen dier gevonden in de afbeelding.")
    cap.release()

def recognize_drawing():
    st.write("# Afbeelding Herkenning - Tekenen en Herkennen")

    # Canvas om op te tekenen
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Oranje transparante kleur
        stroke_width=10,
        stroke_color="rgb(255, 165, 0)",
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

            # Voer de getekende afbeelding in het model
            model = load_model()
            predictions = predict_image(drawn_image, model)

            # Resultaten weergeven
            st.write("### Voorspellingen:")
            for i, (imagenet_id, label, score) in enumerate(predictions[0]):
                st.write(f"{i + 1}. {label} ({score * 100:.2f}%)")

            # Dierenherkenning
            top_prediction_score = predictions[0][0][2]
            if top_prediction_score > 0.5:
                st.success("Deze getekende afbeelding bevat een dier!")
            else:
                st.warning("Geen dier gevonden in de getekende afbeelding.")

def main():
    st.title("Afbeelding Herkenning App")

    # Navigatiebalk
    app_mode = st.sidebar.radio("Selecteer een optie:", ["Home", "Upload Afbeelding", "Webcam", "Tekenen en Herkennen"])

    if app_mode == "Home":
        st.write("Deze app stelt je in staat om dieren te herkennen aan de hand van afbeeldingen. Kies een optie in de navigatiebalk om verder te gaan.")
    elif app_mode == "Upload Afbeelding":
        recognize_uploaded_image()
    elif app_mode == "Webcam":
        recognize_webcam_image()
    elif app_mode == "Tekenen en Herkennen":
        recognize_drawing()

if __name__ == "__main__":
    main()
