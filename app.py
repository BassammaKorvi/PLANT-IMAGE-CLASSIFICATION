import streamlit as st
import tensorflow as tf
from PIL import Image

# Load your trained model here
model = tf.keras.models.load_model('C:/Users/chanu/PycharmProjects/internship/potato.h5')

# Define a function to preprocess the image and make predictions
def classify_image(image):
    # Preprocess the image
    image = image.resize((128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # Make predictions with the model
    predictions = model.predict(image)

    # Get the class with the highest probability
    class_idx = tf.math.argmax(predictions, axis=1)[0]
    class_names = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']  # Replace with your own class names
    class_name = class_names[class_idx]

    return class_name

def main():
    st.title("Plant Image Classification App")
    st.write("Upload an image and the app will classify it!")

    # File uploader widget
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify the image
        predicted_class = classify_image(image)

        # Show the prediction
        st.write(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
