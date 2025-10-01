import gradio as gr
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os

# --- 1. Global Setup (Load Model and Answer Key once) ---
try:
    # Check if files exist before loading
    if not os.path.exists('omr_cnn.keras'):
        raise FileNotFoundError("omr_cnn.keras not found. Please ensure the model file is in the same directory.")
    if not os.path.exists('answer_key.csv'):
        raise FileNotFoundError("answer_key.csv not found. Please ensure the answer key file is in the same directory.")

    # Load the trained Keras model
    MODEL = tf.keras.models.load_model('omr_cnn.keras')
    print("Model loaded successfully.")

    # Load and Preprocess Answer Key
    df_ans = pd.read_csv("answer_key.csv")
    
    # Clean and binarize the answers (assuming answers are comma-separated in the CSV)
    df_ans["answer"] = df_ans["answer"].apply(lambda x: x.split(","))

    # Initialize MultiLabelBinarizer for A, B, C, D, E options
    mlb = MultiLabelBinarizer(classes=["A", "B", "C", "D", "E"])
    encoded = mlb.fit_transform(df_ans["answer"])
    
    # Global numpy array for the correct answers
    CORRECT_ANSWERS_ARRAY = encoded
    print("Answer key loaded and preprocessed successfully.")

except Exception as e:
    print(f"Error during global setup: {e}")
    # Set placeholders if loading fails to allow the UI to start, but the function will fail gracefully
    MODEL = None
    CORRECT_ANSWERS_ARRAY = None

# --- 2. Core OMR Processing Function ---
def evaluate_omr(omr_image: np.ndarray) -> str:
    """
    Processes an uploaded OMR image, predicts answers using the loaded CNN model,
    and returns the final score and performance message.

    Args:
        omr_image: The OMR sheet image loaded as a numpy array (BGR by default in CV2 context).

    Returns:
        A string containing the score and evaluation message.
    """
    if MODEL is None or CORRECT_ANSWERS_ARRAY is None:
        return "ERROR: Model or Answer Key failed to load during startup. Cannot evaluate."

    try:
        # 1. Resize and Crop (Based on Notebook hardcoded dimensions)
        # Note: CV2 uses BGR channel order, Gradio/PIL might use RGB.
        # Since the subsequent steps use CV2 operations (cvtColor), it should be consistent.
        img1 = cv2.resize(omr_image, (2000, 2500))

        # Define the hardcoded cropping regions for the left and right halves (30 questions total)
        left_half = img1[760:2290, 350:750]
        right_half = img1[760:2290, 1130:1510]

        # 2. Extract and Prepare Slices (30 total)
        images = []
        
        # Process Left Half (Questions 1-15)
        h_left, w_left, _ = left_half.shape
        slice_height_left = h_left // 15
        for i in range(15):
            start = i * slice_height_left
            end = start + slice_height_left
            img_slice = left_half[start:end, :]
            
            # Convert to Grayscale and Blur (as done in the notebook)
            img_slice_gray = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
            img_slice_blur = cv2.GaussianBlur(img_slice_gray, (5, 5), 1)

            # Resize to model input shape (56, 280)
            img_slice_processed = cv2.resize(img_slice_blur, (280, 56))
            
            # Normalize and reshape for CNN input
            img_slice_processed = np.expand_dims(img_slice_processed, axis=-1)
            img_slice_processed = img_slice_processed / 255.0
            images.append(img_slice_processed)

        # Process Right Half (Questions 16-30)
        h_right, w_right, _ = right_half.shape
        slice_height_right = h_right // 15
        for i in range(15):
            start = i * slice_height_right
            end = start + slice_height_right
            img_slice = right_half[start:end, :]
            
            # Convert to Grayscale and Blur
            img_slice_gray = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
            img_slice_blur = cv2.GaussianBlur(img_slice_gray, (5, 5), 1)
            
            # Resize to model input shape (56, 280)
            img_slice_processed = cv2.resize(img_slice_blur, (280, 56))
            
            # Normalize and reshape for CNN input
            img_slice_processed = np.expand_dims(img_slice_processed, axis=-1)
            img_slice_processed = img_slice_processed / 255.0
            images.append(img_slice_processed)


        # 3. Prediction
        images_array = np.array(images)
        # Final reshape: (30, 56, 280, 1)
        images_array = images_array.reshape(-1, 56, 280, 1) 
        
        pred = MODEL.predict(images_array, verbose=0)

        # 4. Thresholding and Scoring
        threshold = 0.5
        predicted_answers = []
        for prediction in pred:
            # Convert continuous prediction to binary (0 or 1)
            multi_label = (prediction >= threshold).astype(int)
            predicted_answers.append(multi_label)

        # Compare predicted answers with the correct answer key
        correct = 0
        total_questions = len(predicted_answers)
        
        for i in range(total_questions):
            # Check if all elements in the predicted row match the correct row
            if (predicted_answers[i] == CORRECT_ANSWERS_ARRAY[i]).all():
                correct += 1

        # 5. Format Output
        if correct >= 20:
            result_message = f"Excellent! You scored {correct} marks out of {total_questions}."
        elif correct >= 10:
            result_message = f"Bingo! You scored {correct} marks out of {total_questions}. Keep practicing!"
        else:
            result_message = f"You scored {correct} marks out of {total_questions}. Needs Improvements."

        return result_message

    except Exception as e:
        return f"An error occurred during processing: {e}"

# --- 3. Gradio Interface Setup ---

# Define the Gradio Interface
iface = gr.Interface(
    fn=evaluate_omr,
    inputs=gr.Image(type="numpy", label="Upload OMR Sheet Image"),
    outputs=gr.Text(label="Evaluation Result"),
    title="CNN-Based OMR Sheet Evaluator",
    description="Upload a standardized OMR sheet image to get an instant score and feedback. The app requires 'omr_cnn.keras' and 'answer_key.csv' in the directory to function.",
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

