📝 OMR Evaluation using CNN

This project automates the evaluation of OMR (Optical Mark Recognition) sheets using a Convolutional Neural Network (CNN). Instead of manually checking OMR answer sheets, you can simply feed an image of an OMR sheet into the model, and it will predict the scored marks.

🚀 Features

Accepts scanned/photographed OMR sheet images as input.

Preprocessing of images (grayscale + thresholding).

Trained CNN model that learns to detect filled bubbles and calculate total marks.

Outputs the predicted score directly.

Can be extended for different answer key formats.

Support multiple options type as well

⚙️ How It Works

Image Preprocessing

Converts OMR sheet to grayscale.

Apply Gausian blur and normalises it for better understanding of the cnn model.

CNN Model

Takes the processed OMR image as input.

Detects marked answers.

Compares with the correct answer key.

Calculates the total score.

Output

Predicted score (marks) displayed or saved in a file.

📊 Model Training

Model: Convolutional Neural Network (CNN)

Training Data: Labeled OMR sheet images

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Evaluation Metric: Accuracy

🏆 Results

The trained CNN achieves good accuracy in predicting scores on test OMR sheets.

🙌 Acknowledgments

OpenCV for image preprocessing

TensorFlow/Keras for building CNN model

Dataset prepared from custom OMR sheets
