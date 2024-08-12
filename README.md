# Potato Leaf Disease Detection

This project is a web application that detects diseases in potato leaves using a Convolutional Neural Network (CNN) model. The application is built using Streamlit and TensorFlow, and it can classify potato leaf images into three categories: Early Blight, Late Blight, or Healthy. Additionally, it provides remedy suggestions based on the detected disease.

## How It Works

1. **Upload Image**: The user uploads an image of a potato leaf.
2. **Model Prediction**: The app preprocesses the image, runs it through the trained CNN model, and predicts the disease.
3. **Confidence Score**: The app displays the confidence level of the prediction.
4. **Remedy Information**: The app provides remedies for the detected disease.

## Model Characteristics

### Architecture
- Model type: Convolutional Neural Network (CNN)
- Number of layers: 6 convolutional layers, 2 fully connected layers
- Input shape: 256x256x3 RGB images
- Output: 3 classes (Early Blight, Late Blight, Healthy)

### Training
- Number of training images: 2152
- Image preprocessing: Resizing to 256x256, Rescaling pixel values to 0-1 range
- Data augmentation techniques: Random horizontal and vertical flipping, Random rotation (up to 20 degrees)
- Optimizer: Adam
- Learning rate: 0.001 (default for Adam optimizer)
- Number of epochs: 25
- Early stopping: Implemented with patience of 5 epochs, monitoring validation accuracy
- Batch size: 32

### Performance
- Accuracy on test set: 98.05%
- Loss on test set: 0.0410

### Model Size
- Number of parameters: 183,747
- Model file size: 2.3 MB

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/potato-leaf-disease-detection.git
   cd potato-leaf-disease-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Technologies Used

- Streamlit: For building the web application interface.
- TensorFlow: For loading and running the trained CNN model.
- PIL: For image processing.
- NumPy: For handling image data as arrays.
