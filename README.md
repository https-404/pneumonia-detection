# Pneumonia Detection using Convolutional Neural Network

This repository contains a project focused on using a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The dataset used for training and validation consists of labeled images indicating whether pneumonia is present or not.

## Project Overview

Pneumonia is a serious lung infection that primarily affects the alveoli, causing inflammation and fluid build-up. Early detection is crucial for effective treatment, and this project aims to automate the process using deep learning.

The notebook provides a comprehensive guide to building, training, and evaluating a CNN for the classification of X-ray images as either showing signs of pneumonia or being normal. 

## Key Features

- **Dataset**: Chest X-ray images used for training and validation.
- **Preprocessing**: Images are resized, normalized, and augmented to improve the generalization of the model.
- **Model Architecture**: A CNN model is designed from scratch or using a pre-trained model (details within the notebook).
- **Training**: The model is trained on a labeled dataset using supervised learning techniques.
- **Evaluation**: Performance is evaluated based on accuracy, precision, recall, and F1-score, among others.

## Dataset

The dataset used in this project is publicly available and consists of labeled X-ray images categorized into two classes:

1. **Normal** - X-ray images showing no signs of pneumonia.
2. **Pneumonia** - X-ray images showing signs of pneumonia.

The dataset can be downloaded from the [Kaggle Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Requirements

To run the notebook, you will need the following dependencies:

- Python 3.x
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV (for image processing)
- PIL (Python Imaging Library)

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Project Structure

- `Pneumonia_Detection_using_Conv_Neural_Network.ipynb`: Main notebook containing the implementation of the CNN model.
- `data/`: Folder containing the dataset (ensure this is correctly structured before running the notebook).
- `models/`: Folder where trained models and checkpoints will be saved.
- `README.md`: Project documentation.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/https-404/pneumonia-detection.git
cd pneumonia-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the `data/` directory.

4. Open the Jupyter notebook and run the cells to preprocess data, build the CNN model, and train the model.

```bash
jupyter notebook Bytewise_Final_Areesh_Pneumonia_Detection_using_Conv_Neural_Network.ipynb
```

5. Monitor the training process and evaluate the model performance on the validation set.

## Results

After training, the model achieves good accuracy in detecting pneumonia from X-ray images. Results such as accuracy, confusion matrix, and classification report are available in the notebook.

### Example Results

- **Accuracy**: `85%`
- **Precision**: `X`
- **Recall**: `X`
- **F1-Score**: `X`

(The exact metrics will depend on the final training run.)

## Future Work

- Further improve the accuracy by experimenting with more advanced CNN architectures.
- Add real-time prediction using a Flask or Django-based web application.
- Explore the possibility of transfer learning using pre-trained models like VGG16, ResNet, etc.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

