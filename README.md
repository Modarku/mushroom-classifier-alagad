# Mushroom Classifier

The Mushroom Classifier is a mobile-friendly deep learning application that identifies common mushroom genera from images. Built on an augmented MobileNetV3 architecture, the model was trained on the [maysee/mushrooms-classification-common-genuss-images dataset](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images), which contains thousands of labeled mushroom photos across common genera.

This application is simply an experiment to learn about image classification utilizing Keras and Tensorflow.

[Open in Streamlit](https://mushroom-classifier-alagad.streamlit.app/)

## How to run it on your own machine
### ⚠️ Python Version Warning
> **This application is not compatible with Python 3.13 or above.**  
> Please use **Python 3.8–3.12** to avoid issues with TensorFlow installation.  
> Will update the warning when Tensorflow supports Python 3.13.

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
