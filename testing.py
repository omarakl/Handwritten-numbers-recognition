import cv2               # OpenCV for image processing
import numpy as np       # NumPy for numerical operations
import matplotlib as plt  # Matplotlib for displaying images
import tensorflow as tf  # TensorFlow for machine learning
import os               # OS module for file operations

# loading the pre-trained model (MNIST)
model = tf.keras.models.load_model('Handwritten_Numbers_project/handwritten_numbers.model')

# variable to track the image number
image_number = 1

# iterates through the images found in the directory
while os.path.isfile(f"Handwritten_Numbers_project/digits/digit{image_number}.png"):
    try:
        # reading the image using openCV and extract the first color channel (grayscale)
        img = cv2.imread(f"Handwritten_Numbers_project/digits/digit{image_number}.png")[:, :, 0]

        # inverting the colors (black turns white / white turns black)
        img = np.invert(np.array([img]))

        # make a prediction
        prediction = model.predict(img)

        # printing the prediction with highest probability
        print(f"This digit is probably a {np.argmax(prediction)}")

        # displaying the image with matplotlib
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    
    except:
        # handle any exceptions
        print('')
    
    finally:
        # incrementing the counter to change images
        image_number += 1
