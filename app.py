from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import imutils

# Load the saved model
model_path =r'C:\Users\sneha\OneDrive\Desktop\gan final project\gan final project\Gan project final\single image dehazing project\original project\dehaze_object\Single-Image-Dehazing-Python-master\trained_model'
loaded_model = tf.keras.models.load_model(model_path, compile=False)

# Function to load and preprocess the image
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=(384, 384), antialias=True)
    img = img / 255.0
    return img

# Function to process the uploaded image and generate the dehazed image
def process_image(image_path, model):
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)
    dehazed_image = model(image, training=False)
    return dehazed_image[0]

# Create a Flask app
app = Flask(__name__)

UPLOAD_FOLDER = r'static\uploads'
OUTPUT_FOLDER = r'static\outputImages'
OBJECT_FOLDER = r'static\objectImages'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['OBJECT_FOLDER'] = OUTPUT_FOLDER

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe('C:\\Users\\sneha\\OneDrive\Desktop\\gan final project\\gan final project\\Gan project final\\single image dehazing project\\original project\\dehaze_object\\Single-Image-Dehazing-Python-master\\MobileNetSSD_deploy.prototxt.txt','C:\\Users\\sneha\\OneDrive\\Desktop\\gan final project\\gan final project\\Gan project final\\single image dehazing project\\original project\\dehaze_object\\Single-Image-Dehazing-Python-master\\MobileNetSSD_deploy.caffemodel')
def calculate_image_quality(original_image_path, compressed_image_path):
    original_image = cv2.imread(original_image_path)
    compressed_image = cv2.imread(compressed_image_path)

    # Resize the original image to match the dimensions of the compressed image
    original_image_resized = cv2.resize(original_image, (compressed_image.shape[1], compressed_image.shape[0]))

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2GRAY)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original_gray - compressed_gray) ** 2)

    # Calculate Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255 / np.sqrt(mse))

    # Calculate Structural Similarity Index (SSIM)
    ssim_value = ssim(original_gray, compressed_gray, multichannel=True)

    return mse, psnr, ssim_value


# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route to handle the file upload and processing
@app.route('/upload', methods=['GET','POST'])
def upload_file():
    processed_img = None
    if request.method == 'POST':
        # Check if the file is present in the request
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        # If the file has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            original_img_path = file_path
            processed_img = process_image(file_path, loaded_model)

            # Convert processed_img (EagerTensor) to numpy array
            processed_img_array = np.array(processed_img * 255, dtype=np.uint8)

            # Save the processed image
            processed_img_filename = 'processed_image.jpg'
            processed_img_path = os.path.join(app.config['OUTPUT_FOLDER'], processed_img_filename)
            cv2.imwrite(processed_img_path, processed_img_array)

            # Calculate object detection and save the result
            image = cv2.imread(processed_img_path)
            # Perform object detection
            # (your object detection code here)
            # Save the detected object image
            detected_object_filename = 'detected_object.jpg'
            detected_object_path = os.path.join(app.config['OUTPUT_FOLDER'], detected_object_filename)
            cv2.imwrite(detected_object_path, image)
            
            
            image = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], processed_img_filename))
            image = imutils.resize(image, width=400)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(CLASSES):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            cv2.imwrite(os.path.join(app.config['OBJECT_FOLDER'], 'processed_detection.jpg'), image)
            
            # Calculate image quality
            if original_img_path:
                mse, psnr, ssim_value = calculate_image_quality(original_img_path, processed_img_path)
                print("MSE:", mse)
                print("PSNR:", psnr)
                print("SSIM:", ssim_value)

            return render_template('result.html', processed_img=processed_img_filename, filename=filename)


    return render_template('index.html')
           
@app.route('/object')
def object():
    return render_template('object.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



