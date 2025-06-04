import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Recommended GPU disabling method
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import datetime
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from skimage.feature import local_binary_pattern
import joblib

from sqlalchemy import create_engine
# Optional imports – include only if needed
import subprocess
import csv
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2VecModel

# spark = SparkSession.builder.appName("website").getOrCreate()

spark = SparkSession.builder \
    .appName("Word2VecServing") \
    .master("local[*]") \
    .getOrCreate()

app = Flask(__name__)


# def word2vec(text, eye:bool):


scaler = joblib.load('scaler.save')  # Used only for ResNet features
model_with_word2vec = tf.keras.models.load_model('image_word_model_tf.keras')
model_without_word2vec = tf.keras.models.load_model('image_model_tf.keras')
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
# tf.config.set_visible_devices([], 'GPU')  # Optional: disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Recommended GPU disabling method




left_w2v_model = Word2VecModel.load("model_trained/word2vec_model_left_eye")
right_w2v_model = Word2VecModel.load("model_trained/word2vec_model_right_eye")


def extract_features_from_image(img):
    """Extracts features from image using ResNet and histogram methods."""
    # ResNet feature extraction
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_standardized = (img_normalized - mean) / std
    img_final = np.expand_dims(img_standardized, axis=0)

    resnet_feats = resnet_model.predict(img_final).flatten()
    resnet_feats_scaled = scaler.transform([resnet_feats])[0]  # Only scale ResNet features

    # Color histogram
    chans = cv2.split(img)
    color_hist = []
    for chan in chans:
        h = cv2.calcHist([chan], [0], None, [8], [0, 256])
        h = cv2.normalize(h, h).flatten()
        color_hist.extend(h)
    color_hist = np.array(color_hist)

    # LBP texture features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    texture_hist = hist

    histogram_features = np.hstack([color_hist, texture_hist])
    combined_features = np.hstack([resnet_feats_scaled, histogram_features])
    return combined_features

def predict_image_class(image_path, word2vec_feats=None):
    """Predicts the class of an eye image, optionally using word2vec demographic embeddings."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    image_features = extract_features_from_image(img)

    if word2vec_feats is not None:
        full_input = np.hstack([image_features, word2vec_feats])
        pred_probs = model_with_word2vec.predict(np.expand_dims(full_input, axis=0))
    else:
        pred_probs = model_without_word2vec.predict(np.expand_dims(image_features, axis=0))


    pred_class = int(np.argmax(pred_probs, axis=1)[0])
    return pred_class, pred_probs



def get_word2vec_feature(diagnostic, eye):
    if not diagnostic:
        return None

    try:
        tokens = diagnostic.strip().lower().split()
        df = spark.createDataFrame([(tokens,)], ["text"])

        if eye == 'left':
            transformed = left_w2v_model.transform(df).collect()
        elif eye == 'right':
            transformed = right_w2v_model.transform(df).collect()
        else:
            return None

        return np.array(transformed[0]['text'], dtype=np.float32)

    except Exception as e:
        print(f"[Word2Vec Error]")
        return None

# engine = create_engine("mysql+pymysql://teammate:2025@localhost/tabular_data_stats")
engine = create_engine("mysql+pymysql://teammate:2025@localhost/tabular_data_stats?unix_socket=/home/csmajs/kgonz192/mysql.sock")


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

label_encoder = joblib.load('label_encoder.save')

@app.route('/')
def background():
    return render_template('background.html')

@app.route('/statistics')
def statistics():
    df_sex = pd.read_sql("SELECT * FROM distinct_patients_by_sex", con=engine)
    sex_data = df_sex.to_dict(orient='records')

    df_diag_sex = pd.read_sql("SELECT * FROM distinct_diagnosis_by_sex", con=engine)
    diag_sex_data = df_diag_sex.to_dict(orient='records')

    df_diag_age = pd.read_sql("SELECT * FROM distinct_diagnosis_by_age_group", con=engine)
    diag_age_data = df_diag_age.to_dict(orient='records')

    return render_template(
            'statistics.html',
            sex_data=sex_data,
            diag_sex_data=diag_sex_data,
            diag_age_data=diag_age_data
    )



@app.route('/submit_statistics', methods=['POST'])
def submit_statistics():
    csv_file = 'submissions.csv'

    # Handle form fields first
    verification_code = request.form.get('verification_code')

    if verification_code != '222':
        return render_template('thankyou.html', message="Submission rejected: invalid verification code.")

    # Handle uploaded images
    left_eye_file = request.files.get('left_eye_image')
    right_eye_file = request.files.get('right_eye_image')

    # Handle form fields
    patient_age = request.form.get('patient_age')
    left_diagnosis = request.form.get('left_diagnosis')
    right_diagnosis = request.form.get('right_diagnosis')
    left_notes = request.form.get('left_notes')
    right_notes = request.form.get('right_notes')
    verification_code = request.form.get('verification_code')

    # Filenames and paths
    left_eye_filename = None
    right_eye_filename = None

    if left_eye_file and left_eye_file.filename != '':
        left_eye_filename = secure_filename(f"left_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{left_eye_file.filename}")
        left_eye_path = os.path.join(app.config['UPLOAD_FOLDER'], left_eye_filename)
        left_eye_file.save(left_eye_path)

    if right_eye_file and right_eye_file.filename != '':
        right_eye_filename = secure_filename(f"right_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{right_eye_file.filename}")
        right_eye_path = os.path.join(app.config['UPLOAD_FOLDER'], right_eye_filename)
        right_eye_file.save(right_eye_path)

    # Append to CSV
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                'timestamp', 'patient_age', 
                'left_eye_filename', 'right_eye_filename',
                'left_diagnosis', 'right_diagnosis',
                'left_notes', 'right_notes',
                'verification_code'
            ])
            
        writer.writerow([
            datetime.datetime.now().isoformat(),
            patient_age,
            left_eye_filename,
            right_eye_filename,
            left_diagnosis,
            right_diagnosis,
            left_notes,
            right_notes,
            verification_code
         ])
    
    # Redirect or render a thank you page
    return render_template('thankyou.html', patient_age=patient_age)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    image_url = None
    diagnostic_text = None
    diagnostic_model = None
    features_word2vec = None
    eye = None
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            diagnostic_text = request.form.get('diagnostic')
            eye = request.form.get('eye')
            if image.filename != '':
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                image_url = url_for('static', filename=f'uploads/{filename}')

            # if diagnostic_text:
            #     if eye == 'left':
            #         print("left eye was found")
            #     #calls the word2vec function for the left eye
            #     else:
            #         print(f"{eye} was found")
            #         # calls the word2vec for the right eye


            try:
                features_word2vec = get_word2vec_feature(diagnostic_text, eye)
                predicted_class, predicted_prob = predict_image_class(image_path=filepath, word2vec_feats=features_word2vec)
                pred_label = label_encoder.inverse_transform([predicted_class])[0]
                diagnostic_model = f"{pred_label}, with a {np.max(predicted_prob):.3f} confidence"
                diagnostic_text = str(diagnostic_text)

            except Exception as e:
                print("Prediction failed:", e)

            
                

    return render_template('prediction.html', image_url=image_url, diagnostic=diagnostic_text,diagnostic_on_model = diagnostic_model, eye = eye)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0',port=5000,debug=False, use_reloader=False)
