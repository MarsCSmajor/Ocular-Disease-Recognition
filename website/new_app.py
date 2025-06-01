from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.applications import ResNet50
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load assets once at startup
scaler = joblib.load('scaler.save')  # Used only for ResNet features
model_with_word2vec = tf.keras.models.load_model('image_word_model_tf.keras')
model_without_word2vec = tf.keras.models.load_model('image_model_tf.keras')
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
tf.config.set_visible_devices([], 'GPU')  # Optional: disable GPU

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





# After loading other assets
label_encoder = joblib.load('label_encoder.save')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    image_url = None
    diagnostic_text = None
    prediction_result = None
    error_message = None

    print("[DEBUG] /prediction endpoint called, method:", request.method)

    if request.method == 'POST':
        diagnostic_text = request.form.get('diagnostic')
        print("[DEBUG] Diagnostic text received:", diagnostic_text)
        word2vec_feats = None  # Replace with actual demographic embedding if available

        if 'image' in request.files:
            image = request.files['image']
            print("[DEBUG] Image file received:", image.filename)
            if image.filename != '':
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                image_url = url_for('static', filename=f'uploads/{filename}')
                print("[DEBUG] Image saved at:", filepath)

                try:
                    pred_class, pred_probs = predict_image_class(filepath, word2vec_feats)
                    pred_label = label_encoder.inverse_transform([pred_class])[0]
                    print("[DEBUG] Prediction made, class index:", pred_class, "label:", pred_label)
                    prediction_result = {
                        'class': pred_label,
                        'probabilities': pred_probs.tolist()
                    }
                except Exception as e:
                    error_message = f"Error processing image: {str(e)}"
                    print("[ERROR] Exception during prediction:", error_message)
            else:
                print("[DEBUG] No image filename provided.")
        else:
            print("[DEBUG] No image part in request.files")
    else:
        print("[DEBUG] Request method is not POST")

    if prediction_result is None:
        print("[DEBUG] prediction_result is None before rendering template")
    else:
        print("[DEBUG] prediction_result is set:", prediction_result)

    return render_template('prediction.html',
                           image_url=image_url,
                           diagnostic=diagnostic_text,
                           prediction=prediction_result['class'] if prediction_result else None,
                           error=error_message)


@app.route('/')
def background():
    return render_template('background.html')


@app.route('/statistics')
def statistics():
    return render_template('statistics.html')


if __name__ == '__main__':
    app.run(debug=True)
