import cv2
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.applications import ResNet50
from skimage.feature import local_binary_pattern

# 1. Load scaler and model (assumes scaler saved as 'scaler.save' and model saved as 'image_word_model_tf.keras')
scaler = joblib.load('scaler.save')
model = tf.keras.models.load_model('image_word_model_tf.keras')

# 2. Load ResNet50 model for feature extraction
tf.config.set_visible_devices([], 'GPU')  # disable GPU if necessary
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_image(img):
    # Extract ResNet features
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_standardized = (img_normalized - mean) / std
    img_final = np.expand_dims(img_standardized, axis=0)
    resnet_features = resnet_model.predict(img_final).flatten()

    # Extract color histogram
    chans = cv2.split(img)
    color_hist = []
    for chan in chans:
        h = cv2.calcHist([chan], [0], None, [8], [0, 256])
        h = cv2.normalize(h, h).flatten()
        color_hist.extend(h)
    color_hist = np.array(color_hist)

    # Extract texture histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    texture_hist = hist

    # Combine color + texture histograms
    histogram_features = np.hstack([color_hist, texture_hist])

    return resnet_features, histogram_features

def prepare_features_for_model(resnet_feats, histogram_feats, word2vec_feats=None):
    # Normalize ResNet features with saved scaler
    resnet_norm = scaler.transform([resnet_feats])  # scaler expects 2D input
    
    # Handle word2vec features - if you don't have them at inference, use zero vector of proper size
    if word2vec_feats is None:
        # Combine all features horizontally
        combined_features = np.hstack([resnet_norm, histogram_feats.reshape(1, -1)])
        return combined_features
        
    
    # Combine all features horizontally
    combined_features = np.hstack([resnet_norm, histogram_feats.reshape(1, -1), word2vec_feats])
    return combined_features

def predict_image_class(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image {image_path} could not be loaded")

    # Extract features
    resnet_feats, histogram_feats = extract_features_from_image(img)

    # Prepare feature vector
    features = prepare_features_for_model(resnet_feats, histogram_feats)

    # Predict class probabilities
    pred_probs = model.predict(features)
    pred_class = np.argmax(pred_probs, axis=1)[0]

    return pred_class, pred_probs

if __name__ == "__main__":
    test_image_path = "your_test_image.jpg"
    pred_class, pred_probs = predict_image_class(test_image_path)
    print(f"Predicted class: {pred_class}")
    print(f"Class probabilities: {pred_probs}")
