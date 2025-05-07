from pyspark.sql import SparkSession
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50

from skimage.feature import local_binary_pattern

import glob

# Start Spark session
spark = SparkSession.builder.appName("ParallelImageFeatureExtraction").getOrCreate()

def process_images(image_paths):
    '''
    # --- Local function: color histogram ---
    def compute_color_histogram(image, bins=8):
        chans = cv2.split(image)
        hist = []
        for chan in chans:
            h = cv2.calcHist([chan], [0], None, [bins], [0, 256])
            h = cv2.normalize(h, h).flatten()
            hist.extend(h)
        return hist

    # --- Local function: texture histogram ---
    def compute_texture_histogram(gray_image):
        from skimage.feature import local_binary_pattern
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist.tolist()
    '''

    # --- Per-partition function ---
    def resnet_extraction(partition):
        import tensorflow as tf
        import cv2
        import numpy as np
        from tensorflow.keras.applications import ResNet50

        # Set TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')

        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        for path in partition:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue

                # Preprocess for ResNet
                resnet_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resnet_img = cv2.resize(resnet_img, (224, 224))
                resnet_img = resnet_img.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                resnet_img = (resnet_img - mean) / std
                resnet_img = np.transpose(resnet_img, (2, 0, 1))
                resnet_img = np.transpose(resnet_img, (1, 2, 0))
                resnet_img = np.expand_dims(resnet_img, axis=0)
                resnet_features = model.predict(resnet_img).flatten().tolist()

                # Color histogram
                #color_hist = compute_color_histogram(img)

                # Texture histogram
                #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #texture_hist = compute_texture_histogram(gray_img)

                # Combine features
                #all_features = resnet_features + color_hist + texture_hist
                all_features = resnet_features
                yield (path, all_features)

            except Exception as e:
                continue

    # Spark RDD
    rdd = spark.sparkContext.parallelize(image_paths)
    feature_rdd = rdd.mapPartitions(resnet_extraction)
    return feature_rdd

# Example usage
image_paths = glob.glob("/home/devin/seniorDesign/preprocessed_images/*.jpg")
print(image_paths[0])
features_rdd = process_images(image_paths[:1])
features = features_rdd.collect()



from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

# Define schema
schema = StructType([
    StructField("image_path", StringType(), False),
    StructField("features", ArrayType(DoubleType()), False)
])

# Convert RDD to DataFrame
features_df = spark.createDataFrame(features_rdd, schema)


jdbc_url = "jdbc:mysql://localhost:3306/resnet_features?socket=/home/cs179g/mysql.sock"

connection_properties = {
    "user": "teammate",
    "password": "2025",
    "driver": "com.mysql.cj.jdbc.Driver"
}


features_df.write.jdbc(
    url=jdbc_url,
    table="_features",
    mode="overwrite",  # or "append" if table already exists and you want to add to it
    properties=connection_properties
)
