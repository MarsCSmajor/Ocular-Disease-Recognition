# Ocular Disease Recognition

This project aims to build a machine learning model to classify final ocular diagnoses based on images, age, sex, and diagnostic text. The final model and analysis will be accessible through a web interface.

---

## 📅 Project Timeline & Milestones

---

### **Phase 1: Project Proposal & Data Collection**  
**Deadline: April 29, 2025**

#### **Tasks:**
- Download and store the dataset on an EC2 instance.
- Preprocess structured data and store in a structured format.
- Test basic Spark functions for data manipulation and validation.
- Document dataset statistics (record count, attributes, distribution).

#### **Deliverables:**
- Project proposal document
- Data uploaded to EC2
- Initial Spark code and screenshots

#### **References:**
- [Spark Cluster Mode Overview - Spark 3.5.5 Documentation](https://spark.apache.org/docs/latest/cluster-overview.html)
- [GitHub - amplab/spark-ec2](https://github.com/amplab/spark-ec2)
- [Project Proposal Document](https://docs.google.com/document/d/1-asX-Gn_jQ_FPAIIX8xtVH_RwHyVZhlWpB_nUczI8AY/edit?tab=t.0)

---

### **Phase 2.1: Data Processing**  
**Deadline: May 12, 2025**

#### **Tasks:**
- **Feature Engineering:**
  - Convert age into age groups (e.g., 0–20, 21–40, etc.)
  - One-hot encode sex (e.g., male: 0, female: 1)
  - Encode diagnoses as integers

- **Text Featurization:**
  - Use NLP techniques: TF-IDF, Word2Vec, or BERT embeddings to convert diagnostic text to numerical features

- **Image Feature Extraction:**
  - Use data augmentation to increase training diversity
  - Extract image embeddings using pre-trained CNNs (e.g., ResNet, VGG)
  - Use OpenCV for manual features (e.g., color histograms)

- **Data Integration:**
  - Combine image, text, and structured features
  - Store processed data in MySQL or Cassandra for downstream use

#### **Deliverables:**
- Augmented dataset with feature-engineered attributes
- Documentation describing processing steps

#### **References:**
- [ML Pipelines - Spark 3.5.4](https://spark.apache.org/docs/latest/ml-pipeline.html)
- [Spark SQL & DataFrames Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [MySQL 8.4 Documentation](https://dev.mysql.com/doc/refman/8.4/en/)
- [JDBC Guide - Spark 3.5.4](https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html)

---

### **Phase 2.2: Model Training**  
**Deadline: May 20, 2025**

#### **Tasks:**
- Train classification models (Random Forest, GBT, Logistic Regression) using extracted features
- Build separate models with and without text-based diagnostics

#### **Performance Evaluation:**
- Use metrics: Accuracy, Precision, Recall, F1-Score
- Tune hyperparameters using cross-validation
- Compare training time with different Spark worker configurations

#### **Deliverables:**
- Trained models (with and without diagnostic text)
- Model performance report
- Execution time graphs for various cluster configurations

#### **References:**
- [Classification & Regression - Spark MLlib](https://spark.apache.org/docs/latest/ml-classification-regression.html)
- [ML Tuning - Spark](https://spark.apache.org/docs/latest/ml-tuning.html)

---

### **Phase 3: Web Interface Development**  
**Timeline: May 20 – June 3, 2025**

#### **Tasks:**
- **Frontend Implementation (Flask):**
  - Search/filter patient data by demographics and diagnoses
  - Visualize eye image sets for exploration
  - Show documentation and visual workflow of processing & training pipeline

- **Model Deployment:**
  - Allow real-time diagnosis prediction
  - Display similar eye images based on predicted diagnosis

- **Backend Integration:**
  - Connect to MySQL/Cassandra to retrieve processed data
  - Implement API endpoints for data queries and predictions

#### **Deliverables:**
- Web UI with flexible filter options:
  - Example filters: "eyes of women age 50+ with cataracts", "eyes age 60+ with no ocular disease"
- Diagnosis prediction input form
- Interactive results and visual feedback
- API for diagnosis prediction

---

## 📊 Analysis & Insights

- Feature importances visualized per model
- Performance comparison of models (text vs. no-text input)
- Exploration of correlations between features and diagnoses

---

## 🛠 Technologies Used

- **Apache Spark 3.5.4**
- **AWS EC2**
- **Flask (Web Interface)**
- **MySQL / Cassandra**
- **OpenCV, TF-IDF, Word2Vec, BERT, ResNet/VGG**



## Image Feature Creation DAG
![imageDAG](https://github.com/user-attachments/assets/445205c7-2c1a-4559-bea3-3132b5c57fed)

## Feature Joining DAG
![feature_extraction_DAG](https://github.com/user-attachments/assets/601fe250-c233-4d1a-8c33-bd216adf609b)


## Tools Used
