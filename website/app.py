
from flask import Flask, render_template
from flask import request, url_for
from werkzeug.utils import secure_filename
#from sqlalchemy import create_engine
import pandas as pd
import os
import subprocess
import datetime
import csv


from eval import model_image_prediction

app = Flask(__name__)
#engine = create_engine("mysql+pymysql://teammate:2025@localhost/tabular_data_stats")


# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def background():
    return render_template('background.html')

@app.route('/statistics')
def statistics():
    # Read from MySQL
    #df_sex = pd.read_sql("SELECT * FROM distinct_patients_by_sex", con=engine)

    # Convert to list of dicts for Jinja
    #sex_data = df_sex.to_dict(orient='records')

    #return render_template('statistics.html', sex_data=sex_data)
    return render_template('statistics.html')



@app.route('/submit_statistics', methods=['POST'])
def submit_statistics():
    csv_file = 'submissions.csv'

    # Handle form fields first
    verification_code = request.form.get('verification_code')

    if verification_code != '222':
        return render_template('thankyou.html', 
                               message="Submission rejected: invalid verification code.")

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
    eye = None
    
    prediction = None
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            diagnostic_text = request.form.get('diagnostic')
            if image.filename != '':
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                image_url = url_for('static', filename=f'uploads/{filename}')

            if not diagnostic_text:
                diagnostic_model = model_image_prediction(model_path="image_model_tf.keras",image_path=filepath)
            
                

    return render_template('prediction.html', image_url=image_url, diagnostic=diagnostic_text,diagnostic_on_model = diagnostic_model)


if __name__ == '__main__':
    app.run(debug=True)
