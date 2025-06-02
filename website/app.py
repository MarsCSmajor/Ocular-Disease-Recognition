
from flask import Flask, render_template
from flask import request, url_for
from werkzeug.utils import secure_filename
import os
import subprocess


from eval import model_image_prediction

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def background():
    return render_template('background.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

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
