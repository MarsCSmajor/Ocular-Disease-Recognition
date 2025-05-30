
from flask import Flask, render_template
from flask import request, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/background')
def background():
    return render_template('background.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    image_url = None
    diagnostic_text = None

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            diagnostic_text = request.form.get('diagnostic')
            if image.filename != '':
                filename = secure_filename(image.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(filepath)
                image_url = url_for('static', filename=f'uploads/{filename}')
    
    return render_template('prediction.html', image_url=image_url, diagnostic=diagnostic_text)


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request, url_for

# app = Flask(__name__)

# @app.route("/")
# def background():
#     return render_template("background.html", 
#                            background = url_for('background'),
#                            statistics = url_for('stats'), 
#                            prediction = url_for('prediction'))

# @app.route("/statistics")
# def stats():
#     return render_template("background.html", 
#                            background = url_for('background'),
#                            statistics = url_for('stats'), 
#                            prediction = url_for('prediction')) 

# @app.route("/prediction")
# def prediction():
#     return render_template("background.html", 
#                            background = url_for('background'),
#                            statistics = url_for('stats'), 
#                            prediction = url_for('prediction')) 



'''
ASPECTS NEEDED:

    INHERITABLE BASE TEMPLATE FOR HEADER TO BE REPLICATED ACROSS PAGES 

    BACKGROUND

        DATA ORIGINS *CAN BE FOUND ON KAGGLE 
        Tech Stack we used in general with links to documentation 


    STATS 

        STATISTICS AND GRAPHS WE USED TO VERIFY LEGITIMANCY OF OUR DATA
            COMPARED TO CURRENT KNOWLEDGE SURROUNDING OCULAR DISEASE


    PREDICTION 

        FORM FOR USERS TO UPLOAD THEIR EYE IMAGES + DEMOGRAPHICS 
            TWO SLOTS, TO DROP IMAGE FILES 

            PROCESS THEIR INPUT SERVER SIDE AND THEN RUN THROUGH MODEL 

        ROUTE TO RESULTS PAGE 

        
        RESULTS 

            PREDICTED DIAGNOSIS FOR EACH EYE 
                DISCLOSE MODEL ACCURACY FOR TRANSPARENCY 
                POSSIBLY SHOW ACCURACY FOR EACH DIAGNOSIS TYPE AS WELL? 

            *NEEDS TO BE PROTECTED FROM BEING ROUTED TO WITHOUT INPUT IN PREDICITON FORM

    
'''