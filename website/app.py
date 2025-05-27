from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route("/")
def background():
    return render_template("background.html", 
                           background = url_for('background'),
                           statistics = url_for('stats'), 
                           prediction = url_for('prediction'))

@app.route("/statistics")
def stats():
    return render_template("background.html", 
                           background = url_for('background'),
                           statistics = url_for('stats'), 
                           prediction = url_for('prediction')) 

@app.route("/prediction")
def prediction():
    return render_template("background.html", 
                           background = url_for('background'),
                           statistics = url_for('stats'), 
                           prediction = url_for('prediction')) 




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