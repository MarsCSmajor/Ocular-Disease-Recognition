from flask import Flask, render_template, url_for

#Creating Flask application instance
app = Flask(__name__)

#Defining URLs (routes)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
