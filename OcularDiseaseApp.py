from flask import Flask

#Creating Flask application instance
app = Flask(__name__)

#Defining URLs (routes)
@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == "__main__":
    app.run(debug=True)
