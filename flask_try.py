# third-party HTTP client library
import requests
from flask import Flask, jsonify, request, Response, render_template
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)

# assume that "app" below is your flask app, and that
# "Response" is imported from flask.

# @app.route("/abc/")
# def receiver():
#     return 0 

@app.route('/')
def main_page():
    return render_template("main.html")

# @app.route('/upload')
# def upload_file():
#     return """
#     <html>
#     <body>
#       <form action = "http://localhost:5000/uploader" method = "POST" 
#          enctype = "multipart/form-data">
#          <input type = "file" name = "file" />
#          <input type = "submit"/>
#       </form>   
#     </body>
#     </html>
#     """
#    return render_template('./abcd.html')

	
@app.route('/uploader', methods = ['POST'])
def get_uploaded_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
    return render_template("uploader.html")
    
		
if __name__ == '__main__':
   app.run(debug = True)