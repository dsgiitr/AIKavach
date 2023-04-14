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

@app.route('/', methods=["GET", "POST"])
def main_page():
    if request.method == 'POST':
        # getting input with name = fname in HTML form
        print(request.form)
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return render_template("uploader.html")
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
		
if __name__ == '__main__':
   app.run(debug = True)