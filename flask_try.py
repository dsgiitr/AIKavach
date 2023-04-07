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

# if __name__ == '__main__':
#     app.run(debug = True)

@app.route('/')
def main_page():
    return """
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
      <style>
      body{ background-color: #89CFF0;
      margin-left: 75px;

      font-family: Sans-serif;}
      h1{
        font-size: 100px;
        
      }
      </style>
    </head>
    <body>
      
      <div style="display: flex">
      <form action = "http://localhost:5000/uploader"" method = "POST" 
         enctype = "multipart/form-data" >
         <h1 style="margin-top:150px;">AISec</h1> 
         <input type = "file" name = "file" />
         <input type = "submit" style="margin-top:5px; width: 90px;" class="btn btn-primary"/>
      </form>
      <div style="margin-left:300px; margin-top:100px">
    <img src="https://media.istockphoto.com/id/1280478624/vector/robot-or-bot-reading-ai-exercise-book.jpg?s=612x612&w=0&k=20&c=qDmpUu2sfzayFfYGsy_BqdW2e91bsh1QnjHbHCyHvJY=" style="width: 480px; height: 350px" alt="Italian Trulli">
    </div>
    </div>
    </body>
    </html>
    """

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
        
        return """
          <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
                <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
                <style>
                  body{ background-color: #89CFF0;
                  margin-left: 75px;
                  font-family: Sans-serif;}
                </style>
          </head>
          <p>file uploaded successfully</p>
          </html>
          """
    
		
if __name__ == '__main__':
   app.run(debug = True)