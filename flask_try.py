# third-party HTTP client library
import requests
from flask import Flask, jsonify, request, Response, render_template
from werkzeug.utils import secure_filename
import requests
import os
import subprocess, sys, time

app = Flask(__name__)

# TODO:ADD the function and the parameters
# I am imagining the command to be taking a file_name and some parameters and return the output file when it becomes available 
'''
The parameters to be taken by the file are 
python dsrs/sampler.py mnist location_to_model_weights 0.50 --disttype general-gaussian --k 380 --N 50000 --alpha 0.0005 --skip 10 --batch 400

1. Input_file_name
2. Model(with layers)
3. sigma: default=0.5
4. Distributon_type: default=general-gaussian
5. the training dataset : default=mnist
6. k (or the formula to calculate k)
7. N: default = 
8. Alpha : default = 
9. batch: default = 400
'''
def calculate_np_radius(
  input_file_name, 
  model,
  sigma= 0.5,
  Distributon_type="general-gaussian",
  training = "mnist",
  k =380,
  N=100000,
  Alpha = 0.05,
  batch = 400
):
  # functions to mimic the subprocess
  proc = subprocess.Popen(
        ['bash', './nn.sh', 'localhost', '3'],
        stdout=subprocess.PIPE,
    )
  outf = sys.stdout
  for line in iter(proc.stdout.readline, ''):
      outf.write( 'Reader thread ...\n' )
      outf.write( line.rstrip() + '\n' )
      outf.flush()


# assume that "app" below is your flask app, and that
# "Response" is imported from flask.

# @app.route("/abc/")
# def receiver():
#     return 0 

# if __name__ == '__main__':
#     app.run(debug = True)

@app.route('/upload')
@app.route("/")
def upload_file():
    return """
    <html>
    <body>
      <form action = "http://localhost:5000/uploader" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>   
    </body>
    </html>
    """
#    return render_template('./abcd.html')

	
@app.route('/uploader', methods = ['POST'])
def get_uploaded_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename)) 
        calculate_np_radius(f.filename, "resnet")
        return "file uploaded successfully"
    
		
if __name__ == '__main__':
   app.run(debug = True)