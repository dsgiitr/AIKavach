# third-party HTTP client library
import requests
from flask import Flask, jsonify, request, Response, render_template, redirect
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
  result = []
  return result


form_dict = []

@app.route('/', methods=["GET", "POST"])
def main_page():
  if request.method == 'POST':
      try: # to prevent empty form from being sent empty
        f = request.files['model_file']
        f.save("./models/"+secure_filename(f.filename))  
      except:
        return render_template("main.html")
      example_tested = dict(request.form.lists())
      if example_tested['dataset'][0] == "custom": # to check whether we have dataset values
        try:
          f = request.files['dataset_file']
          f.save("./custom_datasets/"+secure_filename(f.filename))
        except:
          return render_template("main.html")
      form_dict.append(example_tested)
      # check if the file has a pth extension
      return redirect("/uploader")
  return render_template("main.html")


@app.route('/get_results', methods=["GET"])
def get_results():
  # the API that returns results when they are compiled
  
  # you just need to add the function here that produce the results and then it is good to go
  '''
  def example_func(model):
    .... run some process 

    return results

  # TODO: Make this subAPI in such a manner that it responds to multiple requests at a time using auth and async functions 
  '''
  # Execute the first Bash script
  script1_path = '/path/to/script1.sh'
  p1 = subprocess.Popen(['bash'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out1, err1 = p1.communicate()

  # Execute the second Bash script
  script2_path = '/path/to/script2.sh'
  p2 = subprocess.Popen(['bash', script2_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out2, err2 = p2.communicate()

  # Check for any errors
  if err1 or err2:
    error = 'An error occurred while executing the scripts'
    return jsonify({'error': error})
  results = [
        {"image_name": "image1.jpg", "object": "person", "confidence": 0.95},
        {"image_name": "image1.jpg", "object": "car", "confidence": 0.85},
        {"image_name": "image2.jpg", "object": "dog", "confidence": 0.75},
        {"image_name": "image2.jpg", "object": "cat", "confidence": 0.65},
    ]
  return jsonify(results)

@app.route('/uploader')
def display_result():
  return render_template("uploader.html")

if __name__ == '__main__':
   app.run(debug = True)