import os
import subprocess
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request, send_file, redirect
import asyncio
import time

app = Flask(__name__)


# TODO:ADD the function and the parameters
# I am imagining the command to be taking a file_name and some parameters and return the output file when it becomes available 
'''
The parameters to be taken by the file are 
python dsrs/sampler.py mnist location_to_model_weights 0.50 --disttype general-gaussian --k 380 --N 50000 --alpha 0.0005 --skip 10 --batch 400

1. Input_file_name
2. form_dict_used
3. sigma: default=0.5
4. Distributon_type: default=general-gaussian
5. the training dataset : default=mnist
6. k (or the formula to calculate k)
7. N: default = 
8. Alpha : default = 
9. batch: default = 400
'''
async def calculate_dsrs_radius(
  denoised_model, 
  form_used: dict,
  sigma= 0.5,
  Distributon_type="general-gaussian",
  training = "mnist",
  k =380,
  N=100000,
  Alpha = 0.05,
  batch = 400,

):
  if form_used["dataset"][0] == "mnist":
    d = 784
    k = 380
    num_classes = 10
  elif form_used["dataset"][0] == "cifar":
    d = 3*32*32
    k = 1530
    num_classes = 10
  else:
    d = int(form_used['channels'])*int(form_used['height'])*int(form_used['width'])
    k = d//2 - 8
    num_classes = int(form_used['n'])
  sigma_p = float(form_used['sigma'][0])

  if  sigma_p >0.2 and sigma_p < 0.8:
    sigma_q = sigma_p - 0.1
  elif sigma_p > 0.8:
    sigma_q = sigma_p - 0.2
  else:
    sigma_q = sigma_p / 2
  secure_model = FinishedModel(denoised_model, d, k, num_classes, form_used["distribution"][0], form_used["distribution"][0], float(form_used['sigma'][0]),float(sigma_q), float(form_used['alpha']), num_sampling_min = 100)
  x = torch.randn((28, 28)).float()
  label = secure_model.label_inference_without_certification(x, int(form_used['N']), 0.01, batch_size = int(form_used['batch_size']))
  logits_old = secure_model.logits_inference_without_certification(x, int(form_used['N']), 0.01, batch_size = int(form_used['batch_size']))
  logits, r = secure_model.inference_and_certification(x,  int(form_used['N']), 0.01, batch_size = int(form_used['batch_size']))
  model_id = form_used["model_id"]
  final_path = f"/final_model_weights/final_model_{model_id}"
  torch.save(final_model,final_path)
  return logits, r

models = [{
    "name":"test1",
    "file_name":"test.pt",
    "timestamp": "",
    "results_calculated":False,
    "id":0,
    "Certified Radius":0.01,
    "model_id":0,
}]

@app.route('/', methods=['GET'])
def dashboard():
    return render_template('index.html', models=models)

@app.route('/form', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        example_tested = dict(request.form.lists())
        try: # to prevent empty form from being sent empty
            f = request.files['model_file']
            f.save("./models/"+secure_filename(f.filename))  
            example_tested['name'] = f.filename.split(".")[0]
            example_tested['file_name'] = f.filename
        except:
            return render_template("form.html")
        if example_tested['dataset'][0] == "custom": # to check whether we have dataset values
            try:
                f = request.files['dataset_file']
                f.save("./custom_datasets/"+secure_filename(f.filename))
                example_tested['dataset_file_name'] = "./custom_datasets/"+secure_filename(f.filename)
            except:
                return render_template("form.html")
        else:
            example_tested['dataset_file_name'] = ""
        example_tested["model_id"] = len(models)
        example_tested["timestamp"] = time.time()
        models.append(example_tested)
        print(models)
        #check if the file has a pth extension
        return redirect("/")
    return render_template('form.html')

@app.route('/calculate_certified_radius',methods=['POST', "GET"] )
def calculate_denoised_form():
    if request.method == 'POST':
        print(request.form.get("model_id"))
        model_id = int(request.form.get("model_id"))
        model_dict = models[model_id]
        denoised_model = f"/denoised/model_{model_id}.pth"
        p1 = subprocess.Popen(['python', 'train.py','-e', '1', '-n', f'{denoised_model}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out1, err1 = p1.communicate()
        if err1:
            error = 'An error occurred while executing the scripts'
            return jsonify({'error': error})
        #Execute the second Bash script
        if not os.path.exists("/final_model_weights/"):
            os.makedirs("/final_model_weights/")
        final_model_path = f"/final_model_weights/final_model_weight_{model_id}.pth"
        logits, r = calculate_dsrs_radius(denoised_model, form_used)
         # Check for any errors
        results = [
            {"confidence Radius": r},
        ]
        models[model_id]["results_calculated"] = True
        return jsonify(results)
    return redirect("/")


@app.route('/download_weights/<model_id>')
def download_weights(model_id):
    # Assuming the updated weights file is located in the "updated_weights" directory
    # with the filename format "model_id_weights_updated.h5"
    final_path = f"/final_model_weights/final_model_{model_id}",

    if os.path.isfile(filepath):
        # If the file exists, send it as a downloadable file
        return send_file(filepath, as_attachment=True), redirect("/")
    else:
        # If the file doesn't exist, return a 404 error
        return jsonify({'error': '404 File not found'})

if __name__ == '__main__':
    app.run(debug=True)
