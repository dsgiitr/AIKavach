import os
import subprocess
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, render_template, request, send_file, redirect
import asyncio
import time
from packaging_class import *
from datetime import datetime
import torch
from train_denoiser import denoiser
from integrate_dsrs_denoiser import DenoisedModel
import utils_image as util
import numpy as np

app = Flask(__name__)

async def calculate_dsrs_radius(
  denoised_model,
  de, 
  form_used: dict,
  sigma= 0.5,
  Distributon_type="general-gaussian",
  training = "mnist",
  k =380,
  N=100000,
  in_nc = 3,
  Alpha = 0.05,
  batch = 400,
  img_size = (32,32)

):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if form_used["dataset"][0] == "mnist":
        d = 784
        k = 380
        num_classes = 10
        img_size = (28,28)
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
    if form_used["distribution"][0] == "general_gaussian":
        dist1 = "general-gaussian"
        dist2 = "general-gaussian"
    elif form_used["distribution"][0] == "standard_gaussian":
        dist1 = "gaussian"
        dist2 = "gaussian"
    
    final_model = DenoisedModel(de,denoised_model,img_size)

    secure_model = FinishedModel(final_model, d, k, num_classes, dist1,dist2, float(form_used['sigma'][0]),float(sigma_q), float(form_used['alpha'][0]), num_sampling_min = 100)
    noise_level_img = 15
    noise_level_model = 15
    y = torch.randn((32, 32, in_nc)).float()
    img_L = util.uint2single(y)
    img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
    img_L = util.single2tensor4(img_L)
    img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
    x = img_L.to(device)
    label = secure_model.label_inference_without_certification(x, int(form_used['N'][0]), 0.01, batch_size = int(form_used['batch_size'][0]))
    logits_old = secure_model.logits_inference_without_certification(x, int(form_used['N'][0]), 0.01, batch_size = int(form_used['batch_size'][0]))
    logits, r = secure_model.inference_and_certification(x,  int(form_used['N'][0]), 0.01, batch_size = int(form_used['batch_size'][0]))
    model_id = form_used["model_id"]
    final_path = f"final_model_weights/final_model_{model_id}.pth"
    torch.save(secure_model,final_path)
    return r

async def train_denoiser(
    epochs = 1,
    in_nc = 4,
    out_nc = 3,
    nc = [64,128,256,512],
    nb = 4,
    pth = None,
    model_path = None
):
    de=denoiser(in_nc = in_nc,out_nc=out_nc,nc = nc, nb=nb)
    if model_path:
        de.ld(model_path)
    de.train_drunet(epochs,pth)
    return de.drunet.netG

models = [{
    "name":"test1",
    "file_name":"test.pt",
    "timestamp": "",
    "results_calculated":False,
    "id":0,
    "certified_radius":0.01,
    "model_id":0,
    "timestamp": datetime.now()
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
        example_tested["timestamp"] = datetime.now()
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
        print("flag1")
        # TODO: Add option to train denoisers
        # p1 = subprocess.Popen(['python', 'train.py','--epochs', '1', '--in_nc','2','--out_nc','1','--model_name',f'{model_id}','--dataset','mnist/testSample'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print("flag")
        # out1, err1 = p1.communicate()
        # print(out1)
        # if err1:
        #     error = 'An error occurred while executing the scripts'
        #     return jsonify({'error': error})
        # print("flag2")
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()

        # For now images in trainsets/trainH will be used for training of the denoiser
        in_nc = 4
        out_nc = 3
        if model_dict["dataset"][0] == "mnist":
            in_nc = 2
            out_nc = 1
        de = loop.run_until_complete(train_denoiser(in_nc = in_nc,out_nc = out_nc))
        #Execute the second Bash script
        
        denoised_model = f"models/{model_dict['file_name']}"
        if not os.path.exists("final_model_weights/"):
            os.makedirs("final_model_weights/")
        final_model_path = f"final_model_weights/final_model_weight_{model_id}.pth"
         # Check for any errors
         # sample results 
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        r = loop.run_until_complete(calculate_dsrs_radius(denoised_model,de,model_dict,in_nc = out_nc))
        results = [
            {"confidence Radius": r},
        ]

        models[model_id]["results_calculated"] = True
        models[model_id]["certified_radius"] = r
        return jsonify(results[0])
    return redirect("/")


@app.route('/download_weights/<model_id>')
def download_weights(model_id):
    # Assuming the updated weights file is located in the "updated_weights" directory
    # with the filename format "model_id_weights_updated.h5"
    filepath = f"final_model_weights/final_model_{model_id}.pth"
   
    if os.path.isfile(filepath):
        # If the file exists, send it as a downloadable file
        return send_file(filepath, as_attachment=True)
    else:
        # If the file doesn't exist, return a 404 error
        return jsonify({'error': '404 File not found'})

if __name__ == '__main__':
    app.run(debug=True)
