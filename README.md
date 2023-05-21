# Certify-Net

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
 - [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Neural networks (NNs) have achieved great advances in a wide range of classification tasks. However, it is widely acknowledged that even subtle modifications to the inputs of image classification systems can lead to incorrect predictions. Take, for example, a model trained to classify images of a deer. The model easily classifies an image of the animal grazing in a grassy field. Now, if just a few pixels in that image are maliciously altered, you can get a very different and wrong prediction despite the image appearing unchanged to the human eye. Sensitivity to such input perturbations, which are known as adversarial examples, raises security and reliability issues for the vision-based systems that we deploy in the real world. To tackle this challenge, our project aims to provide certified robustness for large-scale datasets against adversarial perturbations using various defenses proposed to robustify deep learning models, particularly randomized smoothing.


 ## Motivation


This project incorporates techniques such as DSRS (Double Sampling Randomized Smoothing), ISS (Input-Specific Sampling), and Denoised Smoothing to establish robust models and certified radii.

Robustness certification aims to determine a robust radius for a given input instance and model, representing the maximum allowable perturbation without affecting the final prediction. Certification approaches tend to be conservative, offering a lower bound on the robust radius, while the actual maximum robust radius for a specific input instance may surpass the certified value.

Randomized smoothing has gained popularity as a technique for providing certified robustness in large-scale datasets. Essentially, randomized smoothing smooths out a classifier: it takes a “brittle” or “jittery” function and makes it more stable, helping to ensure predictions for inputs in the neighborhood of a specific data point are constant. It involves sampling noise from a smoothing distribution to construct a smoothed classifier, thereby certifying the robust radius for the smoothed classifier. Compared to other techniques, randomized smoothing is efficient, model-agnostic, and applicable to a wide range of machine-learning models.

In the case of randomized smoothing (RS), the most widely used certification approach is known as Neyman-Pearson-based certification. It relies on the probability of the base model predicting each class under the input noise to compute the certification. However, this approach encounters difficulties scaling to large datasets due to the "Curse of Dimensionality."

To overcome the limitations of Neyman-Pearson-based certification, we have employed DSRS, which samples the prediction statistics of the base model under two different distributions and utilizes the joint information for certification computation. By incorporating more information, this certification approach surpasses the barrier posed by Neyman-Pearson-based certification and provides a tighter (if not equal) certification than the widely used approach.

![Figure Demonstrating DSRS and NP approach](readme_images/overall_pipeline.png)

Here,

![Figure Demonstrating DSRS and NP approach](readme_images/equation.png)


<!-- ```math
P_A = f^{P}(x_0)_{y_0} = \mathbf{Pr}_{{\varepsilon }\sim P}\, [F(x_0+\varepsilon )=y_0]\\
Q_A = f^{Q}(x_0)_{y_0} = \mathbf{Pr}_{{\varepsilon }\sim Q}\, [F(x_0+\varepsilon )=y_0]  -->


 More details can be found in [this](https://arxiv.org/abs/2206.07912) paper.

 <!-- Now,can we generate a provably robust classifier from off-the-shelf pretrained classifiers without retraining them specifically for robustness? No, hence we use denoised smoothing. Via the simple addition of a pretrained denoiser, we can apply randomized smoothing to make existing pretrained classifiers provably robust against adversarial examples without custom training. -->

In practice, it is hard to calculate P<sub>A</sub> and Q<sub>A</sub>, so it’s estimated using Monte-Carlo sampling, which gives a confidence interval of P<sub>A</sub> and Q<sub>A</sub>.For a base classifier ___F___
, one can apply the above procedure to get a prediction of any data point along with a robustness guarantee in the form of a certified radius, the radius around a given input for which the prediction is guaranteed to be fixed. But the above procedure assumes that the base classifier  ___F___ classifies well under Gaussian perturbations of its inputs. Now what if the base classifier  ___F___ is some off-the-shelf classifier that wasn’t trained specifically for randomized smoothing—that is, it doesn’t classify well under noisy perturbations of its inputs. With denoised smoothing, we make randomized smoothing effective for classifiers that aren’t trained specifically for randomized smoothing. The method is straightforward; as mentioned above, instead of applying randomized smoothing to these classifiers, we prepend a custom-trained denoiser in front of these classifiers and then apply randomized smoothing. The denoiser helps by removing noise from the noisy synthetic copies of the input, which allows the pre-trained classifiers to give better predictions. In this project, we used **DRUNET** as a denoiser for denoised smoothing.


![DRUNET Architecture](readme_images/denoiser_arch.png)

DruNet Architecture [paper](https://arxiv.org/pdf/2008.13751.pdf)

 Now the focus is on improving the computational efficiency of certified robustness in randomized smoothing. While randomized smoothing has shown promising results in terms of robustness, the certification process can be computationally demanding, making it less practical for real-world applications.

 One of the main challenges lies in the estimation of confidence intervals, which heavily relies on large sample approximations. Existing methods typically use an input-agnostic sampling (IAS) scheme, where the sample size for the confidence interval is fixed regardless of the input. However, this approach may result in a suboptimal trade-off between the average certified radius (ACR) and runtime.

 To address this issue, we used an approach called Input-Specific Sampling (ISS) acceleration, which aims to achieve cost-effective robustness certification by adaptively reducing the sampling size based on the input characteristics. By doing so, the proposed method improves the efficiency of the certification process while still maintaining control over the decline in the certified radius resulting from the reduced sample size.

 The experiments from [this](https://arxiv.org/abs/2112.12084) paper demonstrate that ISS can speed up the certification process by more than three times with a limited cost of reducing the certified radius by 0.05. Additionally, ISS outperforms the input-agnostic sampling (IAS) scheme in terms of the average certified radius across various hyperparameter settings.


 ## File Structure



 ### A typical top-level directory layout

 ```
 .
├── LICENSE
├── Makefile                     <- Makefile with commands like `make data` or `make train`
├── README.md                    <- The top-level README for developers using this project
├── readme_images                <- Directory for readme-related images
├── data
│   ├── external                 <- Data from third party sources
│   ├── interim                  <- Intermediate data that has been transformed
│   ├── processed                <- The final, canonical data sets for modeling
│   └── raw                      <- The original, immutable data dump
├── docs                         <- A default Sphinx project; see sphinx-doc.org for details
├── models                       <- Trained and serialized models, model predictions, or model summaries
├── denoiser
│   ├── data                     <- Intermediate data that has been transformed
│   ├── train_denoiser.py        <- Python script for training the denoiser
│   ├── denoising
│   │   └── drunet               <- The final, canonical data sets for modeling
│   ├── utilities                <- Utility scripts for the denoiser
│   │   ├── extra                <- Extra utilities
│   │   └── utils_image.py       <- Utility functions for image processing
│   ├── models                   <- Contains models of the denoiser
│   │   ├── model_base.py        <- Base model class
│   │   ├── model_plain.py       <- Plain model class
│   │   ├── network_unet.py      <- U-Net network architecture
│   │   └── select_network.py    <- Script for selecting the network
│   └── options                  <- Default JSON file for training
├── references                   <- Data dictionaries, manuals, and all other explanatory materials
├── web                          <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── app.py                   <- Python script for the web application
│   └── flask_try.py             <- Generated graphics and figures to be used in reporting
├── requirements.txt             <- The requirements file for reproducing the analysis environment
├── setup.py                     <- Make this project pip installable with `pip install -e`
├── src                          <- Source code for use in this project
│   ├── main.py                  <- Python module
│   ├── main.sh                  <- Shell script
│   ├── data                     <- Scripts to download or generate data
│   │   └── make_dataset.py      <- Script to make dataset
│   ├── features                 <- Scripts to turn raw data into features for modeling
│   │   ├── algo.py              <- Intermediate data that has been transformed
│   │   ├── distribution.py      <- Intermediate data that has been transformed
│   │   └── smooth.py            <- Script for smoothing data
│   ├── models                   <- Scripts to train models and make predictions
│   │   ├── train.py             <- Script for model training
│   │   └── sample.py            <- Script for model prediction sampling
│   └── utilities                <- Scripts for exploratory and results-oriented visualizations
│       └── utils.py             <- Utility functions
└── tox.ini                      <- tox file with settings for running tox
```


 <!-- - [data](aisec/data)
      - [dataset_fdcnn.py](aisec/data/dataset_fdncnn.py)
      - [utils_image.py](aisec/data/utils_image.py)
 - [denoising](aisec/denoising)
 - [drunet](aisec/drunet/options)
 - [extra](aisec/extra)
 - [final_model_weights](aisec/final_model_weights)
 - [models](aisec/models)
 - [options](aisec/options)
 - [templates](aisec/templates)
 - [trainsets](aisec/trainsets/trainH)
 - [algo.py](algo.py)
 - [app.py](app.py)
 - [distribution.py](distribution.py)
 - [packaging_class.py](packaging_class.py)
 - [sample.py](sample.py)
 - [smooth.py](smooth.py)
 - [train_denoiser.py](train_denoiser.py)
 - [train.py](train.py)
 - [utils_image.py](utils_image.py)
 - [utils.py](utils.py) -->

## Installation

To use this Project, follow these steps:

1. Clone the repository: `https://github.com/dsgiitr/aisec.git`
2. Install dependencies: `npm install`
3. Configure the application by updating the `config.js` file with your settings.
4. Launch the application: `npm start`
5. Open your web browser and visit `http://localhost:3000`.

## Usage

##### Preparation

1. Install recommended environment: python 3.9.7, scikit-learn >= 0.24, torch 1.10 with GPU (for fast sampling of DNN models).

2. If you need to reproduce the results from certifying pretrained models (see *Guidelines for Reproducing Experimental Results > Scenario B*), download the trained models from Figshare (https://figshare.com/articles/software/Pretrained_Models_for_DSRS/21313548), and unzip to `models/` folder. After unzip, the `models` folder should contain 5 subfolder: `models/cifar10` (smoothadv's pretrained best CIFAR-10 models), `models/consistency` (reproduced consistency models), `models/new_cohen` (reproduced Cohen's Gaussian augmentation models), `models/salman` (reproduced Salman et al's models), and `models/smoothmix` (reproduced SmoothMix models).

##### Running Certification from Scratch

The main entrance of our code is `main.py`, which computes both the Neyman-Pearson-based certification and DSRS certification given the sampled probability folder path.

Given a input test dataset and a model, to compute the DSRS certification, we need the following **three steps**:

1. **Sampling & get the probability (P_A and Q_A) under two distributions**

`sampler.py` loads the model and does model inference via PyTorch APIs. It will output pA or qA to corresponding txt file in `data/sampling/{model_filename.pth (.tar extension name is trimmed)}/` folder (will create the folder). Note that each run only samples pA from one distribution, so:

- If the end-goal is to compute only the Neyman-Pearson-based certification, we only need one probability (P_A), and thus we run sampler.py just once for the model.

- If the end-goal is to compute the DSRS certification, we need two probabilties from two different distributions (P_A and Q_A), and thus we run sampler.py twice for the model.

Main usage:

`python sampler.py [dataset: mnist/cifar10/imagenet/tinyiamgenet] [model: models/*/*.pth.tar] [sigma] --disttype [gaussian/general-gaussian] {--k [k]} {--th [number between 0 and 1 or "x+" for adaptive thresholding]} --N [sampling number, usually 50000 for DSRS, 100000 for Neyman-Pearson] --alpha [confidence, usually 0.0005 for DSRS, 0.001 for Neyman-Pearson]`


There are other options such as batch size, data output directory, GPU no. specification, etc. Please browse `parser.add_arguments()` statements to get familier with them.

- If the distribution is Gaussian, we don't need to specify $k$.
- If the distribution is generalized Gaussian, we need to specify $k$, whose meaning can be found in the paper.
- If the distribution is Gaussian or generalized Gaussian with thresholding, `--th` specifies the threshold. 
  - If the threshold is a static value, `th` is a real number meaning the percentile. 
  - If the threshold is depended by pA, `th` is "x+" (there are other heuristics but they do not work well), and the script will search the pA file to determine the threshold dynamically.
    

2. **Compute the Neyman-Pearson-based certification**

`main.py` is the entrance for computing both the Neyman-Pearson-based certification and the DSRS certification. Note compute the DSRS certification, one first needs to execute this step, i.e., compute the Neyman-Pearson-based certification, since DSRS tries to increase the radius certified strating from the certified radius of Neyman-Pearson-based certification.

`main.py` is built solely on CPU, mainly relying on scikit-learn package.

Main usage:

`python main.py [dataset: mnist/cifar10/imagenet/tinyimagenet] origin [model: *.pth - will read from data/sampling/model, just type in "*.pth" as the name without relative path] [disttype = gaussian/general-gaussian] [sigma] [N: sampling number, used to index the sampling txt file] [alpha: confidence, used to index the sampling file] {--k [k]}`

There are other options that can customized the folder path of sampling data or the parallelized CPU processes. But the default one is already good. Note that some arguments in `main.py` only have effects when computing DSRS certification, such as `-b`, `--improve_*`, `--new_rad_dir`.

If the distribution type is generalized Gaussian, we need to specify k, otherwise not.

For standard Gaussian, we use the closed-form expression in Cohen et al to compute the certification. 
For generalized Gaussian, we use the numerical integration method in Yang et al to compute the certification.

3. **Compute the DSRS certification**

Once the Neyman-Pearson-based certification is computed, we run `main.py` again but use different arguments to compute the DSRS certification. 

Main usage:

`python main.py [dataset: mnist/cifar10/imagenet/tinyimagenet] improved [model: *.pth - will read from data/sampling/model, just type in "*.pth" as the name without relative path] [disttype = gaussian/general-gaussian/gaussian-th/general-gaussian-th] [sigma] [N: sampling number, used to index the sampling txt file] [alpha: confidence, used to index the sampling file] {--k [k]} {-b b1 b2 ...} {--improve_mode grid/fast/precise} {--improve_unit real_number} {--improve_eps real_number}`

Note that the arguments are different from step 2, where `origin` changed to `improved`. The script will read in the previous Neyman-Pearson-based certification files, and compute the improved certification.

Distribution P's parameters are specified by `disttype` and `k`. Specifically, if `disttype` is `gaussian-th` or `general-gaussian-th`, the P distribution is Gaussian or generalized Gaussian respectively, and the Q distribution is thresholded Gaussian or thresholded generalized Gaussian respectively.

Distribution Q is of the same `disttype` and has the same `k` as P. The difference is in variance (if `disttype` is `gaussian` or `general-gaussian`) or the threshold (if `disttype` is `gaussian-th` or `general-gaussian-th`). The variance or the threshold (real number if static percentile threshold, `x+` if dynamic heuristic based threshold) is specified by `b1`, `b2`, ..., where each `bi` stands for one option of Q distribution, i.e., the script supports computing the certification with one P and multiple different Q's in a single run.

`--improve_*` arguments specify the way we try a new robust radius to certify. The most precise way is to conduct binary search as listed in Algorithm 2 in the paper, but for efficiency we can also use `grid` mode as `improve_mode` which iteratively enlarges the radius by `imrpove_unit` and tries to certify.

As mentioned in Appendix E.3, among these three steps, the most time-consuming step is Step 1 on typical image classification datasets.

##### Result Summarization and Plotting

We provide the script `dump_main_result.py` to summarize main experimental data in our paper.

Usage: `python dump_main_result.py`
It will create `result/` folder and dump all main tables and figures there. Some critical results are also printed in stdout.

##### Appendix: Training Scripts

The repo also contains code that trains the model suitable for DSRS certification (as discussed in Appendix I).

`train.py` code is adapted from Consistency training code in https://github.com/jh-jeong/smoothing-consistency.

`train_smoothmix.py` code is adapted from SmoothMix training code in https://github.com/jh-jeong/smoothmix.

For MNIST and CIFAR-10, we train from scratch. For ImageNet, we finetune from pretrained ImageNet models for a few epochs.
Whent the training is finished, we need to copy the `*.pth.tar` model to `models/` folder.

- Gaussian augmentation training:
  - MNIST: `python train.py mnist mnist_43 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00/... --num-noise-vec 1 --lbd 0 --k 380 --k-warmup 100`
  - CIFAR-10: `python train.py cifar10 cifar_resnet110 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00/... --num-noise-vec 1 --lbd 0 --k 1530 --k-warmup 100`
  - ImageNet: `python train.py imagenet resnet50 --lr 0.001 --lr_step_size 1 --epochs 6  --noise 0.25/0.50/1.00 --num-noise-vec 1 --lbd 0 --k 75260 --k-warmup 60000 --batch 96 --pretrained-model ../../pretrain_models/cohen_models/models/imagenet/resnet50/noise_[0.25/0.50/1.00]/checkpoint.pth.tar`
    
    Note: the pretrained models are from Cohen et al's randomized smoothing [repo](https://github.com/locuslab/smoothing), and the direct link is here: https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view.

- Consistency training:
  - MNIST: `python train.py mnist mnist_43 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00 --num-noise-vec 2 --lbd 5 --k 380 --k-warmup 100`
  - CIFAR-10: `python train.py cifar10 cifar_resnet110 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.25/0.50/1.00 --num-noise-vec 2 --lbd 20 --k 1530 --k-warmup 100`
  - ImageNet: `python train.py imagenet resnet50 --lr 0.001 --lr_step_size 1 --epochs 6  --noise 0.25/0.50/1.00 --num-noise-vec 2 --lbd 5 --k 75260 --k-warmup 60000 --batch 96 --pretrained-model ../../pretrain_models/cohen_models/models/imagenet/resnet50/noise_[0.25/0.50/1.00]/checkpoint.pth.tar`
  
- SmoothMix training:
  - MNIST: `python train_smoothmix.py mnist mnist_43 --lr 0.01 --lr_step_size 30 --epochs 90  --noise_sd 0.25/0.50/1.00 --eta 5.00 --num-noise-vec 4 --num-steps 8 --mix_step 1 --k 380 --k-warmup 70`
  - CIFAR-10: `python train_smoothmix.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise_sd 0.5 --eta 5.00 --num-noise-vec 2 --num-steps 4 --alpha 1.0 --mix_step 1 --k 1530 --k-warmup 110`
  - ImageNet: `python train_smoothmix.py imagenet resnet50 --lr 0.01 --lr_step_size 30 --epochs 10  --noise_sd 0.5 --eta 1.00 --num-noise-vec 1 --num-steps 1 --alpha 8.0 --mix_step 0 --k 75260 --k-warmup 200000 --batch 48 --pretrained-model ../../pretrain_models/cohen_models/models/imagenet/resnet50/noise_0.50/checkpoint.pth.tar`


## Contributing

Contributions are welcome! If you want to contribute to this Project, please follow these guidelines:

1. Fork the repository and create your branch: `git checkout -b feature/my-new-feature`
2. Commit your changes: `git commit -am 'Add some feature'`
3. Push to the branch: `git push origin feature/my-new-feature`
4. Submit a pull request detailing your changes.

Please ensure your code follows the established coding conventions and includes appropriate tests.

## License

This project is licensed under the MIT License.

## Additional Sections

- **Deployment**: Visit our deployment guide for instructions on how to deploy the application to various platforms.
<!-- - **Documentation**: Access the full documentation [here](https://docs.myawesomeproject.com). -->
<!-- - **Changelog**: View the changelog to see the history of changes between versions. -->
- **FAQ**: Check out our FAQ section for answers to common questions.
