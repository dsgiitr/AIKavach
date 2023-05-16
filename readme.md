# Certify-Net

## Table of Contents

- [Introduction](#introduction)
- [Overview_and_Ideas](#overview_and_ideas)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Neural networks (NNs) have achieved great advances on a wide range of classification tasks, but have been shown vulnerable against adversarial examples(adversarial pertubations) and thus there is a line of work aiming to provide robustness certification for NNs.

This project aims to provide to provide certified robustness for large-scale datasets against adversarial pertubations using various defenses proposed to robustify deep learning models particularly randomized smoothing. 

## Overview_and_Ideas


This project incorporates various techniques such as DSRS (Double Sampling Randomized Smoothing), ISS (Input-Specific Sampling), and Denoised Smoothing to establish robust models and certified radii.

The goal of robustness certification is to determine a robust radius for a given input instance and model, which represents the maximum allowable perturbation without affecting the final prediction. Certification approaches tend to be conservative, offering a lower bound on the robust radius, while the actual maximum robust radius for a specific input instance may surpass the certified value.

Randomized smoothing has gained popularity as a technique for providing certified robustness in large-scale datasets. It involves sampling noise from a smoothing distribution to construct a smoothed classifier, thereby certifying the robust radius for the smoothed classifier. Compared to other techniques, randomized smoothing is efficient, model-agnostic, and applicable to a wide range of machine learning models.

In the case of randomized smoothing (RS), the most widely used certification approach is known as Neyman-Pearson-based certification. It relies on the probability of the base model predicting each class under the input noise to compute the certification. However, this approach encounters difficulties in scaling to large datasets due to the "Curse of Dimensionality."

To overcome the limitations of Neyman-Pearson-based certification, we have employed DSRS, which samples the prediction statistics of the base model under two different distributions and utilizes the joint information for certification computation. By incorporating more information, this certification approach surpasses the barrier posed by Neyman-Pearson-based certification and provides a tighter (if not equal) certification than the widely used approach.


## Installation

To use My Awesome Project, follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/my-awesome-project.git`
2. Install dependencies: `npm install`
3. Configure the application by updating the `config.js` file with your settings.
4. Launch the application: `npm start`
5. Open your web browser and visit `http://localhost:3000`.

## Usage

Once the application is running, you can perform the following actions:

1. Sign up or log in to your account.
2. Create a new task by clicking on the "New Task" button.
3. Fill in the necessary details such as task name, description, and due date.
4. Save the task and it will appear in your task list.
5. You can edit, delete, or mark tasks as completed.
6. Collaborate with team members by sharing tasks and assigning them.
7. Receive notifications for approaching deadlines.
8. Generate reports to analyze your productivity and task completion rate.

## Contributing

Contributions are welcome! If you want to contribute to My Awesome Project, please follow these guidelines:

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
