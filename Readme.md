# WORKSPACE

Carried our employee retention analysis by applying non-parametric machine learning algorithms with key performance indicators for employee satisfaction.

#

![](assets/images/workspace.png)

#

## Project Description

The workspace project is a comprehensive machine learning pipeline that leverages advanced non-parametric algorithms to analyze employee retention data. This project is designed to provide insights into key performance indicators, such as job satisfaction, salary, and tenure, and to help organizations identify areas of improvement in their retention strategies.

The pipeline is built using a combination of supervised and unsupervised learning techniques, including decision trees, k-nearest neighbors, and random forests. The dataset used for this project is located in the src/data directory and is called employee_retention.csv. The models are evaluated using various performance metrics, such as precision, recall, and F1 score.

The project includes several data preprocessing techniques, including feature scaling, one-hot encoding, and missing data imputation, to ensure that the data is properly formatted for analysis. Additionally, the pipeline incorporates several data visualization techniques, such as heatmaps, to identify patterns and correlations in the data.

This project is designed to be highly scalable and adaptable, allowing HR professionals and data scientists to easily extend and customize the pipeline to suit their specific needs. Overall, the workspace project represents a powerful tool for organizations looking to leverage machine learning to improve their employee retention strategies.

#

## Project Set Up

To set up this project, follow the steps below:

1. Clone this repository to your local machine using the following command:
   ```bash
   $ git clone git@github.com:shahkv95/workspace.git
   $ cd workspace
   ```
2. This project uses `pipenv` to manage the dependencies. Follow the below step if you don't have pipenv already installed in your system

   ```bash
   $ pip install pipenv install
   ```

3. Setup and activate the virtual environment
   ```bash
   $ PIPENV_VENV_IN_PROJECT=true pipenv shell
   ```
4. Install all the dependencies.

   ```bash
   $ pipenv install --dev
   ```

   or alternatively, you can also run the following command

   ```bash
   $ make setup
   ```

5. Run the application

   ```bash
   $ make run
   ```

6. To run all the test cases, run the following command

   ```bash
   $ make test
   ```

7. To cleanup the environment, run the following command
   ```bash
   $ make clean
   ```

#

## Troubleshooting

If you encounter any issues while using the workspace project, try the following troubleshooting steps:

1. Check your Python version: Make sure that you are using a compatible version of Python (e.g., Python 3.7 or higher). You can check your Python version by running 
   ```bash
   $ python3 --version
   ```

2. Check your dependencies: Make sure that you have installed all the required dependencies listed in the Pipfile. You can install them by running
   ```bash
   $ pipenv install --dev
   ```

3. Check your data: Make sure that your data is properly formatted and free of errors. Double-check that all the required columns are present and that the data is in the correct format.

If you are still experiencing issues after trying these troubleshooting steps, please feel free to open an issue on the project's GitHub repository or contact the project maintainers for assistance.

#

## Issue Template

### Description

[Describe the issue in detail]

### Steps to reproduce

[Provide a step-by-step description of how to reproduce the issue, including any relevant code]

### Expected behavior

[Describe what you expected to happen]

### Actual behavior

[Describe what actually happened]

### Environment

[List any relevant details about your environment, such as operating system, browser, or device]

### Additional information

[Provide any additional information that might be helpful, such as screenshots or error messages]

### Rules for Opening an Issue

- Check if the issue has already been reported: Before opening a new issue, search through the existing issues to see if someone has already reported the same problem. If so, add any additional information or comments to that issue instead of creating a new one.

- Be specific: Provide as much detail as possible about the issue, including steps to reproduce it and any relevant code or error messages. The more information you provide, the easier it will be for someone to diagnose and fix the issue.

- Keep it on-topic: Stick to the topic of the issue and avoid introducing unrelated issues or feature requests. If you have a new feature request or idea, open a separate issue for it.
