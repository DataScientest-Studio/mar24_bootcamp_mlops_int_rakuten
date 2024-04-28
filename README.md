# TODO:
20 Minutes PP
Talk about project
talk about architecture 



GitHub Action Docker Build

~~docker airflow~~
docker Model
docker MLflow

Model:
- store text data in db + img name /url
- store images in E3 Bucket

API: 
- text data db model
- model crud functions
- include model crud into predict endpoint 


Docoumentation 


# Descriptions Organization 

# How to launch the app
```bash
docker-compose up
```

# What does it do?
This application starts three docker container.
1. PostgreSQL Database (raktuen_db)which stores the data for user, logs, model data
2. FastAPI provides the connections to db and functionality predicting category 
3. Airflow (exclude?)


# What is in the .github



Rakuten Product Category Classifier
==============================

<<<<<<< HEAD
Project Overview
--------------

This project provides a product category/type classifier of products from the French e-commerce provider Rakuten. The model can predict 27 different categories based on text and image input.

Installation and Setup
-------------
To be filled:
- data acquisition
- training the model
- starting the model
- explain the pipeline
- ...

=======
>>>>>>> 6e297ec (update .gitignore+ readme)

Project Organization
------------
    ├── backend-app                # FastAPI application 
    │   ├── app                    # Fast API logic folder
    │   │   ├── api                # API endpoint implementations
    │   │   │   ├── auth.py        # Authentication related code
    │   │   │   ├── image_759577_product_120185380.jpg  # Image file for testing #! WRONG possition 
    │   │   │   ├── __init__.py
    │   │   │   ├── predict_category.py  # Prediction category endpoint
    │   │   │   ├── test_api.py    # API testing module #! WRONG position 
    │   │   │   └── users.py       # User endpoint implementation
    │   │   ├── config             # Configuration files
    │   │   │   ├── config.py      # Configuration settings
    │   │   │   └── __init__.py
    │   │   ├── core               # Core functionality
    │   │   │   ├── __init__.py
    │   │   │   ├── logger.py      # Logging utility
    │   │   │   ├── security.py    # Security-related functions
    │   │   │   └── settings.py    # Application settings
    │   │   ├── __init__.py
    │   │   ├── main.py            # Main application entry point
    │   │   ├── models             # Data models
    │   │   │   ├── database.py    # Database related functions
    │   │   │   ├── __init__.py
    │   │   │   ├── token.py       # Token management functions
    │   │   │   └── user.py        # User data model
    │   │   ├── sql_db             # SQL database module
    │   │   │   ├── crud.py        # CRUD operations for SQL database
    │   │   │   ├── database.py    # Database connection setup
    │   │   │   └── __init__.py
    │   │   └── tf_models          # TensorFlow models
    │   ├── Dockerfile             # Dockerfile for the FastAPI
    │   ├── requirements.txt       # Requirements file for the backend application
    │   ├── setup_prdcat_tabel.py  # Script for setting up product category table
    │   └── tests                  # Test files
    │       ├── api_test           # API test files
    │       ├── __init__.py
    │       └── unit               # Unit test files
    │           ├── __init__.py
    │           └── test_crud.py   # CRUD test module
    ├── docker-compose.yml         # Docker Compose configuration file
    ├── LICENSE                     # License file
    ├── models                      # Models folder (possibly for trained ML models)
    ├── notebooks                   # Notebooks folder (possibly for Jupyter notebooks)
    ├── pgdata                      # PostgreSQL data folder (error opening directory)
    ├── README.md                   # README file
    ├── references                  # References folder
    ├── reports                     # Reports folder
    │   └── figures                 # Figures folder within reports
    ├── requirements.txt            # Top-level requirements file
    └── src                         # Source code folder
        ├── config                  # Configuration files
        ├── data                    # Data processing scripts
        │   ├── __init__.py
        │   └── make_dataset.py     # Script for making datasets
        ├── features                # Feature engineering scripts
        │   ├── build_features.py   # Script for building features
        │   └── __init__.py
        ├── __init__.py
        ├── models                  # Machine learning models
        │   ├── __init__.py
        │   ├── predict_model.py    # Prediction model script
        │   └── train_model.py      # Training model script
        └── visualization           # Visualization scripts
            ├── __init__.py
            └── visualize.py        # Visualization script

<<<<<<< HEAD
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

Results and Evaluation
---------------
To be filled:
- performance of the model
- ...

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
=======
>>>>>>> 6e297ec (update .gitignore+ readme)
