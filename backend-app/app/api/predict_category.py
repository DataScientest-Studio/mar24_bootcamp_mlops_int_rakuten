from fastapi import APIRouter, UploadFile, HTTPException, Depends

# TODO: new import sort and verify
import os
import numpy as np
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from app.config import config 
from app.sql_db.crud import get_product_category_by_label, get_db
from PIL import Image
import io
from datetime import datetime, timedelta
from app.core.aws import download_from_aws
from app.core.utils import extract_tarfile
from dotenv import load_dotenv


load_dotenv()

# FUSION_MODEL = config.FUSION_MODEL

FUSION_MODEL = os.getenv('FUSION_MODEL')
MODEL_FOLDER_PATH = os.getenv('MODEL_FOLDER_PATH')
MODEL_FOLDER_NAME = os.getenv('MODEL_FOLDER_NAME')
MODEL_BUCKET = os.getenv('MODEL_BUCKET')
MODEL_FOLDER_METADATA_FILE = os.getenv('MODEL_FOLDER_METADATA_FILE')
TAR_PATH = os.getenv('TAR_PATH')
S3_FILE = os.getenv('S3_FILE') # WARNING: May rename to TAR_FILE
TOKENIZER_CONFIG_FILE_PATH = os.getenv('TOKENIZER_CONFIG_FILE_PATH')
max_sequence_length = 10


router = APIRouter(
    tags=['prediction']
)
# OPEN: 
# TODO:  predict endpoint
# TODO: save to db endpoint
# TODO: sql db for label to code to category translation

# def check_model_validity(model_metadata_file, validi_period=7):    
#     try:
#         with open(model_metadata_file, 'r') as file:
#             last_download = file.read().strip()
#
#         last_download_date = datetime.strftime(last_download, '%Y-%m-%d')
#         seven_days_ago = datetime.now().date() - timedelta(days=validi_period)
#
#         if last_download_date < seven_days_ago:
#             download_from_aws(S3_FILE, MODEL_BUCKET, MODEL_FOLDER)
#             extract_tarfile(TAR_PATH)
#             with open(model_metadata_file, 'w') as file:
#                     file.write(datetime.strftime(datetime.now().date(), '%Y-%m-%d'))
#         else:
#             pass
#     except FileNotFoundError:
#             download_from_aws(S3_FILE, MODEL_BUCKET, MODEL_FOLDER)
#             extract_tarfile(TAR_PATH)
#             with open(model_metadata_file, 'w') as file:
#                     file.write(datetime.strftime(datetime.now().date(), '%Y-%m-%d'))
def check_model_validity(model_folder_metadata_file, s3_file, model_bucket, model_folder_path, valid_period=7):
    metadata_file_path = os.path.join(model_folder_path, model_folder_metadata_file)
    try:
        with open(metadata_file_path, 'r') as file:
            last_download_str = file.read().strip()

        last_download_date = datetime.strptime(last_download_str, '%Y-%m-%d').date()
        seven_days_ago = datetime.now().date() - timedelta(days=valid_period)

        if last_download_date < seven_days_ago:
            download_from_aws(s3_file, model_bucket, model_folder_path)
            tar_file_path = os.path.join(model_folder_path, s3_file) 
            extract_tarfile(tar_file_path, model_folder_path)
            with open(metadata_file_path, 'w') as file:
                file.write(datetime.strftime(datetime.now().date(), '%Y-%m-%d'))
        else:
            True
    except FileNotFoundError:
        download_from_aws(s3_file, model_bucket, model_folder_path)
        tar_file_path = os.path.join(model_folder_path, s3_file) 
        extract_tarfile(tar_file_path, model_folder_path)
        with open(metadata_file_path, 'w') as file:
            file.write(datetime.strftime(datetime.now().date(), '%Y-%m-%d'))

def fusion_prediction(input_text, input_img):
    fusion_path = os.path.join(os.getcwd(), MODEL_FOLDER_PATH, MODEL_FOLDER_NAME, FUSION_MODEL)
    fusion_model = load_model(fusion_path)
    prediction = fusion_model((input_text, input_img))
    return prediction 

def tokenize_text(tokenizer_config_file_name, text:str):
    with open(tokenizer_config_file_name, 'r', encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_config)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, max_sequence_length,
            padding="post",
            truncating="post")
    return padded_sequence

def tokenize_texts(tokenizer_config_file_name, texts:list):
    with open(tokenizer_config_file_name, 'r', encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_config)
    sequence = tokenizer.texts_to_sequences(texts)
    padded_sequence = pad_sequences(sequence, max_sequence_length,
            padding="post",
            truncating="post")
    return padded_sequence

def preprocess_text(designation, description):
    #TODO: ADD CORRECT PREPROCESSING
    input_text = designation + ' ' + description
    # input_text = np.expand_dims(np.array([input_text]),axis=0)
    # input_text = input_text.astype(np.object_)
    return input_text

def preprocess_img(image):
    # Use Pillow to open the image
    image = Image.open(io.BytesIO(image))
    image_resized = image.resize((250, 250))
    input_image = np.expand_dims(np.array(image_resized), axis=0)
    input_image = input_image/255
    input_image = input_image.astype(np.float32)
    return input_image

@router.post('/predict_category/')
async def predict_category(image: UploadFile, designation: str, description: str, db:Session = Depends(get_db)):
    check_model_validity(MODEL_FOLDER_METADATA_FILE, S3_FILE, MODEL_BUCKET, MODEL_FOLDER_PATH)
    input_text = preprocess_text(designation, description)
    padded_sequence = tokenize_text(TOKENIZER_CONFIG_FILE_PATH, input_text)
    image = await image.read()
    input_img = preprocess_img(image)

    prediction = fusion_prediction(padded_sequence, input_img)
    max_idx = np.argmax(prediction)
    max_val = prediction[0,max_idx]
    cat = get_product_category_by_label(db, int(max_idx))

    return {'cat':cat.category, 'prob':str(np.array(max_val))}

