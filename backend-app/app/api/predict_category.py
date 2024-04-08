from typing import Annotated
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
from app.models.user import User
from app.api.auth import get_current_active_user

# TODO: new import sort and verify
import os
import numpy as np
from tensorflow.keras.models import load_model 
from app.config import config 
from PIL import Image
import io

FUSION_MODEL = config.FUSION_MODEL

router = APIRouter(
    tags=['prediction']
)
# OPEN: 
# TODO:  predict endpoint
# TODO: save to db endpoint
# TODO: sql db for label to code to category translation

def fusion_prediction(input_text, input_img):
    fusion_path = os.path.join(os.getcwd(),'app', 'tf_models', FUSION_MODEL)
    fusion_model = load_model(fusion_path)
    prediction = fusion_model((input_text, input_img))
    return prediction 

def preprocess_text(designation, description):
    input_text = designation + ' ' + description
    input_text = np.expand_dims(np.array([input_text]),axis=0)
    input_text = input_text.astype(np.object_)
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
async def predict_category(image: UploadFile, designation: str, description: str):
    input_text = preprocess_text(designation, description)
    image = await image.read()
    input_img = preprocess_img(image)
    prediction = fusion_prediction(input_text, input_img)
    max_idx = np.argmax(prediction)
    max_val = prediction[0,max_idx]
    return str(max_val)


# @router.post'/fileupload/', response_model=User
# async def create_upload_filefile: UploadFile, current_user: Annotated[User,  Depends(get_current_active_user)]:
#     # TODO: use CSV, 
#     # TODO: catch wrong content 
#     # 
#     try:
#         if file.filename.split('.')[-1] == 'xlsx': # WARNING: create check_file_type functin?
#             df = pd.read_excel(file.file)
#             df = pd.melt(df, id_vars=['name', 'email', 'id'], var_name='date', value_name='value')
#             return JSONResponse(df.head().to_json())
#         else:
#             raise HTTPException(status_code=422, detail='File needs to have .xlsx format.')
#     except Exception as e:
#         raise e
