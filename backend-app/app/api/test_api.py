from fastapi.testclient import TestClient
from .predict_category import router


import json
from typing import Annotated
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.sql_db import crud 
#import app.models.database as models
import app.models.user as api_m 
from app.sql_db.crud import get_db, get_product_category_by_label
from app.sql_db.database import engine
from app.api.auth import get_current_active_user, get_current_active_admin

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.sql_db.database import Base

from unittest.mock import patch, Mock

from app.models.database import Product_Category



img_name_test = 'app/api/image_759577_product_120185380.jpg'

client = TestClient(router)




@patch("app.api.predict_category.get_product_category_by_label", return_value = Product_Category(category = 'Category 2'), autospec = True)
def test_api_predict(mock_get_product_category_by_label):


    # def mock_get_product_category_by_label(db, label):
    #     if label == 0:
    #         return MagicMock(category="Category 1")
    #     elif label == 1:
    #         return MagicMock(category="Category 2")
    #     else:
    #         return MagicMock(category="Unknown Category")
    # mock_get_product_category_by_label.side_effect = mock_get_product_category_by_label
    
    answer = client.post('/predict_category/?designation=test&description=test', files = {'image':  open(img_name_test, 'rb')})
    
    
    #check for correct status code
    assert answer.status_code == 200

    #check that we got a valid category
    
    
    #check that we got correct category
    assert answer.json()['cat'] ==  'Category 2' #'Second-hand Newspapers and Magazines'
    
    #check that answer is a valid probability
    prop = float(answer.json()['prop'])
    assert (0 <= prop <= 1) == True









