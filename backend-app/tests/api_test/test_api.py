from fastapi.testclient import TestClient
from app.api.predict_category import router
from unittest.mock import patch
import tensorflow as tf
from app.models.database import Product_Category

# TODO: replace with .env?
img_name_test = 'backend-app/tests/api_test/image_759577_product_120185380.jpg'
test_prediction = tf.constant([[0.02415627, 0.03744585, 0.06960037, 0.02850509, 0.01205792, 0.01704547,
                                  0.03257139, 0.04715945, 0.07248406, 0.07602803, 0.03835761, 0.05048478,
                                  0.01647786, 0.03617582, 0.01722865, 0.11012374, 0.0331501,  0.04468789,
                                  0.0568036,  0.03034873, 0.02791215, 0.03209974, 0.016919,   0.01985059,
                                  0.02659598, 0.01565288, 0.0100769]])


client = TestClient(router)




@patch("app.api.predict_category.get_product_category_by_label", return_value = Product_Category(category = 'Category 2'), autospec = True)
@patch("app.api.predict_category.check_model_validity", return_value = True)
@patch("app.api.predict_category.tokenize_text", return_value = [0,0,0,0,73,2,3,5,9,1])
@patch("app.api.predict_category.fusion_prediction", return_value = test_prediction)
def test_api_predict(mock_get_product_category_by_label, mock_check_model_validity, moch_tokenize_text, mock_fusion_prediction):
    
    answer = client.post('/predict_category/?designation=test&description=test', files = {'image':  open(img_name_test, 'rb')})
    
    
    #check for correct status code
    assert answer.status_code == 200

    #check that we got correct category
    assert answer.json()['cat'] ==  'Category 2' #'Second-hand Newspapers and Magazines'
    #check that answer is a valid probability
    prob = float(answer.json()['prob'])
    assert (0 <= prob <= 1) == True









