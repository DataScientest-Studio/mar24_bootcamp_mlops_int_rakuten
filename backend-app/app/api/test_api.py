from fastapi.testclient import TestClient
from predict_category import router
from unittest.mock import patch

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









