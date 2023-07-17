import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)
            
            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Gender:str,
                 Age:str,
                 Occupation:int,
                 City_Category:str,
                 Stay_In_Current_City_Years:str,
                 Marital_Status:int,
                 Product_Category_1:int,
                 Product_Category_2:float,
                 Product_Category_3:float):
        
        self.Gender=Gender
        self.Age=Age
        self.Occupation=Occupation
        self.City_Category=City_Category
        self.Stay_In_Current_City_Years=Stay_In_Current_City_Years
        self.Marital_Status=Marital_Status
        self.Product_Category_1 = Product_Category_1
        self.Product_Category_2 = Product_Category_2
        self.Product_Category_3 = Product_Category_3

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Gender':[self.Gender],
                'Age':[self.Age],
                'Occupation':[self.Occupation],
                'City_Category':[self.City_Category],
                'Stay_In_Current_City_Years':[self.Stay_In_Current_City_Years],
                'Marital_Status':[self.Marital_Status],
                'Product_Category_1':[self.Product_Category_1],
                'Product_Category_2':[self.Product_Category_2],
                'Product_Category_3':[self.Product_Category_3]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)


