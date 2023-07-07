from sklearn.impute import SimpleImputer # Handling Missing Values
from sklearn.preprocessing import StandardScaler # Handling Feature Scaling
from sklearn.preprocessing import OneHotEncoder # OneHot Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

# Custom Transformer that encodes Product_ID using Frequency Encoding
class CustomImputer(BaseEstimator, TransformerMixin,):
    def __init__(self):
        super().__init__()
        self.product_frequency_dict = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X,columns=['Product ID'])
        self.product_frequency_dict = df['Product ID'].value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X,columns=['Product_ID'])
        df['Product_ID']=df['Product_ID'].map(self.product_frequency_dict)
       
        return df.values

## Data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

## Data Ingestionconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):
         
         try:
            logging.info('Data Transformation initiated')

            ## Since our dataset consists of only categorical features, we are not taking numerical features preprccessing into consideration.

            col_to_frequency_encode=['Product_ID']
            cols_to_onehotencode=['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years', 'Marital_Status',
                                      'Product_Category_1','Product_Category_2', 'Product_Category_3']

            ## Categorigal Pipelines
            onehot_pipeline=Pipeline(
                steps=[
                ('onehotencoder',OneHotEncoder())
                ]
            )

            frequency_pipeline=Pipeline(
                steps=[
                ('customimputer',CustomImputer()),
                ('scaler',StandardScaler())
                ]
            )


            preprocessor=ColumnTransformer([
            ('onehot_pipeline',onehot_pipeline,cols_to_onehotencode),
            ('frequency_pipeline',frequency_pipeline,col_to_frequency_encode)
            ])
            

            logging.info('Data Transformation Completed')

            return preprocessor

            
         except Exception as e:
            
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
            

            target_column_name = 'Purchase'
            drop_columns = [target_column_name,'User_ID']

            if train_df.duplicated().sum()!=0:
                train_df=train_df.drop_duplicates(keep='first')



                    
            ## Dividing features into independent and dependent features
            ## Training data
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## Test data
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]


            input_feature_train_df['Occupation']=input_feature_train_df['Occupation'].astype('object')
            input_feature_train_df['Marital_Status']=input_feature_train_df['Marital_Status'].astype('object')
            input_feature_train_df['Product_Category_1']=input_feature_train_df['Product_Category_1'].astype('object')
            input_feature_train_df['Product_Category_2']=input_feature_train_df['Product_Category_2'].astype('object')
            input_feature_train_df['Product_Category_3']=input_feature_train_df['Product_Category_3'].astype('object')

            input_feature_test_df['Occupation']=input_feature_test_df['Occupation'].astype('object')
            input_feature_test_df['Marital_Status']=input_feature_test_df['Marital_Status'].astype('object')
            input_feature_test_df['Product_Category_1']=input_feature_test_df['Product_Category_1'].astype('object')
            input_feature_test_df['Product_Category_2']=input_feature_test_df['Product_Category_2'].astype('object')
            input_feature_test_df['Product_Category_3']=input_feature_test_df['Product_Category_3'].astype('object')

            
            input_feature_train_df['Product_Category_2']=input_feature_train_df['Product_Category_2'].fillna(input_feature_train_df['Product_Category_2'].mode().values[0])
            input_feature_train_df['Product_Category_3']=input_feature_train_df['Product_Category_3'].fillna(input_feature_train_df['Product_Category_3'].mode().values[0])  

            input_feature_test_df['Product_Category_2']=input_feature_test_df['Product_Category_2'].fillna(input_feature_train_df['Product_Category_2'].mode().values[0])
            input_feature_test_df['Product_Category_3']=input_feature_test_df['Product_Category_3'].fillna(input_feature_train_df['Product_Category_3'].mode().values[0])  

            print(input_feature_train_df.isna().sum())
            print("------------------------------")
            print(input_feature_test_df.isna().sum())

         
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
        
            logging.info("Applying preprocessing object on training and testing datasets.")

        

            ## Apply the transformation

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

         
            
            target_feature_train_df=np.array(target_feature_train_df)[:,None]
            target_feature_test_df=np.array(target_feature_test_df)[:,None]

            train_arr = np.c_[input_feature_train_arr.toarray(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]


            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info('Processsor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


  


