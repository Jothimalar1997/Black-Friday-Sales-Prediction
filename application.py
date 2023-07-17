from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Age = request.form.get('Age'),
            Occupation = int(request.form.get('Occupation')),
            City_Category = request.form.get('City_Category'),
            Stay_In_Current_City_Years = request.form.get('Stay_In_Current_City_Years'),
            Marital_Status = int(request.form.get('Marital_Status')),
            Product_Category_1 = int(request.form.get('Product_Category_1')),
            Product_Category_2= float(request.form.get('Product_Category_2')),
            Product_Category_3 = float(request.form.get('Product_Category_3'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)