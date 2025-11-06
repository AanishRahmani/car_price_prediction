from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

import pandas as pd

import pickle

car =pd.read_csv('cleaned_car.csv')


# model=pickle.load(open('LinearRegressionModel.pkl','rb'))

with open('LinearRegressionModel.pkl','rb') as f:
    model=pickle.load(f)


app=FastAPI()



# app.mount("/static",StaticFiles(directory='static'),name='static')

templates=Jinja2Templates(directory='template')

@app.get('/',response_class=HTMLResponse)
async def root(request:Request):
    '''
    companies
    car_models
    year
    fuel_type
    '''
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=sorted(car['fuel_type'].unique())
    return templates.TemplateResponse(
    "index.html",
    {
      "request":request,
      "companies":companies,
      "car_models":car_models,
      "year":year,
      "fuel_type":fuel_type
    }
    )


@app.post("/predict")
async def predict(request: Request):
    try:
        form = await request.form()
        form_dict = dict(form)
        print("🔹 Form data received:", form_dict)

        year = form_dict.get("year")
        km = form_dict.get("kilometer")
        company = form_dict.get("company")
        car_model_name = form_dict.get("car_model" )
        fuel = form_dict.get("fuel")
        
        # for i in [year, km, company, car_model_name, fuel]:
        #     print(type(i))

        # temp1=int(str(year))
        # print(type(temp1))


        if not all([year, km, company, car_model_name, fuel]):
            return JSONResponse(
                content={"error": "All fields are required"}, 
                status_code=400
            )
        
        year = int(str(year))
        km = int(str(km))

        df = pd.DataFrame(
            [[year, km, company, car_model_name, fuel]],
            columns=['year', 'kms_driven', 'company', 'name', 'fuel_type']
        )

        prediction = model.predict(df)

        try:
            predicted_price = float(prediction[0])
        except Exception:
            predicted_price = float(prediction)

        return JSONResponse(content={"predicted_price": float(f"{predicted_price:.2f}")})

    except ValueError as e:
        return JSONResponse(
            content={"error": "Invalid number format"}, 
            status_code=400
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
