import requests

url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
           'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA',
           'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH',
           'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N',
           'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']

data = [[63, 145, 233, 1, 150, 2.3,
         0, 1, 1, 0, 0, 0,
         1, 0, 0, 1, 0, 0, 1, 0]]

payload = {
    "dataframe_split": {
        "columns": columns,
        "data": data
    }
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Prediction:", response.json())
