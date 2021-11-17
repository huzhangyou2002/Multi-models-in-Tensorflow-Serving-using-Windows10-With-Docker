import csv

filename = "./hyperspec.csv"
all_data_len = 563
train_data_len = 500
test_data_len = 5

def load_hyperspec_data(filename):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    pred_x = []
    pred_y = []
    i = 0
    with open(filename) as f:
        line_csv = csv.reader(f)
        for rows in line_csv:
            if(i < train_data_len):
                # 转化为float数值
                value = map(float,rows[:-1])
                train_x.append(list(value))
                # 转化为float数值
                float_y = float(rows[-1:][0])
                train_y.append(float_y)
            elif(i > train_data_len + test_data_len -1):

                value = map(float, rows[:-1])
                pred_x.append(list(value))
                # 转化为float数值
                float_y = float(rows[-1:][0])
                pred_y.append(float_y)
            else:
                # 转化为float数值
                value = map(float, rows[:-1])
                test_x.append(list(value))
                #转化为float数值
                float_y = float(rows[-1:][0])
                test_y.append(float_y)
            i = i + 1
    return train_x,train_y,test_x,test_y,pred_x,pred_y

tr_x,tr_y,t_x,t_y,p_x,p_y = load_hyperspec_data(filename)

import json
data = json.dumps({"signature_name": "serving_default", "instances": t_x})

#通过restful api 调用预测
import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/tfServing_HyperSpectral:predict', data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
for pred in predictions:
    print(pred)

print(t_y)