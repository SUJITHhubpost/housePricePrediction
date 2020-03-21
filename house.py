from flask import Flask, redirect, render_template, request, jsonify
import house_price_prediction as hs
import numpy as np
import pandas as pd

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/prediction/')
def res():
    minx, maxx = hs.minmax()
    return render_template('prediction.html', minx = minx, maxx = maxx)


@app.route('/result')
def result():
    x0 = 1
    x1 = int(request.args.get('x1'))
    x2 = int(request.args.get('x2'))
    x3 = int(request.args.get('x3'))
    x4 = int(request.args.get('x4'))
    
    x = np.array([x0, x1, x2, x3, x4])
    final = hs.predict(x)
    
    return jsonify({"result" : round(final * 75, 4)})

@app.route('/train')
def train():
    
    df_train = pd.read_csv('data/train.csv', usecols = ['GrLivArea','1stFlrSF', '2ndFlrSF', 'LotArea', 'SalePrice'])

    
    df_train.to_html(r'templates/table.html')
    
    return render_template('results.html')

@app.route("/model")
def model():
    iter = int(request.args.get('iter'))
    cst = hs.train(iter)
    return jsonify({"result" : cst })
    

if __name__ == "__main__":
    app.run(debug=True)
