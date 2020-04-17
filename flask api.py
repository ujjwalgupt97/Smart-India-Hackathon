import predicta
from predicta import predict_helper

from flask import Flask,jsonify,request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)



@app.route("/", methods=['POST'])
def index():
    req_data = request.get_json()
    s = [req_data['sentence']]
    #s=['I am mad']
    output = predict_helper(s)
    return jsonify({"predcnn":output})
if __name__ == '__main__':
	app.run(debug=True)

