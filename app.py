# app.py

from flask import Flask, request, render_template, Response
import json
import cv2
#car model classification
from classifier import predict

#car plate detection
from detection import detect

#car plate recognition
from recognizer import recognize

app = Flask(__name__)



@app.route('/upload', methods=['GET', 'POST'])
def service():
	if request.method == 'POST':
		file = request.files['file']
		file.save('image_test.jpg')
		
		# Car model classification
		brand, model, veh_type = predict('image_test.jpg')
		
		#Car plate detection
		detect('image_test.jpg')

		#Car plate recognition
		text, prob = recognize('X000XX000.jpg') 
		response = {"brand":brand,"model":model,"probability":prob,"veh_type":veh_type,"coord":"[(398,292),(573,360)]","id":"0001","plate":text}
		response = json.dumps(response, ensure_ascii=False)

		return Response(response=response, status=200, mimetype="application/json")	
	return render_template("service.html")


# We only need this for local development.
if __name__ == '__main__':
	app.run()
