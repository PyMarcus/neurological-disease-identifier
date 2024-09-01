from flask import Flask, jsonify
from image_classifier import ImageClassifier  
import os 


app = Flask(__name__)

model_path = "trained_t1.h5" 
classifier = ImageClassifier(model_path)

IMAGE_PATH = os.path.abspath(os.path.join(os.getcwd(), "image_uploaded.png"))

@app.route('/result', methods=['GET'])
def result():
    try:
        classification = classifier.classify(IMAGE_PATH)
        print(f"Classification: {classification}")
        return jsonify({"classification": classification}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) # http://127.0.0.1:5000
