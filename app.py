from flask import Flask, request, render_template, jsonify
import os
import torch
from utils.model_utils import load_model, predict
from utils.video_utils import preprocess_video

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = '/Users/faishalkamil/Documents/kuliah/thesis/GUI/model/model_celebdf_10epoch_revisi copy 2.pth'
model = load_model(MODEL_PATH)  # This already loads the model onto MPS device

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)

        try:
            # Preprocess the video (this gives us frames as a 4D numpy array)
            frames = preprocess_video(video_path)
            # Run the model inference on the preprocessed frames
            result = predict(model, frames)  # Predict function is already handling device transfer
        except ValueError as e:
            return jsonify({'error': str(e)})

        # Return the prediction result
        return jsonify({'result': 'Real' if result == 0 else 'Fake'})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
