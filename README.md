# DeepFake GUI  
A streamlined, GUI-Flask powered application for detecting deepfakes using **ResNeXt50 + LSTM** architecture.

🚀 **Detection Goal**:  
Classify videos/images into:  
- `real` (labeled as `0`)  
- `fake` (labeled as `1`)  

---

## 🛠 Features  
- **Deep Learning Model**: Custom architecture combining ResNeXt50 and LSTM for high-accuracy deepfake detection.  
- **GUI Interface**: No command-line hassle—upload files and view results directly via an intuitive user interface.  
- **Input**: Supports video inputs.  

---

## 📦 Set Up a Virtual Environment
- Create a virtual environment
```python3 -m venv myenv```
- Activate the virtual environment
```source myenv/bin/activate```
```myenv\Scripts\activate```
- Install the Dependencies
```pip install Flask tensorflow torch numpy pandas opencv-python dlib tqdm```
