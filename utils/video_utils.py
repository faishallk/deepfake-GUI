import cv2
import numpy as np
import face_recognition

def preprocess_video(video_path, max_frames=150, target_size=(256, 256)):
    """
    Preprocess video by extracting facial features from frames.
    The video is processed to focus on faces only, resized to 256x256, 
    and limited to a maximum number of frames.
    
    Parameters:
        video_path (str): Path to the input video.
        max_frames (int): Maximum number of frames to process.
        target_size (tuple): Desired size of the output frames (height, width).
    
    Returns:
        np.ndarray: Preprocessed frames as a 4D array (batch, channels, height, width).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Stop processing if the video ends
        
        # Convert frame to RGB (face_recognition uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Process the first detected face
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face = rgb_frame[top:bottom, left:right]
            
            # Resize face to target size
            face_resized = cv2.resize(face, target_size)
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized / 255.0
            
            # Append to frames list
            frames.append(face_normalized)
            frame_count += 1
    
    cap.release()
    
    # If no frames are detected, raise an error
    if not frames:
        raise ValueError("No faces detected in the video.")
    
    # Convert list to 4D numpy array (batch, height, width, channels)
    frames = np.array(frames)
    # Convert to NCHW format (batch, channels, height, width)
    frames = frames.transpose(0, 3, 1, 2)
    
    return frames
