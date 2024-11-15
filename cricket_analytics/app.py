from flask import Flask, Response, render_template, request, url_for, send_from_directory, jsonify
import os
import cv2
import time  # For generating unique filenames
from yellow_object_tracker import YellowObjectTracker 
from object_tracker import  YellowObjectTrackerr

app = Flask(__name__)

# Define paths for uploads and processed videos
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
uploaded_video_path = None 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def generate_frames(video_path):
    # Initialize YellowObjectTracker with the video path
    tracker = YellowObjectTracker(video_path)
    cap = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame with YellowObjectTracker
            processed_frame = tracker.process_frame(frame,1)  # Assuming process_frame is a method in YellowObjectTracker
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
def generate_frames1(video_path):
    # Initialize YellowObjectTracker with the video path
    tracker = YellowObjectTrackerr(video_path)
    cap = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame with YellowObjectTracker
            processed_frame = tracker.process_frame(frame,1)  # Assuming process_frame is a method in YellowObjectTracker
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()    

@app.route('/video_feed/<filename>')
def video_feed(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if filename:
        return Response(generate_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video uploaded", 400
@app.route('/video_feed1/<filename>')
def video_feed1(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if filename:
        return Response(generate_frames1(path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video uploaded", 400    
@app.route('/')
def index():
    return render_template('index.html') 

    # List all images in the 'static/class-0' folder
    # image_folder = os.path.join('static', 'class_0')
    # image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    # return render_template('index.html', image_files=image_files)
 # Main upload form
# @app.route('/')
# def detect():
#     return render_template('detect.html') 
#   <video controls="" autoplay="" name="media">
#     <source src="./sample.mp4" type="video/mp4">
#   </video>

@app.route('/upload', methods=['POST'])
def upload_video():
    # Check if the request contains a file

    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if video_file:
        # Save the uploaded video to the designated upload folder
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)

        # Generate a unique filename for the processed video
        output_filename = f"processed_{video_file.filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        boutput_filename = f"processed_b_{video_file.filename}"
        boutput_path = os.path.join(app.config['OUTPUT_FOLDER'],boutput_filename)
        
        # Process the video using your YellowObjectTracker class
        tracker = YellowObjectTracker(video_path=video_path)
        tracker.run(output_path)  # Run the tracking and save the processed video
        ball_tracker = YellowObjectTrackerr(video_path=video_path)
        ball_tracker.run(boutput_path)

        # Generate the URL for the processed video
        # video_url = url_for('display_video', filename=output_filename)

        # Return the URL of the processed video and the filename as JSON response
        # return jsonify({"video_url": video_url, "processed_filename": output_filename})
        return render_template("index.html", 
                               vid=True,
                               vid_url1=f'video_feed/{output_filename}',
                               vid_url2=f'video_feed/{boutput_filename}')
@app.route('/upload1', methods=['POST'])
def upload_video1():
    # Check if the request contains a file

    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if video_file:
        # Save the uploaded video to the designated upload folder
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)

        # Generate a unique filename for the processed video
        output_filename = f"processed_{video_file.filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Process the video using your YellowObjectTracker class
        tracker = YellowObjectTrackerr(video_path=video_path)
        tracker.run(output_path)  # Run the tracking and save the processed video

        # Generate the URL for the processed video
        # video_url = url_for('display_video', filename=output_filename)

        # Return the URL of the processed video and the filename as JSON response
        # return jsonify({"video_url": video_url, "processed_filename": output_filename})
        return render_template("index.html", vid=True,vid_url=f'video_feed1/{output_filename}')    

@app.route('/output/<filename>')
def display_video(filename):
    
    # Send the processed video file with the correct MIME type for video
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, mimetype='video/mp4')




if __name__ == "__main__":     
    app.run(debug=True)
