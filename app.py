from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from ultralytics import YOLO
import cv2


# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Configure SQLite database for storage
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'  # Database file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize the YOLO model
try:
    model = YOLO('best (2).onnx')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Start video capture
cap = cv2.VideoCapture(0)

# Create a table for storing detections
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emotion = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)

# User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the database and tables if they don't exist
with app.app_context():
    db.create_all()
    print("Database initialized!")

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to capture and process video frames
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break  

        # Convert to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert the grayscale image to 3 channels (for YOLO)
        gray_image_3d = cv2.merge([gray_image, gray_image, gray_image]) 
        
        # Use the model for inference
        results = model(gray_image_3d) if model else None
        result = results[0] if results else None

        try:
            # Annotate the frame with detection results
            annotated_frame = result.plot() if result else gray_image_3d
        except AttributeError:
            print("Error: plot() method not available for results.")
            break

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("Failed to encode frame.")
            break
        
        frame = buffer.tobytes()

        # Yield the frame to be sent to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# Routes for authentication

# Route to render the landing page before login
@app.route('/')
def landing_page():
    return render_template('landingpage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# Route to render the main HTML page after login
@app.route('/index')
@login_required
def index():
    return render_template('index.html')

# Route to stream video feed
@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to submit detection data
@app.route('/submit_detection', methods=['POST'])
@login_required
def submit_detection():
    data = request.json
    valid_emotions = ['sad', 'happy', 'angry','neutral']
    if 'emotion' in data and data['emotion'] in valid_emotions:
        try:
            # Save the emotion detection to the database
            detection = Detection(emotion=data['emotion'], timestamp=data['timestamp'])
            db.session.add(detection)
            db.session.commit()
            return jsonify({'status': 'success', 'message': f"'{data['emotion']}' recorded!"}), 200
        except Exception as e:
            return jsonify({'status': 'error', 'message': f"Database error: {e}"}), 500
    return jsonify({'status': 'error', 'message': 'Invalid emotion data!'}), 400

# Route to clear all detections from the database
@app.route('/clear_detections', methods=['DELETE'])
@login_required
def clear_detections():
    try:
        # Delete all detections from the database
        db.session.query(Detection).delete()
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'All detections cleared.'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Failed to clear detections: {e}"}), 500


# Route to fetch all detections from the database
@app.route('/detections')
@login_required
def get_detections():
    try:
        detections = Detection.query.order_by(Detection.id.desc()).limit(50).all()  # Fetch last 50 detections
        result = [{'id': det.id, 'emotion': det.emotion, 'timestamp': det.timestamp} for det in detections]
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Failed to fetch data: {e}"}), 500


# Run the Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error running app: {e}")

