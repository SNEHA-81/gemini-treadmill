import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Configuration ---
VIDEO_PATH = "path/to/your/treadmill_video.mp4" # <--- IMPORTANT: CHANGE THIS TO YOUR VIDEO FILE PATH!
OUTPUT_SNAPSHOT_FOLDER = "treadmill_critical_snapshots" # Folder where extracted snapshots will be saved

# Minimum time interval (in seconds) between saving any two snapshots.
# This prevents saving too many frames for a single critical event.
MIN_SNAPSHOT_INTERVAL_SEC = 0.5 

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, # Set to True if you want each frame processed independently without tracking history
    model_complexity=1,      # 0, 1, or 2 (higher is more accurate but slower)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Helper Functions (re-used from ball.py for consistency) ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (a, b, c) where b is the vertex."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coords(landmarks, landmark_enum, frame_width, frame_height):
    """Safely gets pixel coordinates for a landmark if visible."""
    if not landmarks or not landmarks.landmark:
        return None
    
    lm = landmarks.landmark[landmark_enum]
    if lm.visibility > 0.6: # Adjust visibility threshold as needed
        return [int(lm.x * frame_width), int(lm.y * frame_height)]
    return None

# --- Main Logic for Snapshot Extraction ---

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_SNAPSHOT_FOLDER):
    os.makedirs(OUTPUT_SNAPSHOT_FOLDER)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}. Please check the path.")
    exit()

frame_count = 0
saved_snapshot_count = 0
last_snapshot_time = time.time() # Tracks time since last snapshot was saved

# --- State tracking for Critical Point Detection ---
# History of angles to detect local peaks/troughs (e.g., peak flexion, peak extension)
right_knee_angles_history = []
left_knee_angles_history = []
MAX_HISTORY_SIZE = 10 # Number of past frames to consider for peak/trough detection
ANGLE_CHANGE_THRESHOLD = 5 # Degrees: Minimum angle change to consider a significant event

# State of each knee: 'flexing' (angle decreasing) or 'extending' (angle increasing)
# Helps detect the moment of transition (e.g., from flexing to extending, indicating peak flexion)
right_knee_state = 'unknown' # Can be 'unknown', 'flexing', 'extending'
left_knee_state = 'unknown'


print(f"Starting video processing from: {VIDEO_PATH}")
print(f"Snapshots will be saved to: {OUTPUT_SNAPSHOT_FOLDER}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video stream

    frame_count += 1
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for pose landmarks
    results = pose.process(frame_rgb)

    current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current time of the frame in seconds

    # Only process for critical points if enough time has passed since the last snapshot was saved.
    # This prevents saving multiple very similar frames for the same event.
    if current_frame_time - last_snapshot_time < MIN_SNAPSHOT_INTERVAL_SEC:
        # Display the frame even if not saving a snapshot
        # cv2.imshow("Video Processing (Press 'q' to quit)", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # continue # Skip critical point detection for this frame if in cooldown
        pass # Continue to display and process pose, but don't save yet

    # Extract current knee angles if pose is detected
    current_right_knee_angle = None
    current_left_knee_angle = None

    if results.pose_landmarks:
        landmarks_dict = {
            lm.name: get_landmark_coords(results.pose_landmarks, getattr(mp_pose.PoseLandmark, lm.name), w, h)
            for lm in mp_pose.PoseLandmark
        }
        
        right_hip = landmarks_dict.get('RIGHT_HIP')
        right_knee = landmarks_dict.get('RIGHT_KNEE')
        right_ankle = landmarks_dict.get('RIGHT_ANKLE')
        left_hip = landmarks_dict.get('LEFT_HIP')
        left_knee = landmarks_dict.get('LEFT_KNEE')
        left_ankle = landmarks_dict.get('LEFT_ANKLE')

        # Update knee angle histories
        if all([right_hip, right_knee, right_ankle]):
            current_right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            right_knee_angles_history.append(current_right_knee_angle)
            if len(right_knee_angles_history) > MAX_HISTORY_SIZE:
                right_knee_angles_history.pop(0) # Remove oldest entry

        if all([left_hip, left_knee, left_ankle]):
            current_left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            left_knee_angles_history.append(current_left_knee_angle)
            if len(left_knee_angles_history) > MAX_HISTORY_SIZE:
                left_knee_angles_history.pop(0)

        # --- Critical Point Detection Logic ---
        # We need at least 3 frames in history to detect a local minimum/maximum
        if len(right_knee_angles_history) >= 3 and current_right_knee_angle is not None:
            # Simple state detection: is the angle generally increasing or decreasing?
            if right_knee_angles_history[-1] > right_knee_angles_history[-2] + ANGLE_CHANGE_THRESHOLD:
                right_knee_state = 'extending'
            elif right_knee_angles_history[-1] < right_knee_angles_history[-2] - ANGLE_CHANGE_THRESHOLD:
                right_knee_state = 'flexing'
            
            # 1. Detect Peak Flexion (Knee is maximally bent - angle is at a local minimum)
            # This happens when the knee was flexing and now starts extending (angle increases)
            if right_knee_state == 'extending' and \
               right_knee_angles_history[-2] < right_knee_angles_history[-1] - ANGLE_CHANGE_THRESHOLD and \
               right_knee_angles_history[-2] < right_knee_angles_history[-3]: # Check for local minimum (V-shape)
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_R_KNEE_FLEXION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Right Knee Peak Flexion: {right_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time # Reset cooldown

            # 2. Detect Peak Extension / Foot Strike Candidate (Knee is straightest - angle is at a local maximum)
            # This happens when the knee was extending and now starts flexing (angle decreases)
            elif right_knee_state == 'flexing' and \
                 right_knee_angles_history[-2] > right_knee_angles_history[-1] + ANGLE_CHANGE_THRESHOLD and \
                 right_knee_angles_history[-2] > right_knee_angles_history[-3]: # Check for local maximum (inverted V-shape)
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_R_KNEE_EXTENSION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Right Knee Peak Extension/Foot Strike Candidate: {right_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time


        # --- Repeat for Left Knee ---
        if len(left_knee_angles_history) >= 3 and current_left_knee_angle is not None:
            if left_knee_angles_history[-1] > left_knee_angles_history[-2] + ANGLE_CHANGE_THRESHOLD:
                left_knee_state = 'extending'
            elif left_knee_angles_history[-1] < left_knee_angles_history[-2] - ANGLE_CHANGE_THRESHOLD:
                left_knee_state = 'flexing'
            
            # 1. Detect Peak Flexion (Left Knee)
            if left_knee_state == 'extending' and \
               left_knee_angles_history[-2] < left_knee_angles_history[-1] - ANGLE_CHANGE_THRESHOLD and \
               left_knee_angles_history[-2] < left_knee_angles_history[-3]:
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_L_KNEE_FLEXION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Left Knee Peak Flexion: {left_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time

            # 2. Detect Peak Extension / Foot Strike Candidate (Left Knee)
            elif left_knee_state == 'flexing' and \
                 left_knee_angles_history[-2] > left_knee_angles_history[-1] + ANGLE_CHANGE_THRESHOLD and \
                 left_knee_angles_history[-2] > left_knee_angles_history[-3]:
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_L_KNEE_EXTENSION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Left Knee Peak Extension/Foot Strike Candidate: {left_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time
    
    # --- Visualization (Optional: to see processing live) ---
    # Draw landmarks on the frame if detected
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Green landmarks
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2) # Blue connections
        )
    cv2.putText(frame, f"Frame: {frame_count} | Saved: {saved_snapshot_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Video Processing (Press 'q' to quit)", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\nFinished extracting snapshots. Total saved: {saved_snapshot_count} frames to '{OUTPUT_SNAPSHOT_FOLDER}'")