import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Configuration ---
# *** IMPORTANT: Ensure this path EXACTLY matches where 'testing.mp4' is located ***
# If it's directly in C:\, keep it as r"C:\testing.mp4"
# If it's in C:\temp_videos\, change it to r"C:\temp_videos\testing.mp4"
VIDEO_PATH = r"C:\testing.mp4" # <--- VERIFY THIS PATH IS 100% CORRECT FOR YOUR FILE

OUTPUT_SNAPSHOT_FOLDER = r"C:\Users\Sneha Gupta\Videos\Captures" # Folder where extracted snapshots will be saved

# Minimum time interval (in seconds) between saving any two snapshots.
MIN_SNAPSHOT_INTERVAL_SEC = 0.5 

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Helper Functions ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coords(landmarks, landmark_enum, frame_width, frame_height):
    if not landmarks or not landmarks.landmark:
        return None
    lm = landmarks.landmark[landmark_enum]
    if lm.visibility > 0.6:
        return [int(lm.x * frame_width), int(lm.y * frame_height)]
    return None

# --- Main Logic for Snapshot Extraction ---

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_SNAPSHOT_FOLDER):
    os.makedirs(OUTPUT_SNAPSHOT_FOLDER)

# --- CRITICAL PATH DIAGNOSTIC ---
print(f"\n--- Simplified Path Check ---")
print(f"Checking for video file at: '{VIDEO_PATH}'")
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: The video file DOES NOT EXIST at the specified path: '{VIDEO_PATH}'")
    print("Please ensure the file is exactly at this location and the filename/extension are correct.")
    exit() # Exit immediately if file isn't found
else:
    print("SUCCESS: The video file exists according to os.path.exists().")
print(f"--- End Simplified Path Check ---\n")

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video '{VIDEO_PATH}'.")
    print("Possible reasons for VideoCapture failure (even if file exists):")
    print("- Video file is corrupted or not a valid video format.")
    print("- Missing necessary video codecs on your system for this file type.")
    print("- Permissions issue preventing OpenCV from reading the file.")
    exit() # Exit immediately if video cannot be opened

print(f"Starting video processing from: {VIDEO_PATH}")
print(f"Snapshots will be saved to: {OUTPUT_SNAPSHOT_FOLDER}")

frame_count = 0
saved_snapshot_count = 0
last_snapshot_time = time.time() 

right_knee_angles_history = []
left_knee_angles_history = []
MAX_HISTORY_SIZE = 10 
ANGLE_CHANGE_THRESHOLD = 5 

right_knee_state = 'unknown' 
left_knee_state = 'unknown'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(frame_rgb)
    current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    if current_frame_time - last_snapshot_time < MIN_SNAPSHOT_INTERVAL_SEC:
        pass

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

        if all([right_hip, right_knee, right_ankle]):
            current_right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            right_knee_angles_history.append(current_right_knee_angle)
            if len(right_knee_angles_history) > MAX_HISTORY_SIZE:
                right_knee_angles_history.pop(0)

        if all([left_hip, left_knee, left_ankle]):
            current_left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            left_knee_angles_history.append(current_left_knee_angle)
            if len(left_knee_angles_history) > MAX_HISTORY_SIZE:
                left_knee_angles_history.pop(0)

        if len(right_knee_angles_history) >= 3 and current_right_knee_angle is not None:
            if right_knee_angles_history[-1] > right_knee_angles_history[-2] + ANGLE_CHANGE_THRESHOLD:
                right_knee_state = 'extending'
            elif right_knee_angles_history[-1] < right_knee_angles_history[-2] - ANGLE_CHANGE_THRESHOLD:
                right_knee_state = 'flexing'
            
            if right_knee_state == 'extending' and \
               right_knee_angles_history[-2] < right_knee_angles_history[-1] - ANGLE_CHANGE_THRESHOLD and \
               right_knee_angles_history[-2] < right_knee_angles_history[-3]:
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_R_KNEE_FLEXION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Right Knee Peak Flexion: {right_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time

            elif right_knee_state == 'flexing' and \
                 right_knee_angles_history[-2] > right_knee_angles_history[-1] + ANGLE_CHANGE_THRESHOLD and \
                 right_knee_angles_history[-2] > right_knee_angles_history[-3]:
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_R_KNEE_EXTENSION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Right Knee Peak Extension/Foot Strike Candidate: {right_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time

        if len(left_knee_angles_history) >= 3 and current_left_knee_angle is not None:
            if left_knee_angles_history[-1] > left_knee_angles_history[-2] + ANGLE_CHANGE_THRESHOLD:
                left_knee_state = 'extending'
            elif left_knee_angles_history[-1] < left_knee_angles_history[-2] - ANGLE_CHANGE_THRESHOLD:
                left_knee_state = 'flexing'
            
            if left_knee_state == 'extending' and \
               left_knee_angles_history[-2] < left_knee_angles_history[-1] - ANGLE_CHANGE_THRESHOLD and \
               left_knee_angles_history[-2] < left_knee_angles_history[-3]:
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_L_KNEE_FLEXION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Left Knee Peak Flexion: {left_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time

            elif left_knee_state == 'flexing' and \
                 left_knee_angles_history[-2] > left_knee_angles_history[-1] + ANGLE_CHANGE_THRESHOLD and \
                 left_knee_angles_history[-2] > left_knee_angles_history[-3]:
                
                if current_frame_time - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL_SEC:
                    snapshot_filename = os.path.join(OUTPUT_SNAPSHOT_FOLDER, f"frame_{saved_snapshot_count:05d}_L_KNEE_EXTENSION.jpg")
                    cv2.imwrite(snapshot_filename, frame)
                    print(f"Saved snapshot: {snapshot_filename} (Left Knee Peak Extension/Foot Strike Candidate: {left_knee_angles_history[-2]:.2f} deg)")
                    saved_snapshot_count += 1
                    last_snapshot_time = current_frame_time
    
    # --- Visualization ---
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    cv2.putText(frame, f"Frame: {frame_count} | Saved: {saved_snapshot_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Video Processing (Press 'q' to quit)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\nFinished extracting snapshots. Total saved: {saved_snapshot_count} frames to '{OUTPUT_SNAPSHOT_FOLDER}'")
