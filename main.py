import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose()
pose_drawer = mp.solutions.drawing_utils

def calculate_angle_3d(point_a, point_b, point_c):
    """
    Calculate the angle at point_b formed by vectors point_a - point_b and point_c - point_b.
    Returns the angle in degrees between 0 and 180.
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    vector_ba_norm = vector_ba / np.linalg.norm(vector_ba)
    vector_bc_norm = vector_bc / np.linalg.norm(vector_bc)

    cosine_angle = np.dot(vector_ba_norm, vector_bc_norm)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def draw_text_with_unicode(image, text, position, font_path='arial.ttf', font_size=24, color=(255, 255, 255)):
    """
    Draw Unicode text onto an image using PIL.
    """
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(image_pil)

def get_pose_landmark_3d(landmarks, landmark_enum):
    """
    Extract the 3D (x, y, z) coordinates of a given landmark.
    """
    lm = landmarks[landmark_enum]
    return (lm.x, lm.y, lm.z)

def annotate_frame(frame, angle, threshold, font_path):
    """
    Add angle measurement and optional warning to the frame.
    """
    text_color = (0, 255, 0) if angle < threshold else (0, 0, 255)
    angle_text = f"Head tilt angle: {int(angle)}°"
    frame = draw_text_with_unicode(frame, angle_text, (30, 50),
                                   font_path=font_path, font_size=30, color=text_color)

    if angle >= threshold:
        frame = draw_text_with_unicode(frame, "Alert: bad posture!", (30, 90),
                                       font_path=font_path, font_size=26, color=(0, 0, 255))
    return frame

def process_frame(frame_bgr, font_path, threshold):
    """
    Process a video frame to detect pose and calculate head tilt angle.
    Returns the annotated frame.
    """
    frame_bgr = cv2.flip(frame_bgr, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose_estimator.process(frame_rgb)

    if not pose_results.pose_landmarks:
        return frame_bgr

    pose_drawer.draw_landmarks(frame_bgr, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    landmarks = pose_results.pose_landmarks.landmark

    right_shoulder = get_pose_landmark_3d(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    right_ear = get_pose_landmark_3d(landmarks, mp_pose.PoseLandmark.RIGHT_EAR)
    nose = get_pose_landmark_3d(landmarks, mp_pose.PoseLandmark.NOSE)

    head_tilt_angle = calculate_angle_3d(right_shoulder, right_ear, nose)
    frame_bgr = annotate_frame(frame_bgr, head_tilt_angle, threshold, font_path)

    return frame_bgr

def main():
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    posture_alert_threshold = 30

    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        success, frame_bgr = camera.read()
        if not success:
            break

        processed_frame = process_frame(frame_bgr, font_path, posture_alert_threshold)
        cv2.imshow("Real-time Head Posture Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

