import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

# ---------------- STUDENT DETAILS (EDIT HERE) ----------------
NAME = "Ravisankar"
REGISTER_NUMBER = "22MIA1182"
SUBJECT = "Computer Vision"

# ---------------- HEADER SECTION ----------------
st.markdown(
    f"""
    <div style="background-color:#0E76A8;padding:15px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Shape & Contour Analyzer</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="padding:15px;border:2px solid #0E76A8;border-radius:10px;margin-top:10px">
        <h4><b>Name:</b> {NAME}</h4>
        <h4><b>Register Number:</b> {REGISTER_NUMBER}</h4>
        <h4><b>Subject:</b> {SUBJECT}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")
st.write("Upload an image to detect shapes, count objects, and calculate area & perimeter.")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# ---------------- SHAPE DETECTION FUNCTION ----------------
def detect_shape(cnt):
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    else:
        return "Circle"

# ---------------- MAIN PROCESSING ----------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (600, 400))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = image.copy()
    data = []
    object_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            object_count += 1
            perimeter = cv2.arcLength(cnt, True)
            shape = detect_shape(cnt)

            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    output, shape, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )

            data.append({
                "Object ID": object_count,
                "Shape": shape,
                "Area": round(area, 2),
                "Perimeter": round(perimeter, 2)
            })

    df = pd.DataFrame(data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, channels="BGR")

    with col2:
        st.subheader("Detected Shapes")
        st.image(output, channels="BGR")

    st.subheader("📊 Analysis Results")
    st.write("Total Objects Detected:", object_count)
    st.dataframe(df)

else:
    st.info("Please upload an image to begin.")
