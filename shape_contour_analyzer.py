import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

st.title("🔍 Shape & Contour Analyzer")
st.write("Upload an image to detect shapes, count objects, and calculate area & perimeter.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])


# ---------------- SHAPE DETECTION FUNCTION ----------------
def detect_shape(cnt):
    shape = "Unknown"
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)

    if peri == 0:
        return shape

    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "Triangle"

    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"

    elif vertices == 5:
        shape = "Pentagon"

    else:
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity > 0.75:
            shape = "Circle"

    return shape


# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 200, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    shape_counts = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # Mask to remove white shapes
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_color = cv2.mean(image, mask=mask)

        # Skip white / near-white objects
        if mean_color[0] > 220 and mean_color[1] > 220 and mean_color[2] > 220:
            continue

        shape = detect_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Centroid for label
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(
                image,
                shape,
                (cX - 40, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        shape_counts[shape] = shape_counts.get(shape, 0) + 1

        results.append({
            "Shape": shape,
            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2)
        })

    # ---------------- OUTPUT ----------------
    st.subheader("🖼️ Detected Shapes")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("📊 Shape Count")
    st.dataframe(
        pd.DataFrame(list(shape_counts.items()), columns=["Shape", "Count"]),
        use_container_width=True
    )

    st.subheader("📐 Area & Perimeter Details")
    st.dataframe(pd.DataFrame(results), use_container_width=True)
