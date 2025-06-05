import cv2
import pytesseract
import imutils
import os

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    return edged

def find_plate_contour(edged, image_shape):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    height, width = image_shape[:2]

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h
            image_area = width * height

            if 2 < aspect_ratio < 6 and (0.01 * image_area) < area < (0.15 * image_area):
                return approx
    return None

def ocr_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

def process_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Cannot load image: {filename}")
                continue

            resized = imutils.resize(image, width=600)
            edged = preprocess_image(resized)
            plate_contour = find_plate_contour(edged, resized.shape)

            if plate_contour is not None:
                x, y, w, h = cv2.boundingRect(plate_contour)
                cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                plate_img = resized[y:y+h, x:x+w]
                text = ocr_plate(plate_img)

                print(f"✅ {filename}: Detected Plate Text: {text if text else 'No text detected'}")
                cv2.imshow("Detected Plate", resized)
                cv2.waitKey(0)
            else:
                print(f"🚫 {filename}: No plate detected.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # مسیر پوشه data کنار فایل اسکریپت
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "data")
    print(f"📁 Looking for images in: {folder}")
    process_images_in_folder(folder)
