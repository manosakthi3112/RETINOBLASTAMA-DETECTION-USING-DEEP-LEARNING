import cv2
import numpy as np
import os
from glob import glob

# ---------- 1. Multi-Scale Retinex with Color Restoration (MSRCR) ----------
def single_scale_retinex(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img + 1.0) - np.log10(blur + 1.0)
    return retinex

def multi_scale_retinex(img, scales=[15, 80, 250]):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in scales:
        retinex += single_scale_retinex(img, sigma)
    retinex /= len(scales)
    return retinex

def MSRCR(img, scales=[15, 80, 250], alpha=125, beta=46):
    img = np.float32(img) + 1.0
    retinex = multi_scale_retinex(img, scales)
    
    # Color restoration
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img / img_sum + 1.0))
    
    msrcr = retinex * color_restoration
    msrcr = (msrcr - np.min(msrcr)) / (np.max(msrcr) - np.min(msrcr)) * 255
    return np.uint8(msrcr)

# ---------- 2. CLAHE ----------
def apply_CLAHE(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# ---------- 3. Vessel Suppression ----------
def vessel_suppression(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    suppressed = cv2.add(gray, tophat)
    suppressed = cv2.subtract(suppressed, blackhat)
    return cv2.cvtColor(suppressed, cv2.COLOR_GRAY2BGR)

# ---------- 4. Preprocessing Pipeline ----------
def preprocess_and_save(input_dir, output_dir):
    # Create subfolders
    msrcr_dir = os.path.join(output_dir, "MSRCR_rb")
    clahe_dir = os.path.join(output_dir, "CLAHE_rb")
    vessel_dir = os.path.join(output_dir, "Vessel_Suppression_rb")
    final_dir = os.path.join(output_dir, "Final_rb")
    
    for d in [msrcr_dir, clahe_dir, vessel_dir, final_dir]:
        os.makedirs(d, exist_ok=True)
    
    image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        
        filename = os.path.basename(path)
        
        # Step 1: MSRCR
        msrcr_img = MSRCR(img)
        cv2.imwrite(os.path.join(msrcr_dir, filename), msrcr_img)
        
        # Step 2: CLAHE
        clahe_img = apply_CLAHE(msrcr_img)
        cv2.imwrite(os.path.join(clahe_dir, filename), clahe_img)
        
        # Step 3: Vessel Suppression
        vessel_img = vessel_suppression(clahe_img)
        cv2.imwrite(os.path.join(vessel_dir, filename), vessel_img)
        
        # Step 4: Final Image (same as vessel output here)
        cv2.imwrite(os.path.join(final_dir, filename), vessel_img)
        
        print(f"Processed & saved: {filename}")

# ---------- Run ----------
input_dir = "Input/train/RB"         # folder with original retinal images
output_dir = "processed_images_rb"  # root folder to save results
preprocess_and_save(input_dir, output_dir)
