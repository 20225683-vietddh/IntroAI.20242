import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore

# ---- Cấu hình lớp ----
GENDER_CLASSES = ['Male', 'Female']
AGE_CLASSES = ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# ---- Load mô hình ----
MODEL_PATH = "outputs/trained_model_ver3.h5"  
model = load_model(MODEL_PATH)

# ---- Tiền xử lý ảnh ----
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # Resize về 256x256
    img = img.resize((256, 256))
    
    # Crop chính giữa 227x227
    left = (256 - 227) // 2
    top = (256 - 227) // 2
    right = left + 227
    bottom = top + 227
    img = img.crop((left, top, right, bottom))
    
    # Chuyển thành numpy array và chuẩn hóa
    img_array = np.array(img) / 255.0
    
    # Thêm batch dimension (1, 227, 227, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

# ---- Hàm chọn ảnh và dự đoán ----
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    try:
        input_img, pil_img = preprocess_image(file_path)
        gender_pred, age_pred = model.predict(input_img)

        gender_label = GENDER_CLASSES[np.argmax(gender_pred)]
        age_label = AGE_CLASSES[np.argmax(age_pred)]

        # Hiển thị ảnh và kết quả
        img_display = pil_img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_display)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        result_label.config(text=(
            f"Gender: {gender_label} ({gender_pred[0][np.argmax(gender_pred)]:.2%})\n"
            f"Age: {age_label} ({age_pred[0][np.argmax(age_pred)]:.2%})"
        ))

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ---- Giao diện Tkinter ----
root = tk.Tk()
root.title("Gender & Age Prediction")
root.geometry("350x400")
root.resizable(False, False)

# Nút chọn ảnh
btn = tk.Button(root, text="Choose Image", command=choose_image, font=('Arial', 12))
btn.pack(pady=15)

# Khung hiển thị ảnh
image_label = tk.Label(root)
image_label.pack()

# Kết quả dự đoán
result_label = tk.Label(root, text="", font=('Arial', 14), fg="blue")
result_label.pack(pady=10)

root.mainloop()
