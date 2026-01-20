# 🖼️ AI Background Blur Application

An **AI-powered background blur web application** that automatically separates the foreground from the background using deep learning–based image segmentation and applies adjustable background blur while preserving sharp foreground details.

Built with **Streamlit** and **Hugging Face Transformers**, this project demonstrates practical use of **computer vision**, **image segmentation**, and **interactive UI design**.

---


---

## 🧠 How It Works

1. The user uploads an image via the web interface.
2. A **SegFormer-based image segmentation model** identifies foreground regions.
3. The segmentation mask is refined using thresholding and smoothing.
4. The background is blurred while preserving foreground clarity.
5. The final image is displayed and made available for download.

---

## 🛠️ Tech Stack

| Component        | Technology               |
| ---------------- | ------------------------ |
| Frontend UI      | Streamlit                |
| Image Processing | Pillow (PIL)             |
| AI Model         | SegFormer (Hugging Face) |
| Deep Learning    | PyTorch                  |
| Data Handling    | NumPy                    |

---

## 📁 Project Structure

```
background-blur-app/
│
├── blur_images.py          # Main Streamlit application
├── requirements.txt        # Project dependencies
├── Model/                  # Local segmentation model directory
│   └── models--nvidia--segformer-b2-clothes/
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/background-blur-app.git
cd background-blur-app
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Download Segmentation Model

Download the **SegFormer B2 Clothes model** from Hugging Face and place it in:

```
Model/models--nvidia--segformer-b2-clothes/
```

> You can also modify the model path in `blur_images.py` to load directly from Hugging Face.

---

### 4️⃣ Run the Application

```bash
streamlit run blur_images.py
```

---

## 🖥️ Usage

1. Upload an image (`.jpg`, `.jpeg`, `.png`)
2. Adjust the background blur intensity using the slider
3. Click **Process Image**
4. Preview the result
5. Download the processed image

---

## 🎯 Use Cases

* Profile picture enhancement
* Social media content creation
* E-commerce product images
* Photography post-processing
* AI-powered photo editing tools

---


## 🧩 Known Limitations

* Performance depends on image size and hardware
* Segmentation quality may vary for complex backgrounds
* Initial model loading may take a few seconds

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests to improve functionality or performance.

---

## 📄 License

This project is open-source and available under the **MIT License**.

---

## 👩‍💻 Author

**Monasri M**
B.Tech – Artificial Intelligence & Data Science
AI | Computer Vision | Full-Stack AI Systems

---
