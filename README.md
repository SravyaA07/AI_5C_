Brain MRI Metastasis Segmentation Using Nested U-Net and Attention U-Net
Overview
This project implements brain MRI metastasis segmentation using two advanced deep learning architectures: Nested U-Net (U-Net++) and Attention U-Net. The goal is to compare the performance of both models in identifying and segmenting brain metastases and to create a web application that allows users to upload MRI images and view the segmentation results.

Key Features:
Nested U-Net and Attention U-Net Models: State-of-the-art architectures for segmentation tasks.
Data Preprocessing: Includes CLAHE-based contrast enhancement, normalization, and augmentation for better metastasis segmentation.
Web Application:
Backend: FastAPI to serve the trained segmentation model.
Frontend: Streamlit interface for users to upload images and view segmentation results.
Directory Structure
bash
Copy code
📂 brain-mri-metastasis-segmentation
 ┣ 📂 models            # Contains model implementation files
 ┣ 📂 web_app           # FastAPI and Streamlit files for web app
 ┣ 📜 app.py            # FastAPI backend
 ┣ 📜 streamlit_app.py   # Streamlit frontend
 ┣ 📜 preprocess.py      # Data preprocessing script
 ┣ 📜 train.py           # Model training and evaluation
 ┣ 📜 README.md          # Project documentation (this file)
 ┗ 📜 requirements.txt   # Required Python packages
Dataset
The dataset consists of brain MRI images and corresponding metastasis segmentation masks. The structure is as follows:

scss
Copy code
dataset/
 ┣ TCGA_CS_4941_19960909_1.tif
 ┣ TCGA_CS_4941_19960909_1_mask.tif
 ┣ TCGA_CS_4941_19960909_2.tif
 ┣ TCGA_CS_4941_19960909_2_mask.tif
 ┗ ... (more images and masks)
Note: Ensure that images without masks are excluded from training and testing.

Getting Started
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/brain-mri-metastasis-segmentation.git
cd brain-mri-metastasis-segmentation
2. Install Required Dependencies
Make sure you have Python 3.8 or higher installed. Install the required packages:

bash
Copy code
pip install -r requirements.txt
Dependencies include:

TensorFlow
OpenCV
NumPy
FastAPI
Uvicorn
Streamlit
PIL
Requests
3. Data Preprocessing
We apply CLAHE (Contrast Limited Adaptive Histogram Equalization), normalization, and augmentation for the dataset.

Run the preprocess.py script to perform preprocessing:

bash
Copy code
python preprocess.py
This script:

Enhances contrast using CLAHE.
Normalizes the images to [0, 1].
Augments the dataset for better model generalization.
4. Model Training and Evaluation
Both Nested U-Net and Attention U-Net are implemented and compared using the DICE score as the evaluation metric.

Train and evaluate the models:
bash
Copy code
python train.py
The train.py script:

Trains both the Nested U-Net and Attention U-Net.
Saves the model weights.
Evaluates the models on the test set and reports the DICE score.
5. Web Application Development
The web application consists of:

FastAPI Backend: Serves the best-performing segmentation model.
Streamlit Frontend: Allows users to upload brain MRI images and view the segmented metastasis.
Backend (FastAPI): Start the FastAPI server using:

bash
Copy code
uvicorn app:app --reload
Frontend (Streamlit): Start the Streamlit interface using:

bash
Copy code
streamlit run streamlit_app.py
6. Usage
Go to http://localhost:8501 to access the Streamlit frontend.
Upload an MRI image.
The segmentation result (metastasis mask) will be displayed alongside the original image.
Nested U-Net and Attention U-Net Architectures
Nested U-Net (U-Net++): Adds nested and dense skip connections to the traditional U-Net. This helps in handling the semantic gap between feature maps in the encoder and decoder.
Attention U-Net: Introduces attention mechanisms to the U-Net model, enabling it to focus on the most relevant regions of the image for segmentation.
DICE Score
The DICE score is used as the primary evaluation metric, as it measures the overlap between predicted and ground truth segmentation masks.

Project Files
preprocess.py: Contains code for data preprocessing (CLAHE, normalization, augmentation).
train.py: Code for training both models (Nested U-Net and Attention U-Net) and evaluating their performance using the DICE score.
app.py: FastAPI code to serve the best-performing model.
streamlit_app.py: Streamlit code for the frontend, where users can upload MRI images and view segmentation results.
Model Comparison Results
After training both models, the DICE scores for metastasis segmentation are compared. Here’s an example output:

yaml
Copy code
Nested U-Net DICE Score: 0.89
Attention U-Net DICE Score: 0.91
The model with the best score (in this case, Attention U-Net) is served via the FastAPI backend.

Challenges & Improvements
Challenges:
Data Preprocessing: CLAHE improves contrast but requires careful parameter tuning to avoid over-enhancement.
Model Selection: Nested U-Net requires more parameters, which can slow down training. Attention U-Net, while more accurate, is more computationally intensive.
Segmentation Accuracy: Achieving high overlap on smaller metastases is challenging due to their irregular shape.
Improvements & Future Work:
Better Augmentation: Introduce more advanced augmentation techniques to improve model generalization.
Model Optimization: Apply techniques like model pruning or quantization for faster inference in a clinical setting.
Incorporating 3D Data: Extending the models to handle 3D MRI data for better spatial understanding of metastases.
Future Work
Improving segmentation performance on small metastases.
Exploring 3D convolutional models for volumetric MRI data.
Deploying the model in a cloud environment for clinical use.
How to Contribute
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
U-Net++: Zongwei Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation."
Attention U-Net: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas."
