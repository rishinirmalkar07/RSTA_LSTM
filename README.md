NOₓ Emission Forecasting using RSTA-LSTM Deep Learning Model


Project Overview :

This project implements a deep learning-based time series forecasting model to estimate NOₓ (Nitrogen Oxide) emissions using a hybrid Recurrent Spatiotemporal Attention Long Short-Term Memory (RSTA-LSTM)
architecture. The aim is to provide a precise, scalable, and interpretable solution for air quality modeling, specifically focusing on predicting nitrous oxide emissions from multivariate input features.


Publication :

This work has been published in the International Journal of Research Publication and Reviews, .
Read the Official Publication - https://ijrpr.com/uploads/V6ISSUE5/IJRPR46528.pdf
Certification - https://ijrpr.com/certificate/download.php?paper_id=31548 


Key Features :

1. Custom Spatiotemporal Attention Layer using Keras
2. Multivariate Time Series Modeling
3. Data Preprocessing Pipeline using pandas and MinMaxScaler
4. Train/Val/Test Splitting with preserved temporal order
5. Model Evaluation Metrics: MSE, RMSE, MAE
6. Result Visualization using matplotlib
7. Lightweight and reproducible (single file: main.py)


Dataset :

File - synthetic_nox_emission_data.csv
Shape - Multivariate time series with features relevant to NOₓ emission
Target Variable - NOx_Emission
Time Steps - Window size of 10 for sequence modeling


Model Architecture :

1. RSTA-LSTM Model Pipeline -
  Input → [LSTM → Dropout → LSTM] → Spatiotemporal Attention → Dense → Output
2. Custom Attention Layer -
  A custom SpatiotemporalAttention layer is implemented to capture dependencies in both space (features) and time, using learnable attention weights with L1 regularization.


Tech Stack :

1. Language - Python3
2. Deep Learning - TensorFlow / Keras
3. Data Processing - Pandas, NumPy, Scikit-learn
4. Visualization - Matplotlib
ikit-learn matplotlib


Output & Evaluation :

Upon training completion, the model -
1. predicts NOx emissions on test data
2. Prints the following metrics:
   i. MSE
   ii. RMSE
   iii. MAE
3. Generates a plot comparing actual vs predicted NOx emissions.
<img width="1919" height="980" alt="Screenshot 2025-08-02 203138" src="https://github.com/user-attachments/assets/789fc5b7-b753-4887-997e-0e568f0860f0" />


Sample Output :

Evaluation Metrics:
Mean Squared Error (MSE): 0.0032
Root Mean Squared Error (RMSE): 0.0565
Mean Absolute Error (MAE): 0.0432

 
Future Improvements :

1. Integrate real-time emission data using sensors
2. Support for multi-gas prediction (CO₂, CH₄, N₂O)
3. Deploy model using Flask/Streamlit for public dashboards
4. Use hybrid models: CNN + LSTM, Transformer-based variants
5. Incorporate explainability using SHAP or attention heatmaps


Acknowledgments :

This project was developed under the guidance and support of Bhilai Institute of Technology, Raipur.
Project Mentors & Reviewers Team - Prof.Aparna Pandey , S Shubham , Rishi Nirmalkar , Pranjal Shrivas , Divyanshi Sharma


Contact :

For any queries, suggestions, or collaboration opportunities, feel free to connect -
Email - sshubham22062003@gmail.com
LinkedIn - https://www.linkedin.com/in/s-shubham-317359229
