# pdm-rul
Predictive Maintenance ML model in Python for RUL prediction with Streamlit interface — built for CMMS integration.


1=> Install dependencies

pip install -r requirements.txt






2=> :(OPTIONAL) Run the entire pipeline (retrain model)

python preprocessing.py             # Load + preprocess raw data + compute RUL
python select_top_features.py       # Select top 10 sensor features
python test_preprocessing.py        # Refine to top 5 features
python train_rul_baseline.py        # Train model, evaluate RMSE, save .joblib
streamlit predict.py                #run the model after manually training






3=> Launch the Streamlit app

streamlit run app.py                #run the model after automatically (manual tranining not required)

