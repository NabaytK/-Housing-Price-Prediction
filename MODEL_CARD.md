# retrain.py
import schedule
import time

def retrain_job():
    new_data = load_latest_data()
    X, y = preprocess_data(new_data)
    model = train_models(X, y)
    joblib.dump(model, "new_model.pkl")
    
# Retrain weekly
schedule.every().sunday.at("02:00").do(retrain_job)

while True:
    schedule.run_pending()
    time.sleep(1)
