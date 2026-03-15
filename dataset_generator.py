import pandas as pd
import numpy as np
import random
import os

def generate_dataset(num_samples=2200):
    np.random.seed(42)
    random.seed(42)
    
    crops = [
        "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
        "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
        "banana", "mango", "grapes", "watermelon", "muskmelon",
        "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee"
    ]
    
    # Approx ranges for each parameter to make the model learnable
    crop_ranges = {
        "rice": {"N": (60, 100), "P": (35, 60), "K": (35, 45), "temp": (20, 28), "humidity": (80, 85), "ph": (5.0, 7.5), "rain": (150, 300)},
        "maize": {"N": (60, 100), "P": (35, 60), "K": (15, 25), "temp": (18, 27), "humidity": (55, 75), "ph": (5.5, 7.5), "rain": (60, 110)},
        "chickpea": {"N": (20, 60), "P": (55, 80), "K": (75, 85), "temp": (17, 21), "humidity": (14, 20), "ph": (5.5, 9.0), "rain": (65, 95)},
        "kidneybeans": {"N": (0, 40), "P": (55, 80), "K": (15, 25), "temp": (15, 25), "humidity": (18, 25), "ph": (5.5, 6.0), "rain": (60, 150)},
        "pigeonpeas": {"N": (0, 40), "P": (55, 80), "K": (15, 25), "temp": (18, 38), "humidity": (30, 70), "ph": (4.5, 7.5), "rain": (90, 200)},
        "mothbeans": {"N": (0, 40), "P": (35, 60), "K": (15, 25), "temp": (24, 32), "humidity": (40, 65), "ph": (3.5, 10.0), "rain": (30, 75)},
        "mungbean": {"N": (0, 40), "P": (35, 60), "K": (15, 25), "temp": (27, 30), "humidity": (80, 90), "ph": (6.2, 7.2), "rain": (35, 60)},
        "blackgram": {"N": (20, 60), "P": (55, 80), "K": (15, 25), "temp": (25, 35), "humidity": (60, 70), "ph": (6.5, 7.8), "rain": (60, 75)},
        "lentil": {"N": (0, 40), "P": (55, 80), "K": (15, 25), "temp": (18, 30), "humidity": (60, 70), "ph": (5.5, 7.8), "rain": (35, 55)},
        "pomegranate": {"N": (0, 40), "P": (5, 30), "K": (35, 45), "temp": (18, 24), "humidity": (85, 95), "ph": (5.5, 7.2), "rain": (100, 115)},
        "banana": {"N": (80, 120), "P": (70, 95), "K": (45, 55), "temp": (25, 30), "humidity": (75, 85), "ph": (5.5, 6.5), "rain": (90, 120)},
        "mango": {"N": (0, 40), "P": (15, 40), "K": (25, 35), "temp": (27, 36), "humidity": (45, 55), "ph": (4.5, 7.0), "rain": (85, 100)},
        "grapes": {"N": (0, 40), "P": (120, 145), "K": (195, 205), "temp": (8, 42), "humidity": (80, 85), "ph": (5.5, 6.5), "rain": (65, 75)},
        "watermelon": {"N": (80, 120), "P": (5, 30), "K": (45, 55), "temp": (24, 27), "humidity": (80, 90), "ph": (6.0, 6.8), "rain": (40, 60)},
        "muskmelon": {"N": (80, 120), "P": (5, 30), "K": (45, 55), "temp": (27, 30), "humidity": (90, 95), "ph": (6.0, 6.8), "rain": (20, 30)},
        "apple": {"N": (0, 40), "P": (120, 145), "K": (195, 205), "temp": (21, 24), "humidity": (90, 95), "ph": (5.5, 6.5), "rain": (100, 125)},
        "orange": {"N": (0, 40), "P": (5, 30), "K": (5, 15), "temp": (10, 35), "humidity": (90, 95), "ph": (6.0, 7.5), "rain": (105, 120)},
        "papaya": {"N": (30, 70), "P": (45, 70), "K": (45, 55), "temp": (23, 44), "humidity": (90, 95), "ph": (6.5, 7.0), "rain": (40, 250)},
        "coconut": {"N": (0, 40), "P": (5, 30), "K": (25, 35), "temp": (25, 30), "humidity": (90, 100), "ph": (5.5, 6.5), "rain": (130, 225)},
        "cotton": {"N": (100, 140), "P": (35, 60), "K": (15, 25), "temp": (22, 26), "humidity": (75, 85), "ph": (5.8, 7.5), "rain": (60, 100)},
        "jute": {"N": (60, 100), "P": (35, 60), "K": (35, 45), "temp": (23, 27), "humidity": (70, 90), "ph": (6.0, 7.5), "rain": (150, 200)},
        "coffee": {"N": (80, 120), "P": (15, 40), "K": (25, 35), "temp": (23, 28), "humidity": (50, 70), "ph": (6.0, 7.5), "rain": (110, 200)},
    }
    
    data = []
    samples_per_crop = num_samples // len(crops)
    
    for crop in crops:
        r = crop_ranges.get(crop, {"N": (0, 100), "P": (0, 100), "K": (0, 100), "temp": (10, 40), "humidity": (10, 100), "ph": (0, 14), "rain": (0, 300)})
        for _ in range(samples_per_crop):
            data.append({
                "N": random.uniform(*r["N"]),
                "P": random.uniform(*r["P"]),
                "K": random.uniform(*r["K"]),
                "temperature": random.uniform(*r["temp"]),
                "humidity": random.uniform(*r["humidity"]),
                "ph": random.uniform(*r["ph"]),
                "rainfall": random.uniform(*r["rain"]),
                "label": crop
            })
            
    df = pd.DataFrame(data)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    out_path = os.path.join(os.path.dirname(__file__), 'crop_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} samples into {out_path}")

if __name__ == "__main__":
    generate_dataset()
