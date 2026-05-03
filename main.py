import pandas as pd
import json
import os

def csv_to_json(csv_path, output_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    json_data_list = []
    
    for index, row in df.iterrows():
        record = {
            "id": index,
            "Cografya": {
                "Yukseklik": round(row['Yukseklik'], 2),
                "Egim": round(row['Egim'], 2)
            },
            "Meteoroloji": {
                "Sicaklik": round(row['Sicaklik_C'], 2),
                "Ruzgar_Hizi": round(row['Ruzgar_Hizi'], 2)
            },
            "Yakitlar": {  
                "NDVI": round(row['NDVI'], 2),
                "NDWI": round(row['NDWI'], 2)
            },
            "Gercek_Yangin_Durumu": int(row['YANGIN_DURUMU']) 
        }
        json_data_list.append(record)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data_list, f, ensure_ascii=False, indent=4)
        
    print(f"Converted {len(json_data_list)} records to JSON at {output_path}")

if __name__ == "__main__":
    csv_file = "data/dataset_spatial_filter.csv"
    json_file = "data/formatted_data.json"
    csv_to_json(csv_file, json_file)