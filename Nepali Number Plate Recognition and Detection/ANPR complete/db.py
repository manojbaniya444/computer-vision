import sqlite3
from datetime import datetime
import cv2

def create_database():
    conn = sqlite3.connect('lpr.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS license_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                image_path TEXT,
                capture_date TEXT,
                vehicle_type TEXT
                )''')
    conn.commit()
    conn.close()

# Function to insert license plate information into the database
def insert_license_plate(plate_number, image):
    if "kha" in plate_number:
        vehicle_type = "bus"
    elif "cha" in plate_number:
        vehicle_type = "car"
    elif "pa" in plate_number:
        vehicle_type = "bike"
    else:
        vehicle_type = "unknown"
        
    image_path = f'D:\\license_plate_db\\{plate_number}_{vehicle_type}.jpg'


    cv2.imwrite(image_path, image)

    # Insert the license plate information into the database
    conn = sqlite3.connect('lpr.db')
    c = conn.cursor()

    capture_date = datetime.now().strftime('%d%m%y %H:%M:%S')
    c.execute("INSERT INTO license_plates (plate_number, image_path, capture_date, vehicle_type) VALUES (?, ?, ?, ?)",
              (plate_number, image_path, capture_date, vehicle_type))
    print("Data inserted successfully")
    conn.commit()
    conn.close()
    
    

def fetch_latest_records():
    conn = sqlite3.connect('lpr.db')
    c = conn.cursor()
    c.execute("SELECT plate_number, capture_date, vehicle_type FROM license_plates ORDER BY id DESC LIMIT 15")
    records = c.fetchall()
    conn.close()
    return records