import pandas as pd
import unicodedata
import hashlib
import os
import requests
import tkinter as tk
from tkinter import filedialog

def fetch_names(url):
    """Fetch names dynamically from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return {name.strip().lower() for name in response.text.splitlines()}  # Normalize to lowercase
    else:
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return set()

def normalize_name(name):
    """Normalize names by removing accents and converting to lowercase."""
    if not isinstance(name, str):
        return name
    # Remove accents
    normalized = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    return normalized.lower()

def determine_gender_combined_normalized(row, male_names, female_names):
    """Determine gender based on both first and last name columns, with normalization."""
    names_to_check = []

    # Add First Name and Last Name if they exist
    if pd.notna(row.get('First Name (Billing)', '')):
        names_to_check.append(normalize_name(row['First Name (Billing)'].strip()))
    if pd.notna(row.get('Last Name (Billing)', '')):
        names_to_check.append(normalize_name(row['Last Name (Billing)'].strip()))

    # Check each name individually
    for name in names_to_check:
        if name in male_names:
            return 'Férfi'
        if name in female_names:
            return 'Nő'

    # Only check for "ne" (normalized) if no match was found
    for name in names_to_check:
        if 'ne' in name:  # Check for 'né' normalized form in the full name
            return 'Nő'

    # If no match is found, return 'Ismeretlen'
    return 'Ismeretlen'

def mask_email(email):
    """Mask email addresses."""
    return hashlib.sha256(email.encode()).hexdigest()[:10] + "@masked.com"

def anonymize_shop_data(file_path):
    # Fetch the latest name data from the provided URLs
    male_names_url = 'https://file.nytud.hu/osszesffi.txt'
    female_names_url = 'https://file.nytud.hu/osszesnoi.txt'
    
    print("Fetching male and female names...")
    male_names = {normalize_name(name) for name in fetch_names(male_names_url)}
    female_names = {normalize_name(name) for name in fetch_names(female_names_url)}

    try:
        # Load the dataset
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print("A fájl nem található. Ellenőrizze az elérési utat, és próbálja újra.")
        return

    # Apply the gender determination logic
    df['Gender (Billing)'] = df.apply(
        lambda row: determine_gender_combined_normalized(row, male_names, female_names),
        axis=1
    )

    # Remove personal data
    columns_to_drop = [
        'First Name (Billing)', 'Last Name (Billing)',  # Sensitive data
        'First Name (Shipping)', 'Last Name (Shipping)',  # Sensitive data
        'Phone (Billing)', 'Phone (Shipping)',  # Sensitive data
        'Address 1&2 (Billing)', 'State Code (Billing)', 'Postcode (Billing)', 'Country Code (Billing)',  # Address details
        'Address 1&2 (Shipping)', 'State Code (Shipping)', 'Postcode (Shipping)', 'Country Code (Shipping)'  # Address details
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

    # Mask email addresses
    if 'Email (Billing)' in df.columns:
        df['Email (Billing)'] = df['Email (Billing)'].apply(lambda x: mask_email(x) if pd.notna(x) else x)
    else:
        print("Az 'Email (Billing)' oszlop nem található, az e-mail maszkolás kimarad.")

    # Save the anonymized data
    anonymized_file_path = os.path.splitext(file_path)[0] + '_anonymized.xlsx'
    df.to_excel(anonymized_file_path, index=False)
    print(f"Az anonimizált fájl elmentve: {anonymized_file_path}")

if __name__ == "__main__":
    # Use a file dialog to select the file
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Válassza ki az adatfájlt", filetypes=[("Excel files", "*.xlsx")])
    
    if file_path:
        anonymize_shop_data(file_path)
    else:
        print("Nem lett fájl kiválasztva.")