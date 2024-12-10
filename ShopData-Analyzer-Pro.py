import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import tab20
from google.colab import files
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import gdown
import openai
import textwrap
import warnings

# Csak DeprecationWarning figyelmeztetések elnyomása
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load city-to-county mapping CSV
# city_to_county_mapping_path = '/content/city-to-county.csv'
# city_to_county_df = pd.read_csv(city_to_county_mapping_path)

# Google Drive fájl azonosítója
file_id = "1UPeI6SG3rle05jdVZ106RCw9mtP_SsRn"
download_url = f"https://drive.google.com/uc?id={file_id}"
local_file_path = "city_mapping.csv"

# Fájl letöltése
print("City-to-county fájl letöltése...")
gdown.download(download_url, local_file_path, quiet=False)

# Fájl betöltése
try:
    city_to_county_df = pd.read_csv(local_file_path)
    print("City-to-county mapping fájl sikeresen betöltve.")
except FileNotFoundError:
    raise FileNotFoundError("A fájl letöltése sikertelen. Ellenőrizd a Google Drive linket vagy a fájlazonosítót.")

# File path for the previously uploaded file
previous_file_path = '/content/last_uploaded_file.xlsx'

# Prompt user to decide if they want to upload a new file
if not os.path.exists(previous_file_path) or input("Szeretnél új fájlt feltölteni? (igen/nem): ").strip().lower() == 'igen':
    print("Kérlek tölts fel egy új fájlt az elemzéshez.")
    uploaded = files.upload()
    # Get the uploaded file name
    file_path = list(uploaded.keys())[0]
    # Save the new file with a consistent name for future use
    os.rename(file_path, previous_file_path)
    file_path = previous_file_path
else:
    print(f"A korábban feltöltött fájl lesz használva: {previous_file_path}.")
    file_path = previous_file_path

# Ask user if they want detailed text analysis
use_openai_analysis = input("Szeretnél részletes szöveges elemzést készíteni a végén? (igen/nem): ").strip().lower() == 'igen'

# Load the Excel file (always use the first sheet, regardless of its name)
df = pd.read_excel(file_path, sheet_name=0)

# Define the list of columns to keep
columns_to_keep = [
    'Order Number', 'Order Date', 'Customer Note', 'Order Status',
    'City (Billing)', 'Email (Billing)',
    'Order Shipping Amount', 'Shipping Method Title',
    'Order Line Total (include tax)', 'Order Refund Amount',
    'Payment Method Title', 'Customer Total Spent', 'Customer Total Orders',
    'Customer first order date', 'Customer last order date',
    'SKU', 'Item ID', 'Item Name', 'Product Id', 'Short Description', 'Description',
    'Type', 'Quantity', 'Stock Quantity', 'Stock Status',
    'Item Discount Amount', 'Coupon Code',
    'Full names for categories', 'Category', 'Gender (Billing)', 'Origin'
]

# Csak azok az oszlopok kerülnek kiválasztásra, amelyek léteznek a DataFrame-ben
existing_columns = [col for col in columns_to_keep if col in df.columns]
missing_columns = [col for col in columns_to_keep if col not in df.columns]

# Tájékoztatás a hiányzó oszlopokról
if missing_columns:
    print(f"Hiányzó oszlopok az adatbázisból: {missing_columns}")

# Filter the DataFrame to keep only the existing columns
df_cleaned = df[existing_columns]

# Filter rows where the Order Status is "Teljesítve"
df_cleaned = df_cleaned[df_cleaned['Order Status'] == 'Teljesítve'].copy()

# Convert numeric columns and handle empty values
numeric_columns = ['Order Line Total (include tax)', 'Order Refund Amount', 'Item Discount Amount']
for col in numeric_columns:
    if col in df_cleaned.columns:
        df_cleaned.loc[:, col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)

# Ensure 'Order Date' is in datetime format
df_cleaned['Order Date'] = pd.to_datetime(df_cleaned['Order Date'], errors='coerce')

# Extract Year and Month from the 'Order Date'
df_cleaned['Year'] = df_cleaned['Order Date'].dt.year
df_cleaned['Month'] = df_cleaned['Order Date'].dt.month

# Step 1: Calculate the line-level revenue
df_cleaned['Line Revenue'] = df_cleaned['Order Line Total (include tax)'] - df_cleaned['Item Discount Amount']

# Step 2: Aggregate at the order level
order_aggregated = df_cleaned.groupby('Order Number').agg({
    'Line Revenue': 'sum',  # Sum of all lines for each order
    'Order Refund Amount': 'first',  # Single value for the refund amount per order
    'Order Date': 'first',  # Keep the order date for grouping later
    'Year': 'first',
    'Month': 'first',
}).reset_index()

# Calculate Net Revenue per order
order_aggregated['Net Revenue Per Order'] = order_aggregated['Line Revenue'] - order_aggregated['Order Refund Amount']

# Step 3: Update monthly revenue aggregation
monthly_revenue = order_aggregated.groupby(['Year', 'Month']).agg({
    'Net Revenue Per Order': 'sum'
}).reset_index()

# Pivot for visualization
monthly_revenue_pivot = monthly_revenue.pivot(index='Month', columns='Year', values='Net Revenue Per Order')

# Format the numbers for better readability
monthly_revenue_pivot_formatted = monthly_revenue_pivot.applymap(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

# Calculate yearly totals for additional display
yearly_totals = monthly_revenue_pivot.sum(axis=0)
yearly_totals_formatted = yearly_totals.apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

# Print the monthly revenue data
print("\033[1;34m--- Havi és éves értékesítési adatok ---\033[0m")
print("Havi és éves értékesítési adatok (kumulált):")
print(monthly_revenue_pivot_formatted)

# Print the yearly totals
print("\nÉves összesített bevételek:")
print(yearly_totals_formatted)

# Plot the data
plt.figure(figsize=(14, 8))
monthly_revenue_pivot_numeric = monthly_revenue_pivot.apply(pd.to_numeric, errors='coerce')
monthly_revenue_pivot_numeric.plot(kind='bar', figsize=(14, 8), width=0.8)

plt.title("Értékesítési volumen havi bontásban (kumulált értékek)")
plt.xlabel("Hónap")
plt.ylabel("Bevétel (Ft)")
plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Már", "Ápr", "Máj", "Jún", "Júl", "Aug", "Szept", "Okt", "Nov", "Dec"])
plt.legend(title="Év", loc="upper left")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Folytatás: Népszerű kategóriák elemzése
latest_years = sorted(df_cleaned['Year'].unique())[-2:]  # Utolsó két év
df_recent = df_cleaned[df_cleaned['Year'].isin(latest_years)].copy()

# Kategóriák kinyerése
df_recent['Last Category'] = df_recent['Full names for categories'].str.split('>').str[-1].str.strip()

# Csoportosítás bevétel szerint
category_revenue = df_recent.groupby(['Year', 'Last Category']).agg({
    'Line Revenue': 'sum'  # Use Line Revenue since it's pre-aggregated
}).reset_index()

# Az éves teljes bevétel kiszámítása
yearly_revenue = category_revenue.groupby('Year')['Line Revenue'].sum().to_dict()

# Arányok kiszámítása
category_revenue['Percentage'] = category_revenue.apply(
    lambda row: (row['Line Revenue'] / yearly_revenue[row['Year']]) * 100, axis=1
)

# Top 8 kategória minden évre
top_categories = category_revenue.groupby('Year').apply(
    lambda x: x.nlargest(8, 'Line Revenue')
).reset_index(drop=True)

# **Ábrázolás**

for year in latest_years:
    data = top_categories[top_categories['Year'] == year]
    total_revenue = yearly_revenue[year]

    # Add "Other" category
    other_revenue = total_revenue - data['Line Revenue'].sum()
    other_row = pd.DataFrame({
        'Year': [year],
        'Last Category': ['Minden más'],
        'Line Revenue': [other_revenue],
        'Percentage': [(other_revenue / total_revenue) * 100]
    })
    data = pd.concat([data, other_row], ignore_index=True)

    # Round revenue for display
    data['Line Revenue'] = data['Line Revenue'].round(0).astype(int)
    data['Percentage'] = data['Percentage'].round(1)

    # Print category data BEFORE the pie chart
    print("\033[1;34m--- Kategóriák ---\033[0m")
    print(f"\n--- {year} kategória bevételek ---")
    print(
        data[['Last Category', 'Line Revenue', 'Percentage']]
        .to_string(index=False, header=['Kategória', 'Bevétel (Ft)', 'Arány (%)'])
    )

    # Pie chart visualization with distinct colors
    plt.figure(figsize=(8, 8))
    colors = tab20(range(len(data)))  # Generate distinct colors for each slice
    plt.pie(
        data['Line Revenue'],
        labels=data['Last Category'],
        colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        startangle=90
    )
    plt.title(f"Népszerű termékkategóriák és a teljes bevétel arányai ({year})")
    plt.show()

# Top 8 termék
# Szűrés az utolsó két év adataira, ha van adat
available_years = sorted(order_aggregated['Year'].unique())
latest_years = available_years[-2:] if len(available_years) >= 2 else available_years

# Szűrés a legutóbbi két év adataira
df_recent = df_cleaned[df_cleaned['Year'].isin(latest_years)].copy()

# Nettó rendelési bevétel számítása termékenként (Item ID alapján)
product_revenue = df_recent.groupby(['Year', 'Item ID']).agg({
    'Line Revenue': 'sum',  # Termék szintű bevételek aggregálása
    'Item Name': 'first'    # Az első megnevezés megőrzése a címkézéshez
}).reset_index()

# Kerekítés a kiíráshoz
product_revenue['Line Revenue'] = product_revenue['Line Revenue'].round(0).astype(int)

# Minden évre a top 8 termék kiválasztása a bevétel alapján
top_products = product_revenue.groupby('Year').apply(
    lambda x: x.nlargest(8, 'Line Revenue')
).reset_index(drop=True)

print("\033[1;34m--- Termékek ---\033[0m")
# Eredmények kiírása képernyőre, balra igazított terméknévvel és kerekített összegekkel
for year in sorted(latest_years, reverse=True):
    print(f"\n--- {year} legkeresettebb termékei (nettó bevétel alapján) ---")
    top_year_products = top_products[top_products['Year'] == year]
    print(
        top_year_products[['Item ID', 'Item Name', 'Line Revenue']]
        .to_string(index=False, justify='left', header=['Termék ID', 'Termék név', 'Bevétel (Ft)'])
    )

# Diagram: fekvő oszlopdiagram, terméknevek címkeként, bevétel szerint csökkenő sorrendben
for year in sorted(latest_years, reverse=True):
    # Rendezzük a termékeket bevétel szerint csökkenő sorrendbe
    top_year_products = top_products[top_products['Year'] == year].sort_values('Line Revenue', ascending=False)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_year_products)), top_year_products['Line Revenue'], color='skyblue', align='center')

    # Fordított sorrend a tengelyekhez, hogy a legnagyobb bevételű termék legyen felül
    plt.yticks(range(len(top_year_products)), top_year_products['Item ID'])
    plt.ylabel("Termék ID")
    plt.xlabel("Nettó Bevétel (Ft)")
    plt.title(f"{year} legkeresettebb termékei (nettó bevétel alapján)")

    # Terméknevek hozzáadása az oszlopok végéhez
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 5000, i,  # Az oszlop végéhez írjuk a terméknevet
            top_year_products.iloc[i]['Item Name'],
            va='center', fontsize=9
        )

    # Felső és jobb oldali keret eltüntetése
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.gca().invert_yaxis()  # Fordítsuk meg az Y-tengelyt, hogy a legnagyobb bevétel legyen felül
    plt.show()

# Modul: Átlagos kosárérték elemzése

# Csoportosítás havi szintű aggregációhoz
average_order_value = order_aggregated.groupby(['Year', 'Month']).agg({
    'Net Revenue Per Order': 'mean'  # Átlagos rendelési érték számítása
}).reset_index()

# Pivot a vizualizációhoz (év szerint oszlopok, hónapok szerint sorok)
aov_pivot = average_order_value.pivot(index='Month', columns='Year', values='Net Revenue Per Order')

# Kiírás: Átlagos kosárérték havi bontásban
print("\033[1;34m--- Kosárérték ---\033[0m")
print("\nHavi átlagos kosárérték:")
print(aov_pivot.applymap(lambda x: f"{x:,.2f} Ft" if pd.notnull(x) else "").to_string())

# Vizualizáció: Vonaldiagram
plt.figure(figsize=(12, 6))
for year in aov_pivot.columns:
    plt.plot(aov_pivot.index, aov_pivot[year], marker='o', label=str(year))

plt.title("Átlagos kosárérték havi bontásban")
plt.xlabel("Hónap")
plt.ylabel("Átlagos rendelési érték (Ft)")
plt.xticks(ticks=range(1, 13), labels=["Jan", "Feb", "Már", "Ápr", "Máj", "Jún", "Júl", "Aug", "Szept", "Okt", "Nov", "Dec"])
plt.legend(title="Év")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Modul: Regionális elemzés (megyék és városok)

# Szűrés az utolsó két év adataira, ha van adat
available_years = sorted(order_aggregated['Year'].unique())
latest_years = available_years[-2:] if len(available_years) >= 2 else available_years

# Szűrés a legutóbbi évek adataira
df_recent = df_cleaned[df_cleaned['Year'].isin(latest_years)].copy()

# Megyék hozzárendelése városokhoz (helyes oszlopnevekkel)
df_recent = df_recent.merge(city_to_county_df, how='left', left_on='City (Billing)', right_on='city')

# Nettó rendelési bevétel számítása városok és megyék szerint
regional_revenue = df_recent.groupby(['Year', 'county', 'City (Billing)']).agg({
    'Line Revenue': 'sum'  # Nettó bevétel aggregálása
}).reset_index()
print("\033[1;34m--- Régiók ---\033[0m")
# Az utolsó két év legjobb megyéi és városai
for year in sorted(latest_years, reverse=True):
    print(f"\n--- {year} legjobb megyéi (nettó bevétel alapján) ---")
    top_counties = regional_revenue[regional_revenue['Year'] == year].groupby('county').agg({
        'Line Revenue': 'sum'
    }).nlargest(4, 'Line Revenue').reset_index()
    print(top_counties.to_string(index=False, header=['Megye', 'Bevétel (Ft)']))

    print(f"\n--- {year} legjobb városai (nettó bevétel alapján) ---")
    top_cities = regional_revenue[regional_revenue['Year'] == year].groupby('City (Billing)').agg({
        'Line Revenue': 'sum'
    }).nlargest(4, 'Line Revenue').reset_index()
    print(top_cities.to_string(index=False, header=['Város', 'Bevétel (Ft)']))

    # Vizualizáció: Megyék és városok oszlopdiagram
    plt.figure(figsize=(12, 6))
    plt.bar(top_counties['county'], top_counties['Line Revenue'], color='orange', alpha=0.7, label='Megyék')
    plt.bar(top_cities['City (Billing)'], top_cities['Line Revenue'], color='skyblue', alpha=0.7, label='Városok')

    plt.title(f"Legjobb régiók bevétele ({year})")
    plt.xlabel("Régió")
    plt.ylabel("Bevétel (Ft)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Top 10 és további termékek számának megjelenítése
def display_top_items(df, title, max_items=10):
    total_items = len(df)
    top_items = df.head(max_items)  # Csak az első max_items terméket mutatjuk meg

    print(f"\n--- {title} ---")
    if total_items > 0:
        print(top_items[['Item ID', 'Item Name', 'Quantity', 'Daily Average Sales', 'Stock Quantity']]
              .to_string(index=False, header=['Termék ID', 'Termék név', 'Értékesített mennyiség', 'Átlagos napi fogyás', 'Jelenlegi készlet']))

        # Jelezzük, ha további termékek tartoznak a csoportba
        if total_items > max_items:
            print(f"\n... és még {total_items - max_items} további termék tartozik ebbe a csoportba.")
    else:
        print("Nincs ilyen termék.")

# Modul: Készletgazdálkodás elemzése
print("\033[1;34m--- Készlet - figyelmeztető jelek ---\033[0m")
from datetime import datetime, timedelta

# Szűrés az utolsó 365 nap adataira
one_year_ago = datetime.now() - timedelta(days=365)
df_recent = df_cleaned[df_cleaned['Order Date'] >= one_year_ago].copy()

# Összesített értékesítési mennyiség (Quantity) számítása Item ID szerint
sales_summary = df_recent.groupby(['Item ID', 'Item Name']).agg({
    'Quantity': 'sum',  # Értékesített mennyiség az utolsó évben
    'Stock Quantity': 'first',  # Jelenlegi készletszint
}).reset_index()

# Átlagos napi fogyás számítása
sales_summary['Daily Average Sales'] = sales_summary['Quantity'] / 365

# Kockázatosan alacsony készlet azonosítása
threshold_high_sales = sales_summary['Daily Average Sales'].quantile(0.8)  # Felső 20% határértéke
sales_summary['Low Stock Risk'] = (sales_summary['Daily Average Sales'] > threshold_high_sales) & \
                                  (sales_summary['Stock Quantity'] < (sales_summary['Daily Average Sales'] * 7))

low_stock_risk_items = sales_summary[sales_summary['Low Stock Risk']].sort_values('Daily Average Sales', ascending=False)

# Feleslegesen magas készlet azonosítása
threshold_low_sales = sales_summary['Daily Average Sales'].quantile(0.2)  # Alsó 20% határértéke
sales_summary['Excess Stock Risk'] = (sales_summary['Daily Average Sales'] <= threshold_low_sales) & \
                                     (sales_summary['Stock Quantity'] > (sales_summary['Daily Average Sales'] * 90))

excess_stock_risk_items = sales_summary[sales_summary['Excess Stock Risk']].sort_values('Daily Average Sales', ascending=True)

# Eredmények megjelenítése
display_top_items(low_stock_risk_items, "Kockázatosan alacsony készletszintű termékek (utolsó 365 nap)")
display_top_items(excess_stock_risk_items, "Feleslegesen magas készletszintű termékek (utolsó 365 nap)")

# Modul: Kuponok és diszkontok elemzése

# Szűrés: Nem üres vagy nulla értékű kedvezmények
discount_data = df_cleaned[df_cleaned['Item Discount Amount'] > 0].copy()

# Kedvezmény összegzése éves bontásban
discount_summary = discount_data.groupby('Year').agg({
    'Item Discount Amount': 'sum',  # Összesített elengedett összeg
    'Order Number': 'nunique'      # Egyedi rendelések darabszáma kedvezménnyel
}).reset_index()

# Átnevezés az oszlopok érthetősége érdekében
discount_summary.rename(columns={
    'Item Discount Amount': 'Elengedett Összeg (Ft)',
    'Order Number': 'Kedvezményes Rendelések Száma'
}, inplace=True)

# Eredmények kiírása
print("\033[1;34m--- Kedvezmények ---\033[0m")
print("\n--- Éves kedvezmény kimutatás ---")
print(discount_summary.to_string(index=False))

# Vizualizáció: Oszlopdiagram
plt.figure(figsize=(10, 6))

# Két tengelyes diagram létrehozása
fig, ax1 = plt.subplots(figsize=(10, 6))

# Kedvezményes rendelések darabszáma (bal tengely)
ax1.bar(discount_summary['Year'], discount_summary['Kedvezményes Rendelések Száma'],
        color='skyblue', alpha=0.7, label='Kedvezményes Rendelések Száma')
ax1.set_ylabel('Kedvezményes Rendelések Száma', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Elengedett összeg (jobb tengely)
ax2 = ax1.twinx()
ax2.plot(discount_summary['Year'], discount_summary['Elengedett Összeg (Ft)'],
         color='orange', marker='o', label='Elengedett Összeg (Ft)')
ax2.set_ylabel('Elengedett Összeg (Ft)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Címek és jelmagyarázat hozzáadása
plt.title("Éves Kedvezmény Kimutatás")
fig.tight_layout()
plt.show()

# Modul: Vásárlások időzítése (héten belüli és napon belüli népszerű időpontok, két év összehasonlítása)

# Szűrés az utolsó két év adataira, ha van adat
latest_years = sorted(df_cleaned['Year'].unique())[-2:] if len(df_cleaned['Year'].unique()) >= 2 else sorted(df_cleaned['Year'].unique())
df_recent = df_cleaned[df_cleaned['Year'].isin(latest_years)].copy()

# Hét napjai és órák kinyerése
df_recent['Weekday'] = df_recent['Order Date'].dt.day_name()  # Hét napjai
df_recent['Hour'] = df_recent['Order Date'].dt.hour  # Órák

# Hét napjai szerint aggregálás év szerint
weekday_revenue = df_recent.groupby(['Year', 'Weekday']).agg({
    'Line Revenue': 'sum'
}).reset_index()

# A hét napjainak sorrendbe állítása
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_revenue['Weekday'] = pd.Categorical(weekday_revenue['Weekday'], categories=weekday_order, ordered=True)
weekday_revenue = weekday_revenue.sort_values(['Year', 'Weekday'])

# Órák szerinti aggregálás év szerint
hourly_revenue = df_recent.groupby(['Year', 'Hour']).agg({
    'Line Revenue': 'sum'
}).reset_index()

print("\033[1;34m--- Faforit vásárlási idő ---\033[0m")
# Vizualizáció: Hét napjai szerint, két év összehasonlítása
plt.figure(figsize=(10, 6))
for year in latest_years:
    data = weekday_revenue[weekday_revenue['Year'] == year]
    plt.plot(data['Weekday'], data['Line Revenue'], marker='o', label=f'{year}')

plt.title('Bevétel a hét napjai szerint (évek összehasonlítása)', fontsize=14)
plt.xlabel('Hét napjai', fontsize=12)
plt.ylabel('Bevétel (Ft)', fontsize=12)
plt.legend(title='Év')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Vizualizáció: Napon belüli órák szerint, két év összehasonlítása
plt.figure(figsize=(12, 6))
for year in latest_years:
    data = hourly_revenue[hourly_revenue['Year'] == year]
    plt.plot(data['Hour'], data['Line Revenue'], marker='o', label=f'{year}')

plt.title('Bevétel a napon belüli órák szerint (évek összehasonlítása)', fontsize=14)
plt.xlabel('Óra', fontsize=12)
plt.ylabel('Bevétel (Ft)', fontsize=12)
plt.xticks(range(0, 24), [f'{h}:00' for h in range(0, 24)], rotation=45)
plt.legend(title='Év')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Eredmények kiírása
print("\n--- Bevétel a hét napjai szerint (évek összehasonlítása) ---")
print(weekday_revenue.to_string(index=False, header=['Év', 'Hét napja', 'Bevétel (Ft)']))

print("\n--- Bevétel a napon belüli órák szerint (évek összehasonlítása) ---")
print(hourly_revenue.to_string(index=False, header=['Év', 'Óra', 'Bevétel (Ft)']))

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Nemek szerinti megoszlás vizsgálata
print("\033[1;34m--- Nem szerini vásárlások ---\033[0m")
if 'Gender (Billing)' in df_cleaned.columns:
    # Összesített nemi megoszlás
    gender_distribution = df_cleaned['Gender (Billing)'].value_counts(normalize=True) * 100
    print("\n--- Vásárlók nemek szerinti megoszlása ---")
    print(gender_distribution.to_string(header=False, float_format="%.2f%%"))

    # Bevétel és rendelésszám nemek szerint
    gender_revenue_orders = df_cleaned.groupby('Gender (Billing)').agg({
        'Order Number': 'nunique',  # Egyedi rendelések száma
        'Line Revenue': 'sum'      # Összesített bevétel
    }).rename(columns={'Order Number': 'Rendelések száma', 'Line Revenue': 'Bevétel (Ft)'})

    print("\n--- Nemek szerinti eredmények ---")
    print(gender_revenue_orders.to_string())

    # Vizualizáció: Bevétel nemek szerint
    plt.figure(figsize=(12, 6))
    gender_revenue_orders['Bevétel (Ft)'].plot(kind='bar', alpha=0.7, color=['blue', 'pink'], label='Bevétel')
    plt.title('Bevétel nemek szerint')
    plt.ylabel('Bevétel (Ft)')
    plt.xlabel('Nem')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Vizualizáció: Rendelések száma nemek szerint
    plt.figure(figsize=(12, 6))
    gender_revenue_orders['Rendelések száma'].plot(kind='bar', alpha=0.7, color=['blue', 'pink'], label='Rendelések száma')
    plt.title('Rendelések száma nemek szerint')
    plt.ylabel('Rendelések száma')
    plt.xlabel('Nem')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("A 'Gender (Billing)' adat nem érhető el az elemzéshez.")

# Origin mező elemzése 2024-től
print("\033[1;34m--- Források ---\033[0m")
if 'Origin' in df_cleaned.columns:
    # Szűrés 2024-től kezdődően
    origin_data = df_cleaned[df_cleaned['Year'] >= 2024].copy()

    # Ellenőrzés, hogy vannak-e megfelelő adatok
    if not origin_data.empty:
        # Origin mező megoszlásának számítása
        origin_distribution = origin_data['Origin'].value_counts(normalize=True) * 100

        # Kiírás az eredményekről a terminálra (minden adat)
        print("\n--- Látogatók érkezési forrásainak megoszlása (2024-től) ---")
        print(origin_distribution.to_string(header=False, float_format="%.2f%%"))

        # 1% alatti források egyesítése csak a tortadiagramhoz
        other_threshold = 1.0
        major_origins = origin_distribution[origin_distribution >= other_threshold]
        other_origins = origin_distribution[origin_distribution < other_threshold]

        # "Egyéb" kategória létrehozása
        if not other_origins.empty:
            major_origins['Egyéb'] = other_origins.sum()

        # Vizualizáció: tortadiagram az "Egyéb" kategóriával
        plt.figure(figsize=(8, 8))
        major_origins.plot(
            kind='pie',
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.tab20.colors[:len(major_origins)],
            labels=major_origins.index
        )
        plt.title('Látogatók érkezési forrásainak megoszlása (2024-től)')
        plt.ylabel('')  # Nem szükséges tengelycím egy tortadiagramnál
        plt.tight_layout()
        plt.show()
    else:
        print("Nincsenek 2024-től származó adatok az 'Origin' mezőhöz.")
else:
    print("Az 'Origin' mező nem érhető el az elemzéshez.")

# Step 1: Feature extraction for clustering
print("\nÜgyfélszegmentálás előkészítése...")
# Aggregating customer data to form features for clustering
print("\033[1;34m--- Vásárlói csoportok  ---\033[0m")

customer_aggregated = df_cleaned.groupby('Email (Billing)').agg({
    'Customer Total Orders': 'sum',  # Az ügyfél összes rendelése
    'Line Revenue': 'sum',  # A tényleges költés (nem az eredeti Customer Total Spent)
    'Customer first order date': 'min',  # Első rendelés dátuma
    'Customer last order date': 'max'    # Utolsó rendelés dátuma
}).reset_index()

# Az oszlopok átnevezése a megfelelő értelmezés érdekében
customer_aggregated.rename(columns={'Line Revenue': 'Total Spent'}, inplace=True)

# Flatten the multi-level columns
customer_aggregated.columns = ['Customer Email', 'Total Orders', 'Total Spent', 'First Order Date', 'Last Order Date']

# Filter customers who have ordered in the last 365 days
customer_aggregated = customer_aggregated[pd.to_datetime('now') - pd.to_datetime(customer_aggregated['Last Order Date']) <= pd.Timedelta(days=365)]

# Calculate additional features
customer_aggregated['Order Recency'] = (pd.to_datetime('now') - pd.to_datetime(customer_aggregated['Last Order Date'])).dt.days

# Calculate order frequency based on the last 365 days
customer_aggregated['Order Frequency'] = customer_aggregated['Total Orders'] / 365

# Step 2: Feature selection and scaling
print("Az ügyfélszegmentálás jellemzőinek előkészítése és normalizálása...")
features = customer_aggregated[['Total Orders', 'Total Spent', 'Order Recency', 'Order Frequency']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Clustering using KMeans
print("KMeans algoritmus használata a szegmentáláshoz...")
# Set an arbitrary number of clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
customer_aggregated['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Analyze and visualize cluster results
print("\n--- Szegmentációs eredmények ---")
cluster_summary = customer_aggregated.groupby('Cluster').agg({
    'Total Orders': 'mean',
    'Total Spent': 'mean',
    'Order Recency': 'mean',
    'Order Frequency': 'mean',
    'Customer Email': 'count'
}).rename(columns={'Customer Email': 'Number of Customers'}).reset_index()

# Print the cluster summary to understand different segments
print(cluster_summary)

# Explanation for the clusters
for _, row in cluster_summary.iterrows():
    print(f"\nKlaszter {int(row['Cluster'])}:")
    print(f"  Átlagos rendelések száma: {row['Total Orders']:.2f}")
    print(f"  Átlagos költés: {row['Total Spent']:.2f} Ft")
    print(f"  Átlagos rendelési gyakoriság: {row['Order Frequency']:.4f} rendelés/nap")
    print(f"  Átlagos frissesség (utolsó rendelés óta eltelt napok): {row['Order Recency']:.1f} nap")
    print(f"  Ügyfelek száma a klaszterben: {int(row['Number of Customers'])}")

# Add textual interpretation of each cluster
print("\n--- Klaszterek értelmezése ---")
for _, row in cluster_summary.iterrows():
    if row['Total Orders'] > 5 and row['Total Spent'] > 50000:
        cluster_type = "Hűséges vásárlók (gyakran visszatérnek és magas költéssel)"
    elif row['Total Spent'] > 100000:
        cluster_type = "Magas értékű vásárlók (VIP szegmens)"
    elif row['Total Orders'] == 1:
        cluster_type = "Új vásárlók (első vásárlás)"
    elif row['Order Recency'] > 180:
        cluster_type = "Inaktív vásárlók (hosszabb ideje nem vásároltak)"
    else:
        cluster_type = "Alkalmi vásárlók (ritkán vásárlók)"

    print(f"Klaszter {int(row['Cluster'])}: {cluster_type}")

# Visualization of clusters based on Total Spent and Order Frequency
print("\nSzegmentáció vizualizáció készítése...")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=customer_aggregated, x='Total Spent', y='Order Frequency', hue='Cluster', palette='tab10', alpha=0.7
)
plt.title('Ügyfélszegmentáció az összköltés és a rendelési gyakoriság alapján')
plt.xlabel('Összköltés (Ft)')
plt.ylabel('Rendelési gyakoriság (rendelés/nap)')
plt.legend(title='Klaszter')
plt.grid(True)
plt.tight_layout()
plt.show()

#New segmentation

# Ügyfélszegmentálás előkészítése
print("\nÜgyfélszegmentálás előkészítése...")

# Aggregating customer data for clustering
customer_aggregated = df_cleaned.groupby('Email (Billing)').agg({
    'Customer Total Orders': 'sum',  # Total number of orders
    'Customer Total Spent': 'sum',  # Total spent
    'Customer first order date': 'min',  # First order date
    'Customer last order date': 'max'   # Last order date
}).reset_index()

customer_aggregated.columns = ['Customer Email', 'Total Orders', 'Total Spent', 'First Order Date', 'Last Order Date']

# Calculate additional features
customer_aggregated['Order Recency'] = (
    pd.to_datetime('now') - pd.to_datetime(customer_aggregated['Last Order Date'])
).dt.days

customer_aggregated['Order Frequency'] = customer_aggregated['Total Orders'] / 365

# Calculate customer lifetime in days
customer_aggregated['Customer Lifetime'] = (
    customer_aggregated['Last Order Date'] - customer_aggregated['First Order Date']
).dt.days

# Step 2: Feature selection and scaling
print("Az ügyfélszegmentálás jellemzőinek előkészítése és normalizálása...")
features = customer_aggregated[['Total Orders', 'Total Spent', 'Order Recency', 'Order Frequency', 'Customer Lifetime']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Clustering using KMeans
print("KMeans algoritmus használata a szegmentáláshoz...")
kmeans = KMeans(n_clusters=4, random_state=42)
customer_aggregated['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Analyze and visualize cluster results
print("\n--- Szegmentációs eredmények ---")
cluster_summary = customer_aggregated.groupby('Cluster').agg({
    'Total Orders': 'mean',
    'Total Spent': 'mean',
    'Order Recency': 'mean',
    'Order Frequency': 'mean',
    'Customer Lifetime': 'mean',  # Include Customer Lifetime in the summary
    'Customer Email': 'count'
}).rename(columns={'Customer Email': 'Number of Customers'}).reset_index()

print(cluster_summary)

# Explanation for the clusters
print("\n--- Klaszterek értelmezése ---")
for _, row in cluster_summary.iterrows():
    if row['Total Orders'] > 5 and row['Total Spent'] > 50000:
        cluster_type = "Hűséges vásárlók (gyakran visszatérnek és magas költéssel)"
    elif row['Total Spent'] > 100000:
        cluster_type = "Magas értékű vásárlók (VIP szegmens)"
    elif row['Total Orders'] == 1:
        cluster_type = "Új vásárlók (első vásárlás)"
    elif row['Order Recency'] > 180:
        cluster_type = "Inaktív vásárlók (hosszabb ideje nem vásároltak)"
    else:
        cluster_type = "Alkalmi vásárlók (ritkán vásárlók)"

    print(f"Klaszter {int(row['Cluster'])}: {cluster_type}")
    print(f"  Átlagos ügyfélélettartam (napokban): {row['Customer Lifetime']:.1f}")

# Visualization of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=customer_aggregated, x='Total Spent', y='Order Frequency', hue='Cluster', palette='tab10', alpha=0.7
)
plt.title('Ügyfélszegmentáció az összköltés és a rendelési gyakoriság alapján')
plt.xlabel('Összköltés (Ft)')
plt.ylabel('Rendelési gyakoriság (rendelés/nap)')
plt.legend(title='Klaszter')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\033[1;34m--- Összegzés, javaslatok ---\033[0m")
# Generate detailed text analysis using OpenAI API if requested
if use_openai_analysis:
    # API kulcs beállítása
    os.environ['OPENAI_API_KEY'] = ''
    openai.api_key = os.environ['OPENAI_API_KEY']

    # 1. Klaszterelemzés prompt
    cluster_prompt = (
        "Az alábbi ügyfélszegmentálási elemzés eredményei alapján kérlek, készíts konkrét üzleti és marketingjavaslatokat "
        "mindegyik klaszterre külön-külön:\n"
        f"{cluster_summary.to_string(index=False)}\n\n"
        "Kérlek, fogalmazz meg specifikus promóciós ötleteket, hűségprogramokat, termékcsomagolási stratégiákat és online marketing stratégiákat."
    )

    try:
        # OpenAI API hívás klaszterelemzéshez
        response_clusters = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in business strategy and customer segmentation."},
                {"role": "user", "content": cluster_prompt}
            ],
            max_tokens=1000
        )
        print("\n--- Klaszterek elemzése ---")
        print("\n".join(textwrap.fill(line, width=90) for line in response_clusters['choices'][0]['message']['content'].split("\n")))
    except Exception as e:
        print(f"Hiba történt a klaszterek elemzése során: {e}")
    # 2. SEO cikkek prompt
    seo_prompt = (
        "Az adatelemzés eredményei alapján kérlek, javasolj SEO szempontú cikkcímeket és témákat. "
        "Az elemzés eredményei:\n\n"
        f"Értékesítési trendek havi bontásban:\n{monthly_revenue_pivot_formatted.to_string()}\n\n"
        f"Legnépszerűbb termékek:\n{top_products[['Item Name', 'Line Revenue']].to_string(index=False)}\n\n"
        f"Legnépszerűbb kategóriák:\n{top_categories[['Last Category', 'Line Revenue']].to_string(index=False)}\n\n"
        f"Készletkezelési problémák:\n{inventory_risk_summary}\n\n"
        "Olyan cikkeket szeretnék, amelyek növelik az organikus forgalmat és kapcsolódnak az e-kereskedelemhez."
    )

    try:
        # OpenAI API hívás SEO cikkekhez
        response_seo = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in SEO content strategy for e-commerce."},
                {"role": "user", "content": seo_prompt}
            ],
            max_tokens=1000
        )
        print("\n--- SEO cikkjavaslatok ---")
        print("\n".join(textwrap.fill(line, width=90) for line in response_seo['choices'][0]['message']['content'].split("\n")))
    except Exception as e:
        print(f"Hiba történt az SEO cikkjavaslatok generálása során: {e}")

    # 3. Általános elemzés prompt
    # Az elemzésből származó kulcsinformációk összegzése
    popular_products_summary = (
        f"Népszerű termékek:\n{top_products[['Item Name', 'Line Revenue']].to_string(index=False)}\n\n"
        if not top_products.empty else "Nincsenek népszerű termékek."
    )

    popular_categories_summary = (
        f"Népszerű kategóriák:\n{top_categories[['Last Category', 'Line Revenue']].to_string(index=False)}\n\n"
        if not top_categories.empty else "Nincsenek népszerű kategóriák."
    )

    inventory_risk_summary = (
        f"Rizikós készlet: {len(low_stock_risk_items)} termék.\n"
        f"Feleslegesen magas készlet: {len(excess_stock_risk_items)} termék.\n"
        if not low_stock_risk_items.empty or not excess_stock_risk_items.empty else
        "Nincs jelentős rizikós vagy feleslegesen magas készlet."
    )

    general_prompt = (
        "Az alábbi adatelemzés eredményeit összefoglaltuk. Kérlek, készíts átfogó elemzést, és javasolj konkrét lépéseket "
        "az értékesítés támogatására:\n\n"
        f"Értékesítési trendek havi bontásban:\n{monthly_revenue_pivot_formatted.to_string()}\n\n"
        f"Átlagos kosárérték havi bontásban:\n{aov_pivot.to_string()}\n\n"
        f"{popular_products_summary}"
        f"{popular_categories_summary}"
        f"Regionális értékesítési elemzés:\n{regional_revenue[['county', 'Line Revenue']].groupby('county').sum().to_string()}\n\n"
        f"Készletkezelési eredmények:\n{inventory_risk_summary}\n\n"
        "Kérlek, javasolj értékesítési stratégiákat, termékpromóciókat, valamint olyan területeket, amelyeken javítani lehetne."
    )

    try:
        # OpenAI API hívás általános elemzéshez
        response_general = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in e-commerce sales strategy and inventory management."},
                {"role": "user", "content": general_prompt}
            ],
            max_tokens=1500
        )
        print("\n--- Általános elemzés és javaslatok ---")
        print("\n".join(textwrap.fill(line, width=90) for line in response_general['choices'][0]['message']['content'].split("\n")))
    except Exception as e:
        print(f"Hiba történt az általános elemzés során: {e}")

print("Az elemzések és vizualizációk sikeresen végrehajtva.")