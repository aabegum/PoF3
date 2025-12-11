"""
create_simple_cof.py - Create Simple CoF (Consequence of Failure) Proxy

Creates basic CoF scores based on equipment type and voltage level
Until real CoF data (replacement costs, customer impact) is available.
"""

import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
INTER_DIR = ROOT / "data" / "ara_ciktilar"
RESULTS_DIR = ROOT / "data" / "sonuclar"

# Load equipment data
equipment = pd.read_csv(INTER_DIR / "equipment_master.csv", encoding="utf-8-sig")

print(f"Loaded {len(equipment):,} equipment")

# Simple CoF proxy based on equipment type and voltage
# Real CoF should come from asset registry with actual costs

COF_BY_TYPE = {
    "Trafo": 50000,      # Transformers: high replacement cost
    "Ayırıcı": 15000,    # Disconnectors: medium cost
    "Sigorta": 5000,     # Fuses: low cost
    "Pano": 25000,       # Panels: medium-high cost
    "Hat": 10000,        # Lines: medium cost (per km)
    "Direk": 3000,       # Poles: low cost
    "Other": 8000,       # Others: baseline
}

# Voltage multiplier (higher voltage = higher consequence)
def get_voltage_multiplier(voltage_str):
    """Extract voltage and apply multiplier."""
    if pd.isna(voltage_str):
        return 1.0
    
    voltage_str = str(voltage_str).lower()
    
    # High voltage (>10kV)
    if any(v in voltage_str for v in ['34.5', '33', '34', '36']):
        return 2.0
    # Medium voltage (>1kV)
    elif any(v in voltage_str for v in ['10', '11', '12', '15']):
        return 1.5
    # Low voltage (<1kV)
    elif any(v in voltage_str for v in ['400', '0.4', '380']):
        return 1.0
    
    return 1.0  # Default

# Calculate CoF
equipment['CoF_Base'] = equipment['Ekipman_Tipi'].map(COF_BY_TYPE).fillna(8000)

if 'Gerilim_Seviyesi' in equipment.columns:
    equipment['Voltage_Multiplier'] = equipment['Gerilim_Seviyesi'].apply(get_voltage_multiplier)
else:
    equipment['Voltage_Multiplier'] = 1.0

equipment['CoF'] = equipment['CoF_Base'] * equipment['Voltage_Multiplier']

# Add random variation (±20%) to simulate real variability
import numpy as np
np.random.seed(42)
equipment['CoF'] = equipment['CoF'] * (1 + np.random.uniform(-0.2, 0.2, len(equipment)))

# Round to nearest 100 TL
equipment['CoF'] = (equipment['CoF'] / 100).round() * 100

# Save
output_df = equipment[['cbs_id', 'CoF']].copy()
output_path = RESULTS_DIR / "cof_pof3.csv"
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✓ Created simple CoF file: {output_path}")
print(f"\nCoF Statistics:")
print(f"  Mean: {equipment['CoF'].mean():,.0f} TL")
print(f"  Median: {equipment['CoF'].median():,.0f} TL")
print(f"  Min: {equipment['CoF'].min():,.0f} TL")
print(f"  Max: {equipment['CoF'].max():,.0f} TL")
print(f"\nCoF by Equipment Type:")
print(equipment.groupby('Ekipman_Tipi')['CoF'].agg(['count', 'mean']).round(0))

print("\n⚠️  WARNING: This is a SIMPLIFIED CoF proxy!")
print("For production use, CoF should include:")
print("  - Actual replacement costs from asset registry")
print("  - Customer impact (number of customers affected)")
print("  - Critical infrastructure flags")
print("  - Service interruption costs")
print("  - Safety consequences")