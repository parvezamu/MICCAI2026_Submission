import numpy as np
'''
# Load one file
data = np.load('/home/pahm409/preprocessed_isles_dual_modality/sub-strokecase0001.npz')

print("Keys:", list(data.keys()))

if 'adc' in data.keys():
    adc = data['adc']
    print(f"ADC shape: {adc.shape}")
    print(f"ADC min: {adc.min()}")
    print(f"ADC max: {adc.max()}")
    print(f"ADC mean: {adc.mean()}")
    print(f"ADC std: {adc.std()}")
    print(f"ADC nonzero: {(adc != 0).sum()} / {adc.size}")
    
    if adc.max() == adc.min():
        print("PROBLEM: ADC is constant!")
else:
    print("PROBLEM: No ADC key!")
'''


import numpy as np

# Load a preprocessed file
data = np.load('/home/pahm409/preprocessed_isles_dual_v2/sub-strokecase0001.npz')

dwi = data['dwi']
adc = data['adc']
mask = data['mask']

print("DWI:")
print(f"  Shape: {dwi.shape}")
print(f"  Range: [{dwi.min():.6f}, {dwi.max():.6f}]")
print(f"  Nonzero count: {(dwi != 0).sum()}")

print("\nADC:")
print(f"  Shape: {adc.shape}")
print(f"  Range: [{adc.min():.6f}, {adc.max():.6f}]")
print(f"  Nonzero count: {(adc != 0).sum()}")

print("\nMask:")
print(f"  Positive count: {(mask > 0).sum()}")

# CRITICAL CHECK
if np.array_equal(dwi, adc):
    print("\n❌ BUG: DWI and ADC are IDENTICAL!")
else:
    print("\n✓ DWI and ADC are different")

# Show first 20 values
dwi_flat = dwi[dwi != 0][:20]
adc_flat = adc[adc != 0][:20]
print(f"\nFirst 20 DWI values: {dwi_flat}")
print(f"First 20 ADC values: {adc_flat}")
