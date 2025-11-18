# Test Trained Model with New Input
# Load model and make predictions on new data

import pickle
import pandas as pd
import numpy as np

# ============================================
# STEP 1: LOAD TRAINED MODEL AND ARTIFACTS
# ============================================

print("📦 Loading trained model and artifacts...")

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
print("  ✅ Model loaded")

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("  ✅ Scaler loaded")

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
print("  ✅ Feature names loaded")
print(f"     Expected features: {len(feature_names)}")

# Load label encoders (if they exist)
try:
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    print("  ✅ Label encoders loaded")
except FileNotFoundError:
    label_encoders = {}
    print("  ⚠️  No label encoders found")

print("\n" + "="*50)

# ============================================
# STEP 2: PREPARE NEW INPUT
# ============================================

print("\n🎯 Choose input method:")
print("1. Manual input (enter values one by one)")
print("2. CSV file (test multiple samples)")
print("3. Single row dictionary (for quick testing)")

choice = input("\nEnter choice (1/2/3): ").strip()

if choice == "1":
    # ========== MANUAL INPUT ==========
    print("\n📝 Enter values for each feature:")
    print("(Press Enter to skip optional features)")
    
    input_data = {}
    for feature in feature_names:
        value = input(f"  {feature}: ").strip()
        
        if value == "":
            input_data[feature] = None
        else:
            # Try to convert to number if possible
            try:
                input_data[feature] = float(value)
            except ValueError:
                input_data[feature] = value
    
    # Create DataFrame
    new_data = pd.DataFrame([input_data])

elif choice == "2":
    # ========== CSV FILE INPUT ==========
    csv_path = input("\n📁 Enter path to CSV file: ").strip()
    new_data = pd.read_csv(csv_path)
    print(f"\n✅ Loaded {len(new_data)} samples from CSV")
    print("\nFirst few rows:")
    print(new_data.head())

elif choice == "3":
    # ========== DICTIONARY INPUT (QUICK TEST) ==========
    print("\n⚡ Quick test mode")
    print("Modify the example below in the code:\n")
    
    # MODIFY THIS: Example input dictionary
    example_input = {
        'feature1': 5.1,
        'feature2': 3.5,
        'feature3': 1.4,
        'feature4': 0.2,
        # Add more features as needed
    }
    
    print(f"Using example: {example_input}")
    new_data = pd.DataFrame([example_input])

else:
    print("❌ Invalid choice!")
    exit()

# ============================================
# STEP 3: PREPROCESS NEW INPUT
# ============================================

print("\n🔧 Preprocessing input...")

# Ensure all required features are present
missing_features = set(feature_names) - set(new_data.columns)
if missing_features:
    print(f"⚠️  Warning: Missing features: {missing_features}")
    for feat in missing_features:
        new_data[feat] = 0  # Fill with default value

# Reorder columns to match training data
new_data = new_data[feature_names]

# Handle missing values in new data
for col in new_data.columns:
    if new_data[col].isnull().any():
        if new_data[col].dtype in [np.float64, np.int64]:
            new_data[col].fillna(0, inplace=True)
        else:
            new_data[col].fillna('unknown', inplace=True)

# Apply label encoding to categorical features
for col, encoder in label_encoders.items():
    if col in new_data.columns:
        try:
            new_data[col] = encoder.transform(new_data[col].astype(str))
        except ValueError as e:
            print(f"  ⚠️  Warning: Unknown category in {col}, using default")
            # Handle unknown categories by assigning to first class
            new_data[col] = 0

print("  ✅ Preprocessing complete")

# ============================================
# STEP 4: MAKE PREDICTIONS
# ============================================

print("\n🔮 Making predictions...\n")

# Scale features
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)

# Get prediction probabilities (if available)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(new_data_scaled)
    has_proba = True
else:
    has_proba = False

# ============================================
# STEP 5: DISPLAY RESULTS
# ============================================

print("="*50)
print("📊 PREDICTION RESULTS")
print("="*50)

for i in range(len(new_data)):
    print(f"\nSample {i+1}:")
    print("-" * 30)
    
    # Show input features
    print("Input:")
    for feature in feature_names[:5]:  # Show first 5 features
        print(f"  {feature}: {new_data.iloc[i][feature]}")
    if len(feature_names) > 5:
        print(f"  ... and {len(feature_names)-5} more features")
    
    # Show prediction
    print(f"\n🎯 Predicted Class: {predictions[i]}")
    
    # Show probabilities if available
    if has_proba:
        print(f"\n📊 Prediction Probabilities:")
        for class_idx, prob in enumerate(probabilities[i]):
            print(f"  Class {class_idx}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Show confidence
        confidence = np.max(probabilities[i])
        print(f"\n✨ Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print()

# ============================================
# STEP 6: SAVE PREDICTIONS (OPTIONAL)
# ============================================

save_results = input("\n💾 Save predictions to file? (y/n): ").strip().lower()

if save_results == 'y':
    # Create results dataframe
    results = new_data.copy()
    results['prediction'] = predictions
    
    if has_proba:
        for class_idx in range(probabilities.shape[1]):
            results[f'probability_class_{class_idx}'] = probabilities[:, class_idx]
    
    # Save to CSV
    output_file = 'predictions.csv'
    results.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to '{output_file}'")

print("\n" + "="*50)
print("✅ Testing complete!")
print("="*50)

# ============================================
# BONUS: BATCH PREDICTION FUNCTION
# ============================================

def predict_new_sample(sample_dict):
    """
    Quick function to predict a single sample
    
    Usage:
        result = predict_new_sample({
            'feature1': 5.1,
            'feature2': 3.5,
            ...
        })
    """
    df = pd.DataFrame([sample_dict])
    df = df[feature_names]
    
    # Apply preprocessing
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except:
                df[col] = 0
    
    # Scale and predict
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    
    if has_proba:
        proba = model.predict_proba(df_scaled)[0]
        return {
            'prediction': prediction,
            'probabilities': proba,
            'confidence': np.max(proba)
        }
    else:
        return {'prediction': prediction}

print("\n💡 Tip: You can also use the predict_new_sample() function")
print("   for quick predictions in Python:")
print("   result = predict_new_sample({'feature1': 5.1, 'feature2': 3.5, ...})")
