import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Dental Clinic Price Predictor", layout="wide", page_icon="🦷")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        color: #2c3e50;
        padding: 1rem 0;
    }
    .metric-card {
        background: black;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #e9f5ff;
    }
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel('dentalprice-100-pilot.xlsx')
    
    # Clean and preprocess data
    df['Clinic-Age'] = pd.to_numeric(df['Clinic-Age'], errors='coerce').fillna(df['Clinic-Age'].median())
    df['google review rating'] = pd.to_numeric(df['google review rating'], errors='coerce').fillna(df['google review rating'].median())
    
    # Convert yes/no to binary
    binary_cols = ['emergency attending (yes/no)', 'drinking water available(yes/no)', 
                   'Is pharma with in(y/n)', 'are dental accessories sold(y/n)', 
                   'LED display board of clinic(y/n)', 'air conditioned(y/n)', 
                   'website(y/n)', 'FB page(y/n)', 'Linked page (y/n)', 
                   'youtube channel(y/n)', 'sms mktg', 'FM radio mktg(y/n)', 
                   'mostly visit by appointment (y/n)']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).fillna(0)
    
    # Convert categorical variables
    if 'owned/ rental/leased' in df.columns:
        df['owned/ rental/leased'] = df['owned/ rental/leased'].astype(str).str.lower().map({
            'owned': 0, 'rental': 1, 'leased': 2}).fillna(0)
    
    if 'accessability from main road(good/average/poor)' in df.columns:
        df['accessability from main road(good/average/poor)'] = df['accessability from main road(good/average/poor)'].astype(str).str.lower().map({
            'good': 0, 'average': 1, 'poor': 2}).fillna(0)
    
    # Parking availability
    if 'parking(no parking/no. of two wheeler/no. of four wheeler)' in df.columns:
        df['parking'] = df['parking(no parking/no. of two wheeler/no. of four wheeler)'].str.contains('available', case=False).astype(int)
    
    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

df = load_data()

# Create a mapping of addresses to clinic data
@st.cache_data
def create_address_mapping(df):
    address_map = {}
    for _, row in df.iterrows():
        address = row.get('address', 'Unknown Address')
        address_map[address] = {
            'Clinic-Age': row.get('Clinic-Age', df['Clinic-Age'].median()),
            'owned/ rental/leased': row.get('owned/ rental/leased', 0),
            'no. of floors occupied': row.get('no. of floors occupied', 1),
            'no. of dental chairs': row.get('no. of dental chairs', 2),
            'no.of dental x ray machines': row.get('no.of dental x ray machines', 1),
            'no. of LCD/TV s in clinic': row.get('no. of LCD/TV s in clinic', 1),
            'no.of fans': row.get('no.of fans', 2),
            'no. of female doctors': row.get('no. of female doctors', 2),
            'no. of female staff': row.get('no. of female staff', 2),
            'google review rating': row.get('google review rating', df['google review rating'].median()),
            'parking': row.get('parking', 1),
            'accessability from main road(good/average/poor)': row.get('accessability from main road(good/average/poor)', 0),
            'emergency attending (yes/no)': row.get('emergency attending (yes/no)', 1),
            'air conditioned(y/n)': row.get('air conditioned(y/n)', 1),
            'mostly visit by appointment (y/n)': row.get('mostly visit by appointment (y/n)', 1),
            'clinic name': row.get('clinic name', 'Unknown Clinic')
        }
    return address_map

address_map = create_address_mapping(df)

# Train models for each procedure
def train_models(df):
    # Features to use for prediction
    features = [
        'Clinic-Age', 'owned/ rental/leased', 'no. of floors occupied', 
        'no. of dental chairs', 'no.of dental x ray machines', 
        'no. of LCD/TV s in clinic', 'no.of fans', 'no. of female doctors', 
        'no. of female staff', 'google review rating', 'parking',
        'accessability from main road(good/average/poor)', 
        'emergency attending (yes/no)', 'air conditioned(y/n)',
        'mostly visit by appointment (y/n)'
    ]
    
    # Ensure features exist in dataframe
    features = [f for f in features if f in df.columns]
    
    # Target variables
    targets = {
        'consultancy charges': 'Consultation',
        'Scaling charges': 'Scaling',
        'Filling charges': 'Filling',
        'Wisdom tooth extraction': 'Wisdom Tooth Extraction',
        'RCT (without cap) charges': 'Root Canal Treatment'
    }
    
    models = {}
    
    for target_col, target_name in targets.items():
        if target_col not in df.columns:
            continue
            
        # Prepare data
        X = df[features]
        y = df[target_col]
        
        # Remove rows with missing target values
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            continue
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            models[target_name] = {
                'model': model,
                'mae': mae,
                'features': features,
                'target_col': target_col
            }
        except Exception as e:
            st.warning(f"Could not train model for {target_name}: {str(e)}")
            continue
    
    return models

models = train_models(df)

# Header
st.markdown("<h1 class='header'>🦷 Dental Clinic Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<p style="color: #555; font-size: 1.1rem;">
Predict prices for common dental procedures based on clinic characteristics in Bangalore.
</p>
""", unsafe_allow_html=True)

# Sidebar with clinic information
with st.sidebar:
    st.markdown("<h2 style='color: #2c3e50;'>Clinic Information</h2>", unsafe_allow_html=True)
    
    # Location section
    st.markdown("### Location Details")
    address_options = list(address_map.keys())
    selected_address = st.selectbox("Select Clinic Address", address_options)
    
    # Get default values from selected address
    clinic_data = address_map.get(selected_address, {})
    
    # Basic info section
    st.markdown("### Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        clinic_age = st.slider('Clinic Age (years)', 1, 50, int(clinic_data.get('Clinic-Age', 10)))
    with col2:
        ownership = st.selectbox('Ownership', ['Owned', 'Rental', 'Leased'], 
                               index=int(clinic_data.get('owned/ rental/leased', 1)))
    
    # Facility info section
    st.markdown("### Facility Information")
    col1, col2 = st.columns(2)
    with col1:
        floors = st.slider('Floors occupied', 1, 3, int(clinic_data.get('no. of floors occupied', 1)))
        chairs = st.slider('Dental chairs', 1, 5, int(clinic_data.get('no. of dental chairs', 2)))
        xray = st.slider('X-ray machines', 0, 2, int(clinic_data.get('no.of dental x ray machines', 1)))
    with col2:
        tvs = st.slider('LCD/TVs in clinic', 0, 10, int(clinic_data.get('no. of LCD/TV s in clinic', 1)))
        fans = st.slider('Number of fans', 0, 10, int(clinic_data.get('no.of fans', 2)))
    
    # Staff info section
    st.markdown("### Staff Information")
    col1, col2 = st.columns(2)
    with col1:
        female_docs = st.slider('Female doctors', 0, 10, int(clinic_data.get('no. of female doctors', 2)))
    with col2:
        female_staff = st.slider('Female staff', 0, 10, int(clinic_data.get('no. of female staff', 2)))
    
    # Ratings and amenities
    st.markdown("### Ratings & Amenities")
    rating = st.slider('Google rating (1-5)', 1.0, 5.0, float(clinic_data.get('google review rating', 4.5)), 0.1)
    access = st.selectbox('Road accessibility', ['Good', 'Average', 'Poor'], 
                         index=int(clinic_data.get('accessability from main road(good/average/poor)', 0)))
    
    col1, col2 = st.columns(2)
    with col1:
        parking = st.checkbox('Parking available', value=bool(clinic_data.get('parking', True)))
        emergency = st.checkbox('Emergency services', value=bool(clinic_data.get('emergency attending (yes/no)', True)))
    with col2:
        ac = st.checkbox('Air conditioned', value=bool(clinic_data.get('air conditioned(y/n)', True)))
        appointment = st.checkbox('Appointment preferred', value=bool(clinic_data.get('mostly visit by appointment (y/n)', True)))

# Map inputs to model features
ownership_map = {'Owned': 0, 'Rental': 1, 'Leased': 2}
access_map = {'Good': 0, 'Average': 1, 'Poor': 2}

input_data = {
    'Clinic-Age': clinic_age,
    'owned/ rental/leased': ownership_map[ownership],
    'no. of floors occupied': floors,
    'no. of dental chairs': chairs,
    'no.of dental x ray machines': xray,
    'no. of LCD/TV s in clinic': tvs,
    'no.of fans': fans,
    'no. of female doctors': female_docs,
    'no. of female staff': female_staff,
    'google review rating': rating,
    'parking': 1 if parking else 0,
    'accessability from main road(good/average/poor)': access_map[access],
    'emergency attending (yes/no)': 1 if emergency else 0,
    'air conditioned(y/n)': 1 if ac else 0,
    'mostly visit by appointment (y/n)': 1 if appointment else 0
}

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Make predictions
with st.container():
    st.markdown("<h2 style='color: #2c3e50;'>Predicted Prices</h2>", unsafe_allow_html=True)
    clinic_name = address_map.get(selected_address, {}).get('clinic name', 'Selected Clinic')
    st.markdown(f"<p style='color: #555;'>For <strong>{clinic_name}</strong> at: <strong>{selected_address}</strong></p>", unsafe_allow_html=True)
    
    if not models:
        st.error("No models were successfully trained. Please check your data.")
    else:
        predictions = {}
        for proc, model_info in models.items():
            try:
                model = model_info['model']
                pred = model.predict(input_df)[0]
                predictions[proc] = {
                    'price': round(pred),
                    'mae': round(model_info['mae'])
                }
            except Exception as e:
                st.warning(f"Could not make prediction for {proc}: {str(e)}")
                continue
        
        # Display predictions in cards
        cols = st.columns(len(predictions))
        for idx, (proc, pred_info) in enumerate(predictions.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{proc}</h3>
                    <h2>₹{pred_info['price']:,}</h2>
                    <p style="color: #666;">Accuracy: ±₹{pred_info['mae']:,}</p>
                </div>
                """, unsafe_allow_html=True)

# Data insights section
with st.container():
    st.markdown("<h2 style='color: #2c3e50;'>Market Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #2c3e50;'>Average Prices</h3>", unsafe_allow_html=True)
        try:
            avg_prices = df[['consultancy charges', 'Scaling charges', 'Filling charges', 
                           'Wisdom tooth extraction', 'RCT (without cap) charges']].mean().round()
            
            avg_prices = avg_prices.rename({
                'consultancy charges': 'Consultation',
                'Scaling charges': 'Scaling',
                'Filling charges': 'Filling',
                'Wisdom tooth extraction': 'Wisdom Tooth Extraction',
                'RCT (without cap) charges': 'Root Canal Treatment'
            })
            
            st.table(avg_prices.rename('Average Price (₹)').reset_index().rename(columns={'index': 'Procedure'}))
        except Exception as e:
            st.error(f"Could not calculate average prices: {str(e)}")
    
    with col2:
        st.markdown("<h3 style='color: #2c3e50;'>Price Influencers</h3>", unsafe_allow_html=True)
        procedure = st.selectbox('Select procedure', list(models.keys()))
        
        try:
            model = models[procedure]['model']
            features = models[procedure]['features']
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
            
            st.bar_chart(importance_df.set_index('Feature')['Importance'])
        except Exception as e:
            st.error(f"Could not show feature importance: {str(e)}")

# Raw data option
with st.expander("Show raw data and statistics"):
    st.markdown("<h3 style='color: #2c3e50;'>Dataset Overview</h3>", unsafe_allow_html=True)
    st.write(f"Using data from {len(df)} dental clinics in Bangalore")
    
    if st.checkbox('Show raw data'):
        st.dataframe(df)
    
    if st.checkbox('Show statistics'):
        st.write(df.describe())