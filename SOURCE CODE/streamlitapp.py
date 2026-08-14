#importing all libraries 
#----------------------- 
import streamlit as st 
import joblib 
import numpy as np 
import os 
from datetime import date 
 
# Page settings 
#--------------- 
st.set_page_config(page_title="HOUSE PRICE PREDICTION", page_icon="🏠", layout="centered") 

if "page" not in st.session_state:
    st.session_state.page = "welcome"

# Welcome Page
#-------------
if st.session_state.page == "welcome":
    st.markdown("")  # Space
    
    # Title section with emoji
    col1, col2 = st.columns([0.2, 10])
    with col1:
        st.write("")
    with col2:
        st.title(" Welcome to House Price Predictor")

    st.subheader("Powered by Genetic Algorithm", divider="blue")
    
    st.write("")
    st.write("")
    
    # Info section
    st.write("### 📋 What this does:")
    
    col1, col2 = st.columns([0.1, 0.9])
    with col2:
        st.write("⟡ Enter house features (bedrooms, area, floors, etc.)")
        st.write("⟡ AI predicts the price instantly.")
        st.write("⟡ Simple & easy to use. ")
    
    st.write("")
    st.write("")
    
    # Features highlight
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🤖 Model", value="RF+GA")
    with col2:
        st.metric(label="📊 Dataset", value="Indian Houses")
    with col3:
        st.metric(label="⚡ Speed", value="Instant")
    
    st.write("")
    st.write("")
    st.divider()
    st.write("")
    
    # Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Predicting", use_container_width=True, key="start_btn"):
            st.session_state.page = "predictor"
            st.rerun()
    
    st.stop()

# Predictor Page 
#---------------------------------------------------------
if st.session_state.page == "predictor":
    
    # Back button
    if st.button("← Back to Welcome"):
        st.session_state.page = "welcome"
        st.rerun()

    # Model path 
    #----------- 
    baselocation = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    modelpath = os.path.join(baselocation, "MODELS", "house_price_ga_random_forest.joblib") 
     
    # Load saved model
    #----------------- 
    modelpackage = joblib.load(modelpath) 
    model = modelpackage["model"] 
    scaler = modelpackage["scaler"] 
    selected_features = modelpackage["selected_features"] 
    selected_feature_names = modelpackage["selected_feature_names"] 
    all_feature_names = modelpackage["all_feature_names"] 
     
    # Title 
    #------ 
    st.title("🏠 HOUSE PRICE PREDICTION") 
    st.write("This application predicts house prices using a Random Forest Regressor optimized with a Genetic Algorithm. Delivers precise valuations by analyzing only the most impactful property characteristics.") 
     
    # Input section 
    #-------------- 
    st.header("Enter House Information") 
    st.write("Please enter valid house details:", "\n_________________________________") 
     
    inputvalues = [] 
     
    # Create input fields 
    #-------------------- 
    for feature in all_feature_names: 

        # Features not used by GA
        #------------------------
        if feature == "Distance from the airport":
            inputvalues.append(0.0)
            continue

        if feature == "Number of schools nearby":
            inputvalues.append(0.0)
            continue

        if feature == "number of bathrooms":
            inputvalues.append(0.0)
            continue

        if feature == "Area of the basement":
            inputvalues.append(0.0)
            continue

        if feature == "Renovation Year":
            inputvalues.append(0.0)
            continue   
        #------------------------------------------------------------------ 
        if feature == "Date": 
            value = st.date_input("Select Date") 
     
            # Epoch date used in the dataset 
            epochdate = date(1899, 12, 30) 
     
            # Calculate number of days from epoch 
            dayssinceepoch = (value - epochdate).days 
     
            inputvalues.append(dayssinceepoch) 
        #------------------------------------------------------------------- 
        elif feature == "number of bedrooms": 
            value = st.selectbox("Number of Bedrooms", list(range(0, 21))) 
     
            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "living area": 
            value = st.text_input("Living Area (Square Feet)", value="0.00") 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Living Area cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "living_area_renov": 
            value = st.text_input("Renovated Living Area-Square Feet", value="0.00") 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Living Area cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "lot_area_renov": 
            value = st.text_input("Renovated Lot Area-Square Feet", value="0.00") 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Area cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "Area of the house(excluding basement)": 
            value = st.text_input(
                "Area of the house(excluding basement)-Square Feet", 
                value="0.00"
            ) 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Area cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "lot area": 
            value = st.text_input("Lot Area (Square Feet)", value="0.00") 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Lot Area cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "Built Year": 
            value = st.text_input("Built Year", value="1900") 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Build year cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "Postal Code": 
            value = st.text_input("Postal Code", value="000000") 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Postal code cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "number of floors": 
            value = st.selectbox("Number of Floors", list(range(0, 11))) 
            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "waterfront present": 
            value = st.selectbox("Waterfront Present", ["Yes", "No"]) 

            if value == "Yes": 
                inputvalues.append(1) 
            else: 
                inputvalues.append(0) 
        #------------------------------------------------------------------- 
        elif feature == "number of views": 
            value = st.selectbox("Number of Views", list(range(0, 11))) 
            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "condition of the house": 
            value = st.selectbox("Condition of the house", list(range(0, 11))) 
            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "grade of the house": 
            value = st.selectbox("Grade of the house", list(range(0, 16))) 
            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        elif feature == "Longitude": 
            value = st.text_input("Longitude", value="0.00") 

            try: 
                value = float(value) 

            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 
        # All other features 
        else: 
            value = st.text_input(feature, value="0.0") 

            try: 
                value = float(value) 
     
                if value < 0: 
                    st.error("Lot Area cannot be negative.") 
                    value = 0.0 
     
            except ValueError: 
                st.error("Please enter correct value.") 
                value = 0.0 

            inputvalues.append(value) 
        #------------------------------------------------------------------- 

    # Prediction button
    if st.button("Predict House Price"): 
     
        # Convert inputs into NumPy array 
        input_array = np.array(inputvalues, dtype=float).reshape(1, -1) 
     
        # Scale the input 
        input_scaled = scaler.transform(input_array) 
     
        # Select features chosen by GA 
        input_selected = input_scaled[:, selected_features] 
     
        # Make prediction 
        prediction = model.predict(input_selected) 
     
        # Display predicted price 
        st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")