from flask import Flask, render_template, request
import pandas as pd
from pycaret.classification import load_model
from pathlib import Path
app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def hotel():
    return render_template('hotel.html')

@app.route('/hotel-predict', methods=['POST'])
def predict():
    columns = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included', 'has_availability', 'instant_bookable',
    'maximum_nights', 'bed_type_Airbed', 'bed_type_Couch', 'bed_type_Futon', 'bed_type_Pull-out Sofa',
    'bed_type_Real Bed', 'cancellation_policy_flexible', 'cancellation_policy_moderate', 'cancellation_policy_no_refunds',
    'cancellation_policy_strict', 'cancellation_policy_super_strict_30', 'property_type_Apartment',
    'property_type_Bed & Breakfast', 'property_type_Boat', 'property_type_Bungalow', 'property_type_Cabin',
    'property_type_Camper/RV', 'property_type_Chalet', 'property_type_Condominium', 'property_type_Earth House',
    'property_type_House', 'property_type_Hut', 'property_type_Loft', 'property_type_Other', 'property_type_Tent',
    'property_type_Tipi', 'property_type_Townhouse', 'property_type_Treehouse', 'property_type_Villa',
    'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'amenities_Buzzer/Wireless Intercom',
    'amenities_Shampoo', 'amenities_Kitchen', 'amenities_Gym', 'amenities_Hair Dryer', 'amenities_Essentials',
    'amenities_Cat(s)', 'amenities_Heating', 'amenities_24-Hour Check-in', 'amenities_Wireless Internet',
    'amenities_Breakfast', 'amenities_Laptop Friendly Workspace', 'amenities_First Aid Kit', 'amenities_Indoor Fireplace',
    'amenities_Wheelchair Accessible', 'amenities_Family/Kid Friendly', 'amenities_Carbon Monoxide Detector',
    'amenities_Pets Allowed', 'amenities_Washer / Dryer', 'amenities_Hot Tub', 'amenities_Doorman',
    'amenities_Lock on Bedroom Door', 'amenities_Safety Card', 'amenities_Cable TV', 'amenities_Other pet(s)',
    'amenities_Suitable for Events', 'amenities_Pets live on this property', 'amenities_Iron', 'amenities_Dryer',
    'amenities_Washer', 'amenities_Free Parking on Premises', 'amenities_Air Conditioning', 'amenities_Hangers',
    'amenities_Pool', 'amenities_Smoke Detector', 'amenities_TV', 'amenities_Smoking Allowed', 'amenities_Dog(s)',
    'amenities_Elevator in Building', 'amenities_Fire Extinguisher', 'amenities_Internet'
    ]
    df = pd.DataFrame(columns=columns)

    # Retrieve form data
    accommodates = int(request.form['accommodates'])
    bathrooms = int(request.form['bathrooms'])
    bedrooms = int(request.form['bedrooms'])
    beds = int(request.form['beds'])
    # Convert checkbox values to numerical inputs
    guests_included = 1 if 'guests_included' in request.form else 0
    has_availability = 1 if 'has_availability' in request.form else 0
    instant_bookable = 1 if 'instant_bookable' in request.form else 0
    maximum_nights = int(request.form['maximum_nights'])
    # Convert room type, property type, bed type, and cancellation policy to arrays
    # Room types array
    room_types = [
        1 if request.form.get('room_type') == t else 0 for t in ['Entire home/apt', 'Private room', 'Shared room']
    ]

    # Property types array
    property_types = [
        1 if request.form.get('property_type') == t else 0 for t in ['Apartment', 'Bed & Breakfast', 'Boat', 'Bungalow', 'Cabin', 'Camper/RV', 'Chalet', 'Condominium', 'Earth House', 'House', 'Hut', 'Loft', 'Other', 'Tent', 'Tipi', 'Townhouse', 'Treehouse', 'Villa']
    ]

    # Bed types array
    bed_types = [
        1 if request.form.get('bed_type') == t else 0 for t in ['Airbed', 'Couch', 'Futon', 'Pull-out Sofa', 'Real Bed']
    ]

    # Cancellation policies array
    cancellation_policies = [
        1 if request.form.get('cancellation_policy') == t else 0 for t in ['flexible', 'moderate', 'no_refunds', 'Strict', 'Super Strict']
    ]

     # Get all amenities options
    amenities_options = [
        "Air Conditioning", "Lock on Bedroom Door", "Laptop Friendly Workspace",
        "Hair Dryer", "Dog(s)", "Kitchen", "Other pet(s)", "Pets Allowed",
        "Iron", "Shampoo", "Buzzer/Wireless Intercom", "Smoke Detector",
        "Washer", "24-Hour Check-in", "Free Parking on Premises", "Gym",
        "Pool", "Elevator in Building", "Cat(s)", "Hangers", "Cable TV",
        "Suitable for Events", "Pets live on this property", "Safety Card",
        "Breakfast", "Doorman", "Family/Kid Friendly", "Washer / Dryer",
        "First Aid Kit", "Hot Tub", "Heating", "Essentials", "Internet",
        "Smoking Allowed", "Carbon Monoxide Detector", "Wireless Internet",
        "Fire Extinguisher", "TV", "Indoor Fireplace", "Wheelchair Accessible",
        "Dryer"
        ]
    print(cancellation_policies)
    # Initialize amenities array with 0 for each amenity
    amenities = [0] * len(amenities_options)
    
    # Set 1 for each checked amenity
    for i, amenity in enumerate(amenities_options):
        if amenity in request.form.getlist('amenities[]'):
            amenities[i] = 1

    # Input the values from the HTML form into the DataFrame
    row = {
    'accommodates': accommodates,
    'bathrooms': bathrooms,
    'bedrooms': bedrooms,
    'beds': beds,
    'guests_included': guests_included,
    'has_availability': has_availability,
    'instant_bookable': instant_bookable,
    'maximum_nights': maximum_nights
    }
    # Concatenate all the lists into one
    all_data = [accommodates, bathrooms, bedrooms, beds, guests_included, has_availability, instant_bookable, maximum_nights]
    all_data.extend(bed_types)
    all_data.extend(cancellation_policies)
    all_data.extend(property_types)
    all_data.extend(room_types)
    all_data.extend(amenities)

    # Create a DataFrame with the concatenated row data
    new_row = pd.DataFrame([all_data], columns=columns)

    # Concatenate the new row with the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.fillna(0)
    df = df.astype(int)

    # Load the PyCaret model
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hotel_model"
    model = load_model(model_path)

    # Make predictions using the loaded model
    predicted_price  = model.predict(df)
    predicted_price_int = int(predicted_price[0])  # Convert predicted price to string

    # Render the template and pass the predicted price to it
    return render_template('hotel-predict.html', predicted_price=predicted_price_int)

if __name__ == '__main__':
    app.run(debug=True)

