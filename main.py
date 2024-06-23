from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import random

app = FastAPI()

# Load data from CSV file hosted on GitHub
file_url = "https://raw.githubusercontent.com/EbadaHamdy/smartto/main/0.csv"
df = pd.read_csv(file_url)

# Function to get recommendations by location
def get_recommendations_by_location(country, governorate, df, num_recommendations=20):
    try:
        # Convert column names to lower case and replace spaces with underscores
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Filter data based on country and governorate
        filtered_df = df[(df['country'].str.lower() == country.lower()) & (df['governorate'].str.lower() == governorate.lower())]
        
        # If fewer rows are found than requested recommendations, return all available rows
        if len(filtered_df) == 0:
            raise ValueError(f"No recommendations found for {country}, {governorate}.")
        elif len(filtered_df) < num_recommendations:
            num_recommendations = len(filtered_df)  # Show all available recommendations
        
        # Randomly select the requested number of recommendations
        selected_indices = random.sample(range(len(filtered_df)), num_recommendations)
        
        # Select columns based on availability in the data
        columns_to_select = ['title', 'price', 'rating', 'review', 'tags']
        if 'address' in filtered_df.columns:
            columns_to_select.append('address')
        if 'img_link' in filtered_df.columns:
            columns_to_select.append('img_link')
        
        recommendations = filtered_df.iloc[selected_indices][columns_to_select].copy()
        
        return recommendations
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Error: {e} column(s) not found in the data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# Define request data model using Pydantic BaseModel
class LocationRequest(BaseModel):
    country: str
    governorate: str

# Example endpoint to get recommendations
@app.post("/recommendations/")
async def get_recommendations(location: LocationRequest):
    country = location.country
    governorate = location.governorate
    num_recommendations = 20  # You can adjust this as needed

    try:
        # Call your recommendation function
        recommendations = get_recommendations_by_location(country, governorate, df, num_recommendations)

        if recommendations.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations found for {country}, {governorate}.")

        # Convert recommendations to JSON-compatible format
        return recommendations.to_dict(orient='records')

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Optional: If you need to run the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
