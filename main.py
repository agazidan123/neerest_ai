import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Load the data from the CSV file (replace with your CSV path)
df = pd.read_csv(r"https://raw.githubusercontent.com/EbadaHamdy/smartto/main/0.csv")

# Convert relevant columns to string type
df['Tag'] = df['Tag'].astype(str)
df['Review'] = df['Review'].astype(str)
df['Comment'] = df['Comment'].astype(str)

# TF-IDF Vectorization setup
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Tag'] + ' ' + df['Review'] + ' ' + df['Comment'])

# Fixed survey responses
fixed_survey_responses = [
    "Tours", "Archaeological tourism", "for fun", "Religious Tourism",
    "parks", "Museums", "malls", "games", "Natural views", "water places"
]

def get_recommendations_with_budget(country, governorates, num_days=7, num_plans=10):
    filtered_df = df[(df['Country'] == country) & (df['Governorate'].isin(governorates))]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified country and governorates.")

    all_recommendations = []

    for plan_num in range(1, num_plans + 1):
        user_profile = f"{country} {' '.join(governorates)} {' '.join(fixed_survey_responses)}"
        user_profile_vectorized = tfidf_vectorizer.transform([user_profile])
        max_price_per_day = 1000  # Assuming a fixed budget of 1000 for example
        recommendations_df = pd.DataFrame(columns=['Title', 'Price', 'tags', 'Governorate', 'Day'])
        recommended_titles = set()
        hotels = filtered_df[filtered_df['tags'].str.lower().str.contains('hotel') & (filtered_df['Price'] <= max_price_per_day)]
        hotel_recommendation = pd.DataFrame()
        hotel_price = 0.0

        if not hotels.empty:
            hotel_recommendation = hotels.sample(1)[['Title', 'Price', 'tags', 'Governorate']]
            recommended_titles.add(hotel_recommendation['Title'].iloc[0])
            hotel_price = hotel_recommendation['Price'].iloc[0] * num_days

        for day in range(1, num_days + 1):
            daily_recommendations = []
            governorate_df = filtered_df[filtered_df['Governorate'].isin(governorates)]
            restaurants = governorate_df[governorate_df['tags'].str.lower().str.contains('restaurant') & (governorate_df['Price'] <= max_price_per_day)]
            if not restaurants.empty:
                restaurant_recommendation = restaurants.sample(1)
                for _, row in restaurant_recommendation.iterrows():
                    row['Day'] = day
                    daily_recommendations.append(row)
                    recommended_titles.add(row['Title'])
            random.shuffle(fixed_survey_responses)
            place_recommendations = []
            for response in fixed_survey_responses:
                response_indices = [i for i, tag in enumerate(governorate_df['tags']) if response.lower() in tag.lower()]
                valid_indices = [idx for idx in response_indices if governorate_df.iloc[idx]['Title'] not in recommended_titles]
                if valid_indices:
                    random_index = random.choice(valid_indices)
                    recommendation = governorate_df.iloc[random_index][['Title', 'Price', 'tags', 'Governorate']]
                    recommendation['Day'] = day
                    place_recommendations.append(recommendation)
                    recommended_titles.add(recommendation['Title'])
            num_additional_recommendations = min(2, len(place_recommendations))
            daily_recommendations.extend(place_recommendations[:num_additional_recommendations])
            for recommendation in daily_recommendations:
                recommendations_df = pd.concat([recommendations_df, pd.DataFrame([recommendation])])

        total_plan_price = hotel_price
        for idx, row in recommendations_df.iterrows():
            total_plan_price += row['Price']

        plan_recommendations = []
        for day in range(1, num_days + 1):
            day_recommendations = recommendations_df[recommendations_df['Day'] == day]
            if len(day_recommendations) >= 3:
                daily_plan = [
                    f"Day {day}:",
                    f"Restaurant: {day_recommendations.iloc[0]['Title']} , Price: {day_recommendations.iloc[0]['Price']}",
                    f"{day_recommendations.iloc[1]['tags']}: {day_recommendations.iloc[1]['Title']} , Price: {day_recommendations.iloc[1]['Price']}",
                    f"{day_recommendations.iloc[2]['tags']}: {day_recommendations.iloc[2]['Title']} , Price: {day_recommendations.iloc[2]['Price']}"
                ]
                plan_recommendations.extend(daily_plan)

        all_recommendations.append({
            'plan_number': plan_num,
            'hotel': hotel_recommendation['Title'].iloc[0] if not hotel_recommendation.empty else "No suitable hotel found",
            'hotel_price_per_day': hotel_recommendation['Price'].iloc[0] if not hotel_recommendation.empty else 0.0,
            'total_hotel_price': hotel_price,
            'plan_recommendations': plan_recommendations,
            'total_plan_price': total_plan_price,
        })

    return all_recommendations

# FastAPI setup
app = FastAPI()

class PlanRequest(BaseModel):
    country: str
    governorates: List[str]

class RecommendationsResponse(BaseModel):
    recommendations: List[dict]

@app.post("/recommendations/", response_model=RecommendationsResponse)
def recommend_plans(request: PlanRequest):
    global recommendations_storage
    recommendations_storage = get_recommendations_with_budget(
        request.country,
        request.governorates,
    )
    return {"recommendations": recommendations_storage}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
