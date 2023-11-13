from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

f = open('SAC_database.json')
recipes = json.load(f)

for recipe in recipes:
    if ("aggregateRating" not in recipe):
        recipe["aggregateRating"] = {}
        recipe["aggregateRating"]["ratingValue"] = "-"

recipe_info = [
    {
        "name": recipe.get("name", "Recipe Name Not Found"),
        "category": recipe.get("recipeCategory", "Category Not Found"),
        "ingredients": ' '.join(recipe.get("recipeIngredient", [])),  # Convert list to string
        "calories": recipe.get("nutrition", {}).get("calories"),
        "rating": recipe.get("aggregateRating", {}).get("ratingValue"),
        "totalTime": recipe.get("totalTime")
    }
    for recipe in recipes
]

recipe_strings = [' '.join(
    [recipe["category"], 
     recipe["ingredients"],
     recipe["calories"],
     recipe["totalTime"],
     recipe["rating"]]) for recipe in recipe_info]

vectorizer = TfidfVectorizer()
vectorized_data = vectorizer.fit_transform(recipe_strings)

cosine_similarities = cosine_similarity(vectorized_data)

np.save("cosine_similarities.npy", vectorizer, True)

f.close()