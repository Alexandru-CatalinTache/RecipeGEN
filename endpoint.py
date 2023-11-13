from flask import Flask, request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

def find_similar_recipes(ingredients, category, calories, rating, time, recipe_info, vectorizer, cosine_similarities):
    query_string = ' '.join([category] + ingredients + [calories] + [time] + [rating])
    query_vector = vectorizer.transform([query_string])

    # Calculate cosine similarities between the query vector and all recipe vectors
    similarities = cosine_similarity(query_vector, recipe_info)

    # Get indices of recipes sorted by similarity (descending order)
    similar_recipe_indices = similarities.argsort()[0][::-1]

    return similar_recipe_indices

app = Flask(__name__)

@app.route('/findSimilarRecipe')
def findSimilarRecipe():
    ingredients_query = request.args.getlist('ingredients_query')
    category_query = str(request.args.get('category_query'))
    calories = str(request.args.get('calories'))
    rating = str(request.args.get('rating'))
    totalTime = str(request.args.get('totalTime'))
    number_of_desired_recipes = int(request.args.get('numberOfDesiredRecipes'))

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

    cosine_similarities = np.load("cosineSimilarities.npy", allow_pickle=True)

    similar_recipes_indices = find_similar_recipes(ingredients_query, category_query, calories, rating, totalTime, vectorized_data, vectorizer, cosine_similarities)

    showArray = []
    count = 0

    for idx in similar_recipes_indices:
        count = count + 1
        recipe = recipe_info[idx]
        if count <= number_of_desired_recipes:
            saveObj = {}
            saveObj["Name"] = recipe['name']
            saveObj["Category"] = recipe['category']
            saveObj["Ingredients"] = recipe["ingredients"]
            saveObj["Calories"] = recipe["calories"]
            saveObj["Rating"] = recipe["rating"]
            saveObj["TotalTime"] = recipe["totalTime"]
            showArray.append(saveObj)
        else:
            break

    f.close()
    return showArray


if __name__ == '__main__':
    app.run(debug=True)
