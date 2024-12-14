import sys
import os
import logging
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

item_shelf_mapping = {
    "apple": "Shelf 1",
    "banana": "Shelf 1",
    "bread": "Shelf 2",
    "milk": "Shelf 3",
    "egg": "Shelf 3",
    "cheese": "Shelf 4",
    "tomato": "Shelf 5",
    "lettuce": "Shelf 5",
    "chicken": "Shelf 6",
    "carrot": "Shelf 7",
    "potato": "Shelf 7",
    "onion": "Shelf 8",
    "cucumber": "Shelf 8",
    "pasta": "Shelf 9",
    "rice": "Shelf 9",
    "flour": "Shelf 10",
    "sugar": "Shelf 10",
    "salt": "Shelf 11",
    "pepper": "Shelf 11",
    "beef": "Shelf 12",
    "pork": "Shelf 12",
    "fish": "Shelf 13",
    "shrimp": "Shelf 13",
    "yogurt": "Shelf 14",
    "butter": "Shelf 14",
    "juice": "Shelf 15",
    "water": "Shelf 15",
    "soda": "Shelf 16",
    "beer": "Shelf 16",
    "wine": "Shelf 17",
    "whiskey": "Shelf 17",
    "shampoo": "Shelf 18",
    "soap": "Shelf 18",
    "toothpaste": "Shelf 19",
    "brush": "Shelf 19",
    "detergent": "Shelf 20",
    "bleach": "Shelf 20"
}

item_list_to_print = ""

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s_word in sentence_words:
        for index, word in enumerate(words):
            if word == s_word:
                bag[index] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for result in results:
        return_list.append({'intent': classes[result[0]], 'probability': str(result[1])})
    return return_list

def get_response(intents_list, intents_json, item_list_to_print):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    if tag == 'shopping_list':
        response, updated_item_list_to_print = handle_shopping_list_response("Here are the shelf numbers for your items:", intents_json, item_list_to_print)
        return response, updated_item_list_to_print
    else:
        for intent in list_of_intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses']), item_list_to_print
        return "Sorry, I didn't understand that.", item_list_to_print

def handle_shopping_list_response(default_response, intents_json, item_list_to_print):
    list_of_intents = intents_json['intents']
    message = input("Please enter your shopping list (items separated by commas): ")
    for intent in list_of_intents:
        if intent['tag'] == 'shopping_list':
            print(random.choice(intent['responses']))
    items = message.split(',')
    items = [lemmatizer.lemmatize(item.strip().lower()) for item in items]
    response = default_response + "\n"
    for item in items:
        shelf = item_shelf_mapping.get(item, "Sorry, item not found")
        response += f"{item.capitalize()}: {shelf}\n"
        item_list_to_print += f"{item.capitalize()}: {shelf}\n"
    response += "\nIs there anything else I can assist you with?"
    return response, item_list_to_print

print("GO! Bot is running!")

while True:
    message = input("You: ")
    ints = predict_class(message)
    res, item_list_to_print = get_response(ints, intents, item_list_to_print)
    print(f"Bot: {res}")
    if ints[0]['intent'] == 'goodbye' or ints[0]['intent'] == 'no':
        if item_list_to_print:
            print(f"\nHere is your item list and their shelf numbers:\n{item_list_to_print}")
        break
