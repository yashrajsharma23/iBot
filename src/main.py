# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import nltk
import json
import pickle

import numpy as np
import random

from nltk.stem import WordNetLemmatizer

import tensorflow as tf

import requests
from datetime import datetime, timedelta

API_URL = "https://cricapi.com/api/"
API_KEY = "o1kw33cvFcgkgXyV2TyeMhhlw9p2"
# Creating GUI with tkinter
from tkinter import *

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('../data/raw/management_intents.json').read()
intents = json.loads(data_file)

from os import path


def create_model():
    import train_model


if path.exists("../models/chatbot_management_model.h5"):
    input = input("Model already exist, do you want to recreate the model? Y/N: ")
    if input.lower() == 'y' or input.lower() == 'yes':
        create_model()
        # train_model.create_model()
else:
    create_model()

# model = load_model('C:/Users/Yash Raj/Python Projects/3335 - Data mining & Analysis/chatbot_model.h5')
model = tf.keras.models.load_model("../models/chatbot_management_model.h5")

intents = json.loads(open('../data/raw/management_intents.json').read())
words = pickle.load(open('../data/processed/management_words.pkl', 'rb'))
classes = pickle.load(open('../data/processed/management_classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


class ApiAction:
    action = ""

    def name(self, act):
        action = act
        return "action_match_news"

    def run(date, timedelta):
        ipl_teams = ['Royal Challengers Bangalore', 'Mumbai Indians', 'Chennai Super Kings',
                     'Delhi Capitals', 'Punjab Lions', 'Kolkata Knight Riders',
                     'Rajasthan Royals', 'Sunrisers Hyderabad']

        res = requests.get(API_URL + "matches" + "?apikey=" + API_KEY)
        if res.status_code == 200:
            # print(res.json())
            data = res.json()["matches"]
            today = date.today()
            tomm = today + timedelta(days=1)
            if " " in str(today):
                today = str(today).split(' ')[0]

            if " " in str(tomm):
                tomm = str(tomm).split(' ')[0]

            count = 0
            out_message = ""
            print(today, ":", tomm)
            for i in data:
                match_date = i['date'].split('T')[0]
                if str(match_date) == str(today):
                    for j in ipl_teams:
                        if j in i["team-1"] or j in i["team-2"]:
                            if "winner_team" in i:
                                count = count + 1
                                out_message = out_message + "\n" + str(
                                    count) + ". The match between {} and {} was recently held and {} won.".format(
                                    i["team-1"], i["team-2"], i["winner_team"])
                                print(count, ")", out_message)

                            else:
                                count = count + 1
                                out_message = out_message + "\n" + str(
                                    count) + ". Today match is between {} and {}.".format(i["team-1"], i["team-2"])
                                print(count, ")", out_message)
                            break
                elif str(match_date) == str(tomm):
                    for j in ipl_teams:
                        if j in i["team-1"] or j in i["team-2"]:
                            count = count + 1
                            out_message = out_message + "\n" + str(
                                count) + ". The next match is between {} vs {} on {}".format(
                                i["team-1"], i["team-2"], tomm)
                            print(count, ")", out_message)
                            break

            msg = ""
            out_message = "Here are some IPL quick info:\n" + out_message
            msg = out_message

        return msg  # []


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            print("Tag::::::: " + tag)
            if tag == "current_matches":
                print("Tag::::::: " + tag)
                result = ApiAction.run(datetime, timedelta)
                print("Match results::::" + result)
                break
            else:
                result = random.choice(i['responses'])
                break
    return result


def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Support: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def default_msg():
    msg = "Hello, I am Jarvis.\nHow may I help you?"
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "Support: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("The matrix squad [DGRSY]")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatLog.config(state=DISABLED)
# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)
# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)
# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
default_msg()
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()
