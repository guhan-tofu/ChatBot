# ChatBot
A chatbot made using predefined intents and a pytorch model which can be manipulated by just changing the contents of the intents.json as well as some minor tweaking in model if necessary. 


# Gathering Data
In order to gather data, either use my web-scraper and get info on your desired website or if you have the info use chatgpt to enter this prompt


"Make me a json file with clearly defined tags, intents and responses based on (____insert your information___) and make it suitable for a chatbot"


It should return the correct prompt, if not use the json template given.


# Installing Dependencies
`pip install flask`


`pip install flask-cors`


`pip install torch`


`pip install nltk`


`pip install numpy`


# Running the Code
`python nltk_utils.py` - uncomment the third line on the first run


`python model.py`

`python train.py`


`flask run` - let the chatbot file run in background


Open the [frontend-standalone file](https://github.com/guhan-tofu/ChatBot/tree/main/standalone-frontend) in another window and run with debugging
