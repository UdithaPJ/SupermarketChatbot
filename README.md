### **Supermarket Chatbot**  

#### **Overview**  
The **Supermarket Chatbot** is a Python-based AI assistant designed to help users locate items in a supermarket by providing their corresponding shelf numbers. The chatbot processes natural language input, recognizes shopping-related queries, and returns relevant shelf information. It utilizes **Natural Language Processing (NLP)** and a pre-trained **deep learning model** for intent recognition.  

#### **Features**  
- **Natural Language Processing (NLP):** Uses NLTK and TensorFlow to process user queries.  
- **Machine Learning-Based Intent Detection:** Classifies user input into predefined categories using a trained deep learning model.  
- **Shopping List Handling:** Accepts multiple grocery items at once and returns their shelf numbers.  
- **Predefined Item-Shelf Mapping:** A dictionary-based approach to locate various supermarket items.  
- **Conversational Interface:** Engages in basic interactions with the user and ends the conversation gracefully.  

#### **Technologies Used**  
- **Python** (Primary language)  
- **TensorFlow/Keras** (Deep learning model for intent classification)  
- **NLTK** (Natural Language Toolkit for text preprocessing)  
- **NumPy** (Data manipulation)  
- **Pickle** (For storing model data)  
- **JSON** (For intent classification storage)  

#### **Installation**  
1. Clone the repository:  
   ```sh
   git clone https://github.com/UdithaPJ/SupermarketChatbot.git
   cd SupermarketChatbot
   ```
2. Install required dependencies:  
   ```sh
   pip install numpy nltk tensorflow
   ```
3. Download NLTK resources:  
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```
4. Run the chatbot:  
   ```sh
   python chatbot.py
   ```

#### **Usage**  
1. The chatbot starts with a greeting.  
2. Users can enter grocery items, and the bot will provide their shelf locations.  
3. If multiple items are entered (comma-separated), the bot will return the shelf numbers for all.  
4. The chatbot continues until the user exits.  

#### **Example Interaction:**  
```
You: Where can I find milk?  
Bot: Milk is on Shelf 3.  

You: I need bread, eggs, and cheese.  
Bot:  
- Bread: Shelf 2  
- Eggs: Shelf 3  
- Cheese: Shelf 4  

Is there anything else I can assist you with?  
```

#### **Project Structure**  
```
├── chatbot.py           # Main chatbot script  
├── intents.json         # Contains predefined intents and responses  
├── words.pkl            # Tokenized words for NLP  
├── classes.pkl          # Classified intents  
├── chatbot_model.h5     # Pre-trained deep learning model  
└── README.md            # Project documentation 
