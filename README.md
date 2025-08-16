import pandas as pd
import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import numpy as np
additional_data = {
    'happy': [
        "I'm feeling great today!",
        "Life is beautiful.",
        "I can't stop smiling.",
        "I'm so excited about everything.",
        "Everything is going my way.",
        "Feeling on top of the world!",
        "I just received the best news!",
        "Today is an amazing day.",
        "I'm surrounded by love and joy.",
        "This is the happiest I've been in a while.",
        "I am happy."
    ],
    'sad': [
        "I feel so down.",
        "Nothing seems to cheer me up.",
        "Tears keep falling from my eyes.",
        "I'm drowning in sorrow.",
        "I just want to be alone.",
        "Everything feels heavy today.",
        "I can't stop crying.",
        "I'm heartbroken.",
        "The sadness is overwhelming.",
        "Why does everything hurt?"
    ],
    'anxious': [
        "My heart is racing for no reason.",
        "I can't stop worrying.",
        "I feel so nervous and unsettled.",
        "I'm scared something bad is going to happen.",
        "Everything is making me panic.",
        "I feel like I'm losing control.",
        "I can't sleep, I'm too anxious.",
        "Even the smallest things stress me out.",
        "I'm constantly on edge.",
        "I feel like I can't breathe."
    ],
    'angry': [
        "I'm so mad right now!",
        "I feel like screaming.",
        "Everything is pissing me off today.",
        "I can't stand this anymore.",
        "I'm furious about what happened.",
        "People are so annoying.",
        "I'm about to explode with rage.",
        "Why do they always do this to me?",
        "I'm not in the mood, back off!",
        "I hate this situation!"
    ],
    'depressed': [
        "I feel completely empty.",
        "Nothing matters anymore.",
        "I'm just existing, not living.",
        "Every day is a struggle.",
        "I feel like giving up on everything.",
        "I've lost all motivation.",
        "The world feels grey.",
        "Even getting out of bed is hard.",
        "I feel stuck and hopeless.",
        "I feel like I'm drowning slowly."
    ],
    'normal': [
        "Just a regular day.",
        "I'm fine, nothing much going on.",
        "Feeling okay, I guess.",
        "It's been an average day.",
        "I'm just chilling.",
        "Not too bad, not too great.",
        "Things are as usual.",
        "Nothing exciting to report.",
        "Today's just normal.",
        "I'm alright."
    ],
    'suicidal': [
        "I want to die.",
        "There's no point in living.",
        "I'm thinking of ending my life.",
        "I feel like giving up.",
        "Everything feels hopeless.",
        "I can't go on anymore.",
        "Life is meaningless.",
        "No one cares if I disappear.",
        "I'm going to end it all.",
        "Death feels like the only escape."
    ]
}
def load_and_train():
    try:
        csv_data_loaded = False
        try:
            df_csv = pd.read_csv("Chatbot-Based Mental Health Support.csv")
            df_csv = df_csv[['statement', 'status']].dropna()
            df_csv.columns = ['text', 'label']
            print(f"CSV data loaded: {len(df_csv)} samples")
            csv_data_loaded = True
        except FileNotFoundError:
            print("CSV file not found. Using additional data only.")
            df_csv = pd.DataFrame(columns=['text', 'label'])
        except Exception as e:
            print(f"Error loading CSV: {e}. Using additional data only.")
            df_csv = pd.DataFrame(columns=['text', 'label'])
        additional_texts = []
        additional_labels = []
        
        for emotion, texts in additional_data.items():
            for text in texts:
                additional_texts.append(text)
                additional_labels.append(emotion)
        
        df_additional = pd.DataFrame({
            'text': additional_texts,
            'label': additional_labels
        })
        
        print(f"Additional data loaded: {len(df_additional)} samples")
        if csv_data_loaded and len(df_csv) > 0:
            df_combined = pd.concat([df_csv, df_additional], ignore_index=True)
            print(f"Combined dataset: {len(df_combined)} total samples")
        else:
            df_combined = df_additional
            print(f"Using additional data only: {len(df_combined)} samples")
        emotion_counts = df_combined['label'].value_counts()
        print("Emotion distribution:")
        for emotion, count in emotion_counts.items():
            X_train, X_test, y_train, y_test = train_test_split(
            df_combined['text'], df_combined['label'], test_size=0.2, random_state=42
        )
        optimal_k = max(3, min(int(np.sqrt(len(X_train))),10))
        print(f"Using k={optimal_k} for KNN classifier")
        model = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english',max_features=3000,ngram_range=(1, 2),min_df=1,max_df=0.8)),
            ('classifier', KNeighborsClassifier(
                n_neighbors=optimal_k,
                weights='distance',  
                algorithm='auto',
                metric='cosine' 
            ))
        ])
        
        print("Training KNN model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"KNN Model trained! Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return model, True, len(df_combined), optimal_k
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during training: {str(e)}")
        return None, False, 0, 0
model, model_ready, total_samples, k_value = load_and_train()
response_map = {
    'happy': [
        "That's wonderful! Your joy is contagious! Keep spreading that positive energy! 😊✨",
        "I love your positive attitude! Stay happy and keep shining bright! 🌟💫",
        "Your happiness makes my day! Keep being amazing! 🌈😄",
        "What a beautiful outlook! Hold onto that feeling! 🌻💛"
    ],
    'sad': [
        "I'm here for you during this difficult time. Your feelings are completely valid 💙🤗",
        "It's okay to feel sad. Take your time processing these emotions, I'm here to support you 💧💖",
        "Sending you comfort and understanding. You're not alone in this journey 🌸💙",
        "Your sadness matters. Let yourself feel it, and know that brighter days will come 🌙💜"
    ],
    'depressed': [
        "You're incredibly brave for sharing this. You matter more than you could ever know 💖🌟",
        "Please remember that you're valued and loved, even when it doesn't feel that way 🕊️💜",
        "Your life has meaning and purpose. Let's take this one gentle step at a time 🌱💙",
        "I see your strength, even in this darkness. You're not alone, and help is available 🤝💚"
    ],
    'angry': [
        "I understand you're feeling frustrated and angry. These emotions are valid 💪🌿",
        "Your anger makes sense. Take some deep breaths with me - you've got this 🧘‍♀️💨",
        "It's completely natural to feel angry. Let's find healthy ways to process this energy 🌊💚",
        "I hear your frustration. Your feelings matter, and we can work through this together 🔥➡️❄️"
    ],
    'anxious': [
        "Let's pause together and focus on breathing. You're safe in this moment 🌸🧘‍♀️",
        "Anxiety can feel overwhelming, but you're stronger than you think. One breath at a time 💪🌟",
        "I'm here with you through this anxious feeling. Focus on what you can control right now 🌿💙",
        "Your anxiety is trying to protect you, but you're safe. Let's ground ourselves together 🏔️💚"
    ],
    'stressed': [
        "I can sense the pressure you're under. Let's find ways to lighten that mental load 🌊💆‍♀️",
        "Stress is tough, but you're tougher. Take it one breath, one moment at a time 🧘‍♀️✨",
        "You're handling so much right now. Please be kind and gentle with yourself 💖🌸",
        "It's okay to feel overwhelmed. Let's break things down into manageable pieces 🧩💙"
    ],
    'suicidal': [
        "I'm deeply concerned about you. Your life has value and meaning. Please reach out for professional help immediately 🆘💖",
        "You matter so much. Please contact a crisis helpline: 988 (US) or emergency services. You're not alone 📞🤝",
        "These feelings are temporary, but suicide is permanent. Please talk to someone right now - call 988 or go to your nearest ER 🏥💜",
        "I care about you. Please reach out to National Suicide Prevention Lifeline: 988. Your life is precious 🌟💙"
    ],
    'normal': [
        "Thanks for sharing with me! It's great to check in. How can I support you today? 😊💬",
        "I'm here and ready to listen to whatever is on your mind, big or small 💜🎧",
        "It's wonderful that you're taking time to connect. I'm here for you 🌟💭",
        "Normal days are important too. I'm here if you want to talk about anything 🤗💫"
    ]
}
def predict_emotion(user_input):
    if not user_input.strip():
        return "normal", "Please tell me how you're feeling.", 0.0
    
    if not model_ready:
        return "The model is working", "Sorry, the model couldn't be loaded. Please check your setup.", 0.0
    
    try:
        prediction = model.predict([user_input])[0].lower()
        try:
            probabilities = model.predict_proba([user_input])[0]
            confidence = max(probabilities)
        except:
            confidence = 0.8
        if prediction == 'suicidal':
            responses = response_map.get('suicidal', ["Please seek immediate help. Call 988."])
        else:
            responses = response_map.get(prediction, ["I'm here for you ❤️"])
        
        return prediction.title(), random.choice(responses), confidence
        
    except Exception as e:
        return "normal", f"Error: {str(e)}", 0.0
def send_message():
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_box.insert(tk.END, f"\nYou: {user_input}", "user")
    entry.delete(0, tk.END)
    predicted_emotion, bot_response, confidence = predict_emotion(user_input)
    confidence_text = f" (Confidence: {confidence:.1%})"
    if predicted_emotion.lower() == 'suicidal':
        chat_box.insert(tk.END, f"\nBot (🚨 {predicted_emotion}{confidence_text}): {bot_response}", "emergency")
    else:
        chat_box.insert(tk.END, f"\nBot ({predicted_emotion}{confidence_text}): {bot_response}", "bot")
    chat_box.see(tk.END)

def on_enter(event):
    send_message()

def start_chat():
    if model_ready:
        chat_box.insert(tk.END, "🤖 Mental Wellness Companion (KNN Classifier)\n", "title")
        chat_box.insert(tk.END, f"Trained on {total_samples} diverse emotional expressions using k={k_value} 📊\n", "info")
        chat_box.insert(tk.END, "Hello! I'm here to support you 💜\n", "bot")
        chat_box.insert(tk.END, "Tell me how you're feeling today!\n\n", "bot")
    else:
        chat_box.insert(tk.END, "❌ Model not ready. Please check your setup.\n", "error")

def clear_chat():
    chat_box.delete(1.0, tk.END)
    start_chat()

def show_help():
    help_text = f"""
💡 Enhanced Mental Health Chatbot (KNN Classifier)

🎯 Trained on {total_samples} emotional expressions using:
• K-Nearest Neighbors (k={k_value})
• TF-IDF Vectorization with bigrams
• Cosine similarity distance metric
• Distance-weighted predictions

📊 Emotion Categories:
• Happy, Sad, Anxious, Angry, Depressed
• Normal, Stressed, Suicidal

📝 Try these examples:
• "I'm feeling great today!"
• "I can't stop worrying about everything"
• "I feel completely empty inside"
• "I'm so excited about my promotion!"

🆘 Crisis Resources:
• National Suicide Prevention Lifeline: 9820466726
• Crisis Text Line: Text HOME to 1800111555
• Emergency Services: 108

🤖 This KNN-based chatbot finds the most similar emotional expressions from training data to provide contextually appropriate responses.
    """
    messagebox.showinfo("Help - KNN Mental Health Chatbot", help_text)

def show_crisis_resources():
    crisis_text = """
🚨 CRISIS RESOURCES 🚨

If you're having thoughts of suicide or self-harm:

📞 National Suicide Prevention Lifeline
• Call or Text: 9820466726
• Available 24/7

💬 Crisis Text Line
• Text HOME to 1800111555

🌐 Online Chat
• suicidepreventionlifeline.org

🏥 Emergency Services
• Call 108 or go to your nearest ER

You are not alone. Help is available.
Your life matters. 💜
    """
    messagebox.showinfo("Crisis Resources", crisis_text)

def show_model_info():
    model_info = f"""
🤖 KNN Model Information

📊 Model Details:
• Algorithm: K-Nearest Neighbors
• k-value: {k_value}
• Distance Metric: Cosine Similarity
• Weighting: Distance-based
• Training Samples: {total_samples}

🔤 Text Processing:
• Vectorizer: TF-IDF
• Features: Up to 3000
• N-grams: 1-2 (unigrams + bigrams)
• Stop words: Removed

💡 How it works:
KNN finds the {k_value} most similar emotional expressions from training data and classifies based on their majority emotion, weighted by similarity distance.

✨ Advantages:
• No assumptions about data distribution
• Captures local patterns well
• Good for complex emotional nuances
• Interpretable predictions
    """
    messagebox.showinfo("KNN Model Information", model_info)
root = tk.Tk()
root.title("KNN Mental Health Chatbot 💬")
root.geometry("1000x700")
root.configure(bg="#f0f8ff")
title_frame = tk.Frame(root, bg="#f0f8ff")
title_frame.pack(pady=10, fill=tk.X)

title_label = tk.Label(title_frame, text="🧠 KNN Mental Health Chatbot", 
                      font=("Helvetica", 18, "bold"), bg="#f0f8ff", fg="#4a4a4a")
title_label.pack()

subtitle_label = tk.Label(title_frame, text=f"AI trained on {total_samples} expressions using k={k_value} neighbors", 
                         font=("Helvetica", 10), bg="#f0f8ff", fg="#666666")
subtitle_label.pack()
button_frame = tk.Frame(root, bg="#f0f8ff")
button_frame.pack(pady=5)

clear_btn = tk.Button(button_frame, text="Clear Chat", command=clear_chat, 
                     font=("Helvetica", 10), bg="#ffe6e6", relief=tk.RAISED, bd=1)
clear_btn.pack(side=tk.LEFT, padx=2)

help_btn = tk.Button(button_frame, text="Help", command=show_help, 
                    font=("Helvetica", 10), bg="#e6f3ff", relief=tk.RAISED, bd=1)
help_btn.pack(side=tk.LEFT, padx=2)

model_info_btn = tk.Button(button_frame, text="Model Info", command=show_model_info, 
                          font=("Helvetica", 10), bg="#f0f0f0", relief=tk.RAISED, bd=1)
model_info_btn.pack(side=tk.LEFT, padx=2)

crisis_btn = tk.Button(button_frame, text="🆘 Crisis Help", command=show_crisis_resources, 
                      font=("Helvetica", 10, "bold"), bg="#ffcccc", relief=tk.RAISED, bd=1)
crisis_btn.pack(side=tk.LEFT, padx=2)
status_color = "#90EE90" if model_ready else "#FFB6C1"
status_text = f"✅ KNN Ready (k={k_value})" if model_ready else "❌ Model Error"
status_label = tk.Label(button_frame, text=status_text, font=("Helvetica", 10, "bold"), 
                       bg=status_color, relief=tk.SOLID, bd=1)
status_label.pack(side=tk.RIGHT, padx=5)
chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 12), 
                                    bg="white", fg="black", height=22)
chat_box.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)
chat_box.tag_config("user", foreground="#0066cc", font=("Helvetica", 12, "bold"))
chat_box.tag_config("bot", foreground="#009900", font=("Helvetica", 12))
chat_box.tag_config("emergency", foreground="#cc0000", font=("Helvetica", 12, "bold"))
chat_box.tag_config("title", foreground="#4a4a4a", font=("Helvetica", 14, "bold"))
chat_box.tag_config("info", foreground="#666666", font=("Helvetica", 11))
chat_box.tag_config("error", foreground="#cc0000", font=("Helvetica", 12))
input_frame = tk.Frame(root, bg="#f0f8ff")
input_frame.pack(padx=15, pady=10, fill=tk.X)
entry = tk.Entry(input_frame, font=("Helvetica", 14), bg="#ffffff", 
                relief=tk.SOLID, bd=1)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry.focus()
entry.bind('<Return>', on_enter)
send_button = tk.Button(input_frame, text="Send 💬", command=send_message, 
                       font=("Helvetica", 12, "bold"), bg="#87ceeb", 
                       relief=tk.RAISED, bd=2)
send_button.pack(side=tk.RIGHT)
footer_label = tk.Label(root, text="💡 Powered by KNN Classifier with TF-IDF • Crisis support available", 
                       font=("Helvetica", 9), bg="#f0f8ff", fg="#666666")
footer_label.pack(pady=5)
start_chat()
root.mainloop()
