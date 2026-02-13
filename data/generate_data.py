import pandas as pd
import random
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# 1. generated "Human" examples (Varied, idiomatic, sometimes messy)
human_texts = [
    "I dunno man, it just feels weird. Like, why would they do that? It makes no sense.",
    "The sunset was crashing purple and orange, a bruise on the sky. I stood there, watching it fade.",
    "Hey!! Did u see the game last nite?? It was insane...",
    "Code didn't run. Checked the logs, found a null pointer. Fixed it. Classic Friday deploy.",
    "Honestly, I prefer the old version. The new UI is just too cluttered for my taste.",
    "Yo, can we reschedule? Something came up.",
    "The algorithmic complexity is O(n), but in practice it's faster because of the cache locality.",
    "I've been thinking about what you said. It really resonated with me.",
    "Running late! Start without me.",
    "The rain drummed against the window, a steady, rhythmic beat that lulled me to sleep.",
    "OMG that movie was terrible lol. 0/10 would not recommend.",
    "Listen, we need to talk about the project timeline. It's slipping.",
    "My cat just knocked over my coffee. Great start to the morning.",
    "Im so tired of this weather. When is summer coming???",
    "Just pushed the commit. Let me know if it builds.",
    "The sheer audacity of that claim is laughable.",
    "Pizza for dinner? I'm feeling lazy.",
    "Look, it is what it is. We can't change the past.",
    "Debugging this legacy code is like trying to untangle headphones in the dark.",
    "Happy birthday!!! Hope you have an awesome day!",
    "I think we should pivot. The current strategy isn't working.",
    "Can you believe he said that? Unbelievable.",
    "Sent the email. waiting for reply.",
    "The coffee shop was buzzing with activity, the smell of roasted beans answering the air.",
    "I'll be there in 5.",
    "Did you try turning it off and on again?",
    "That's a hard pass from me.",
    "I'm going to grab some lunch. Want anything?",
    "The logic holds up, but the implementation is messy.",
    "Wow. Just wow.",
    "Can u send me the link pls?",
    "I'm not sure if I agree with that premise.",
    "Let's touch base tomorrow morning.",
    "The drive was long, but the view was worth it.",
    "Seriously? Again?",
    "I need a vacation. Like, yesterday.",
    "The server crashed again. 500 error.",
    "Got it. Will do.",
    "I'm open to suggestions.",
    "That's hilarious!",
    "The data looks corrupt. We need to restore from backup.",
    "Who's responsible for this mess?",
    "I'm heading out. Cya.",
    "The conference was a waste of time.",
    "I love that song!",
    "Let's agree to disagree.",
    "The update broke everything.",
    "I'm so ready for the weekend.",
    "Can you double check the numbers?",
    "It's a feature, not a bug."
]

# 2. generated "AI" examples (Structured, repetitive, transition-heavy, explanation-heavy)
ai_texts = [
    "In conclusion, the implementation of this algorithm offers significant advantages in terms of efficiency.",
    "However, it is important to consider the potential drawbacks of this approach.",
    "The sunset displayed a vibrant array of colors, predominantly purple and orange, resembling a bruise.",
    "Furthermore, the user interface has been redesigned to enhance user experience and accessibility.",
    "To resolve the issue, strictly follow these steps: 1. Restart the server. 2. Clear the cache.",
    "Additionally, the system provides a robust mechanism for error handling.",
    "I am an AI language model and cannot have personal opinions or feelings.",
    "The algorithmic complexity is calculated as O(n), ensuring optimal performance for large datasets.",
    "Moreover, the integration of third-party libraries facilitates rapid development.",
    "It is crucial to adhere to the project timeline to ensure successful delivery.",
    "The weather affects mood significantly. Many people prefer summer over winter due to increased sunlight.",
    "Here is a summary of the key points discussed in the meeting:",
    "Consequently, the data must be validated before processing to prevent errors.",
    "The movie received negative reviews due to its poor plot and character development.",
    "I can assist you with scheduling the meeting. Please provide the preferred time and date.",
    "Debugging legacy code requires a systematic approach to identify and resolve issues.",
    "Here is a list of potential solutions to the problem:",
    "The concept of 'turning it off and on again' is a common troubleshooting step.",
    "In summary, the proposed strategy aligns with the company's long-term goals.",
    "It is recommended to back up the data regularly to avoid validation errors.",
    "The prompt asks for a short paragraph about the solar system. The solar system consists of the Sun and...",
    "As an artificial intelligence, I do not process personal emotions.",
    "The error 500 indicates an internal server error. Please check the server logs.",
    "There are several factors to consider when choosing a programming language.",
    "Firstly, we must analyze the requirements. Secondly, we must design the architecture.",
    "On the other hand, there are alternative methods that may yield better results.",
    "To optimize the code, one should consider using a hash map.",
    "The quick brown fox jumps over the lazy dog is a pangram.",
    "Please let me know if you require any further assistance.",
    "Understanding the root cause of the problem is essential for finding a permanent fix.",
    "The user experience is paramount in modern web development.",
    "Machine learning models require large datasets for effective training.",
    "Therefore, we can conclude that the hypothesis is valid.",
    "This essay will discuss the impact of technology on society.",
    "The output of the function is determined by the input parameters.",
    "In addition to the benefits mentioned above, there are cost savings.",
    "It is important to note that the results may vary depending on the environment.",
    "The following code snippet demonstrates how to implement the function.",
    "Can you please provide more context regarding your request?",
    "Effective communication is key to successful collaboration.",
    "The standard deviation is a measure of the amount of variation or dispersion.",
    "Python is a versatile programming language known for its readability.",
    "The project was completed on time and within budget.",
    "Let's explore the various options available for implementation.",
    "The database schema must be normalized to reduce redundancy.",
    "In the realm of software engineering, testing is a critical phase.",
    "The benefits of regular exercise include improved cardiovascular health.",
    "Please verify the integrity of the files before proceeding.",
    "The automated system detected an anomaly in the data stream.",
    "Thank you for your inquiry. I have processed your request."
]

# Create balanced dataset (multiply to get decent volume)
texts = []
labels = []

# Multiply x20 to get ~2000 examples
for _ in range(20):
    for t in human_texts:
        texts.append(t)
        labels.append(0) # Human
    for t in ai_texts:
        texts.append(t)
        labels.append(1) # AI

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1).reset_index(drop=True) # Shuffle

print(f"Generated {len(df)} training examples.")
df.to_csv("data/train.csv", index=False)
print("Saved to data/train.csv")
