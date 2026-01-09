import random

RESPONSES = {
    "sadness": [
        "I'm really sorry you're feeling this way.",
        "It sounds heavy. I'm here with you."
    ],
    "anxiety": [
        "Let's slow things down together.",
        "Try taking a deep breath."
    ],
    "exhaustion": [
        "You've been carrying a lot.",
        "Rest matters."
    ],
    "stress": [
        "That sounds overwhelming.",
        "You're doing your best."
    ],
    "neutral": [
        "I'm here if you want to talk."
    ]
}

def generate_response(emotion):
    return random.choice(RESPONSES.get(emotion, RESPONSES["neutral"]))
