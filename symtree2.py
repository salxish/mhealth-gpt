import json
import random

SYMPTOM_TREES = {
    "chest_start": {
        "question" : "Is the pain radiating to your arm?",
        "options" : {
            "yes" : "sweat_breath",
            "no" : "jaw"
        }
    },

    "sweat_breath": {
        "question" : "Are you sweating or short of breath?",
        "options" : {
            "yes" : "emergency",
            "no" : "high_risk"
        }
    },

    "jaw": {
        "question" : "Is the pain radiating to your jaw?",
        "options" : {
            "yes" : "sweat_breath",
            "no" : "sharp"
        }
    },

    "sharp": {
        "question" : "Is the pain sharp?",
        "options" : {
            "yes" : "breath",
            "no" : "low_risk"
        }
    },

    "breath": {
        "question" : "Is it worse when you breathe?",
        "options" : {
            "yes" : "medium_risk",
            "no" : "low_risk"
        }
    },

    "emergency" : {
        "risk": "EMERGENCY",
        "action": "Call emergency services immediately"
    },

    "high_risk" : {
        "risk": "HIGH",
        "action": "Seek urgent care"
    },

    "medium_risk" : {
        "risk": "MEDIUM",
        "action": "See a GP or clinic"
    },

    "low_risk" : {
        "risk": "LOW",
        "action": "Seek urgent care"
    },


}

def random_answer():
    return random.choice([
        "yes",
        "no",
    ])

def engine(tree):
    node = tree["chest_start"]
    conversation = []

    while True:
        if "risk" in node:
            conversation.append({
                "role": "specialist",
                "content": f"Risk: {node['risk']}. {node['action']}"
            })
            break

        conversation.append({
            "role": "specialist",
            "content": node["question"]
        })

        answer = random_answer()

        conversation.append({
            "role": "user",
            "content": answer
        })

        next_node = node["options"][answer]
        node = tree[next_node]

    return conversation

print("conversation returned!")
dataset = []

for i in range(100):

    conv = engine(SYMPTOM_TREES)

    dataset.append({
        "messages": conv
    })

with open("symptoms.jsonl", "w") as file:
    for sample in dataset:
        file.write(json.dumps(sample))
        file.write("\n")