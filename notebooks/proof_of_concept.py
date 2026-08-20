# -*- coding: utf-8 -*-
"""Original Colab proof-of-concept, archived for reference.

Ported into src/ as config.py, data.py, client.py, classify.py, persist.py,
pipeline.py. Kept here unmodified except the hardcoded API key, which was
removed (it had been pasted in plaintext and should be treated as leaked/
rotated). See src/pipeline.py for the maintained version.

Original file was located at:
    https://colab.research.google.com/drive/1QtepLF1jS72WHfEs4j-gMjx05wwrxIo_
"""

!pip install -q openai

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64

from PIL import Image
from openai import OpenAI

# Load CIFAR-10-C files
defocus = np.load("defocus_blur.npy")
labels = np.load("labels.npy")

# CIFAR-10 cat label = 3
cat_indices = np.where(labels == 3)[0]

# First cat
cat_number = 0
image_number = cat_indices[cat_number]

# Start with severity 1
severity = 1
images_per_level = 10000

index = (severity - 1) * images_per_level + image_number

# Get image
image_array = defocus[index]

# Show it
plt.imshow(image_array)
plt.title("Cat 0 - Defocus Blur - Severity 1")
plt.axis("off")
plt.show()

# Save as a normal PNG
Image.fromarray(image_array).save("test_cat.png")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="TODO",  # was hardcoded in the original — rotate that key, never commit a real one
)

with open("test_cat.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.chat.completions.create(
    model="openrouter/free",

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
Classify this CIFAR-10 image.

Choose exactly ONE label:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Respond with only the label.
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

prediction = response.choices[0].message.content.strip().lower()

print("Prediction:", prediction)

true_label = "cat"

correct = 1 if prediction == true_label else 0

print("True label:", true_label)
print("Prediction:", prediction)
print("Correct:", correct)

results = pd.DataFrame([
    {
        "model": "openrouter_free_test",
        "blur_type": "defocus",
        "severity": severity,
        "image_id": image_number,
        "true_label": true_label,
        "prediction": prediction,
        "correct": correct
    }
])

display(results)

results.to_csv("results.csv", index=False)

import pandas as pd
import base64
from PIL import Image

results_list = []

true_label = "cat"
images_per_level = 10000

for severity in range(1, 6):
    # Get the same cat at this severity
    index = (severity - 1) * images_per_level + image_number
    image_array = defocus[index]

    # Save temporary image
    temp_filename = f"cat_severity_{severity}.png"
    Image.fromarray(image_array).save(temp_filename)

    # Convert image to base64
    with open(temp_filename, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Send to OpenRouter
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Classify this CIFAR-10 image.

Choose exactly ONE label:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Respond with only the label.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    prediction = response.choices[0].message.content.strip().lower()
    correct = 1 if prediction == true_label else 0

    results_list.append({
        "model": "openrouter_free_test",
        "blur_type": "defocus",
        "severity": severity,
        "image_id": int(image_number),
        "true_label": true_label,
        "prediction": prediction,
        "correct": correct
    })

# Turn results into a DataFrame
new_results = pd.DataFrame(results_list)

# Append to existing CSV if it already exists
try:
    old_results = pd.read_csv("results.csv")
    all_results = pd.concat([old_results, new_results], ignore_index=True)
except FileNotFoundError:
    all_results = new_results

# Save updated CSV
all_results.to_csv("results.csv", index=False)

# Show just the 5 new rows
display(new_results)

from google.colab import files

files.download("results.csv")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

plt.plot(df["severity"], df["correct"], marker="o")

plt.xlabel("Blur Severity")
plt.ylabel("Correct (1 = Yes, 0 = No)")
plt.title("Defocus Blur Classification - Cat 0")

plt.xticks([1, 2, 3, 4, 5])
plt.yticks([0, 1], ["Incorrect", "Correct"])

plt.show()
