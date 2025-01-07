# Clinical Name Entity Recognition and Classification

## Overview

This project involves the classification of medical entities in discharge summaries using ClinicalBERT for feature extraction and a Dense neural network for classification. The workflow includes tokenization, feature extraction, encoding, and classification of entities into categories such as "problem," "procedure," or "anatomical structure."

---

## Dataset

The dataset consists of discharge summaries with associated medical entities. Each entity includes attributes such as:

- Text (e.g., "cancer")
- Category (e.g., "problem")
- Confidence score
- Position within the text

### Example Input Data

| Discharge Summary                                       | entity                                                               |
| ------------------------------------------------------- | -------------------------------------------------------------------- |
| Patient shows signs of cancer with associated pain.     | {'[[cancer, problem, 0.934, 24-30], [pain, symptom, 0.876, 45-49]]'} |
| Thoracotomy syndrome observed in post-surgery patients. | {'[[Thoracotomy syndrome, problem, 0.95, 0-20]]'}                    |

---

## Data Processing

### Parsing Entities

The `entity` column contains a string representation of entities. These are parsed and structured into individual rows for easier processing and analysis.

#### Example Code for Parsing:

```python
import ast

def parse_entity(entity_str):
    try:
        entity_list = ast.literal_eval(entity_str)
        parsed_data = []
        for item in entity_list:
            if len(item) == 4:
                parsed_data.append({
                    "text": item[0],
                    "category": item[1],
                    "confidence": item[2],
                    "position": item[3]
                })
        return parsed_data
    except Exception as e:
        print(f"Error parsing entity: {e}")
        return []
```

### Flattening Data

Each entity is extracted into a structured DataFrame with columns:

- Discharge Summary
- Entity Text
- Category
- Confidence
- Position

#### Example Output Data

| Discharge Summary                                       | Entity Text          | Category | Confidence | Position |
| ------------------------------------------------------- | -------------------- | -------- | ---------- | -------- |
| Patient shows signs of cancer with associated pain.     | cancer               | problem  | 0.934      | 24-30    |
| Patient shows signs of cancer with associated pain.     | pain                 | symptom  | 0.876      | 45-49    |
| Thoracotomy syndrome observed in post-surgery patients. | Thoracotomy syndrome | problem  | 0.95       | 0-20     |

---

## Feature Extraction

Tokenization and feature extraction are performed using ClinicalBERT. The `first_entity_features` column contains embeddings generated from ClinicalBERT's final hidden layer for each entity. These embeddings represent the semantic meaning of the entity text.

---

## Label Encoding

The `Second Entity` column is encoded into numerical labels using `LabelEncoder`. A dictionary mapping labels to categories is maintained for decoding predictions back to human-readable form.

---

## Model Training

The classification model is built using a Dense neural network in TensorFlow/Keras. The model architecture includes:

- Input layer
- Two hidden layers with ReLU activation
- Output layer with softmax activation for multi-class classification

### Training and Evaluation

The dataset is split into training and testing sets (80% train, 20% test). The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.

#### Example Code:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
```

---

## Predictions

The model predicts encoded labels for test data, which are then decoded back into human-readable categories using the label mapping dictionary.

#### Example Code for Decoding:

```python
predicted_labels = [key for pred in y_pred for key, value in label_dict.items() if value == pred]
print(predicted_labels)
```

---

## Dependencies

- Python
- pandas
- numpy
- TensorFlow/Keras
- scikit-learn
- transformers

---

## Results

The trained model achieves a classification accuracy of \~90% on the test set. Additional evaluation metrics, such as precision, recall, and F1-score, can be calculated using scikit-learn's `classification_report` and `confusion_matrix`.

---

## Future Work

- Enhance the model with additional contextual embeddings.
- Explore fine-tuning ClinicalBERT for domain-specific data.
- Implement advanced tokenization techniques to handle complex medical terminology.

---

## Author

This project was implemented by Amandeep Yadav.

