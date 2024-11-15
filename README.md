### README.md


# Naive Bayes Classifier

This repository contains a Python implementation of a **Naive Bayes Classifier** from scratch. The project utilizes the "Play Tennis" dataset to demonstrate how probabilistic models can be used for classification tasks. The classifier is capable of handling categorical data and uses **Laplace smoothing** to address zero probability issues.

---

## Features
- Train a Naive Bayes model using a JSON dataset.
- Save and load the trained model in JSON format.
- Make predictions for new instances.
- Log predictions, actual classes, and accuracy in a detailed log file.

---

## Installation
1. Clone the repository:
   
   git clone <repository_url>
   cd naive-bayes-classifier

2. Install dependencies:
   
   pip install pandas
   

---

## Dataset
The "Play Tennis" dataset is used for training and testing. Each record contains weather-related features and a target variable (`PlayTennis`) indicating whether tennis was played.

### Dataset Format
The dataset should be saved in a JSON file (`play_tennis_dataset.json`) in the following format:


{"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"}
{"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"}


---

## Usage
1. **Train the Model**:
   The model is trained using the "play_tennis_dataset.json" file. Run the script to train and save the model:
   
   python naive_bayes_classifier.py
   

2. **Predict New Instances**:
   The script includes a hardcoded test instance for prediction:
   
   test_instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Weak'}
   
   When the script is run, the predicted class is displayed in the console:
   
   Predicted class: No
   

3. **Log Predictions**:
   After running the script, a "classification_log.txt" file is generated, containing:
   - Features of each instance.
   - Actual and predicted class labels.
   - Overall accuracy.

   **Example Log Output**:
   
   Features: {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Weak'}
   Actual Class: No, Predicted Class: No

   Accuracy: 93.33%
   

---

## File Structure

├── naive_bayes_classifier.py # Main script containing the classifier implementation
├── play_tennis_dataset.json  # Dataset file in JSON format
├── naive_bayes_model.json    # Saved model file (generated after training)
├── classification_log.txt    # Log file for predictions (generated after testing)
├── README.md                 # Project documentation


---

## How It Works
1. **Training**:
   - The model calculates:
     - **Prior probabilities**: Proportion of each class in the dataset.
     - **Conditional probabilities**: Likelihood of each feature value given a class.
   - Laplace smoothing is applied to prevent zero probabilities.

2. **Prediction**:
   - The trained model is loaded from "naive_bayes_model.json".
   - Class scores are computed using the Naive Bayes formula:
     \[
     P(Class | Features) \propto P(Class) \times \prod P(Feature | Class)
     \]
   - The class with the highest score is chosen as the prediction.

3. **Evaluation**:
   - The model's performance is assessed by comparing predicted labels with actual labels.
   - Accuracy is logged in `classification_log.txt`.

---

## Example
**Test Instance**:

test_instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Weak'}


**Predicted Output**:

Predicted class: No


---

## License
This project is licensed under the MIT License.

---

## Author
- **Name**: Necati Koçak
- **GitHub**: https://github.com/NecaTE4
