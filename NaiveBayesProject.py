import pandas as pd
import json
from collections import defaultdict

class SimpleNaiveBayesClassifier:
    def __init__(self):
        self.class_prior_probabilities = {}
        self.conditional_probabilities = defaultdict(lambda: defaultdict(float))

    def fit(self, training_data):
        total_samples = len(training_data)
        class_frequencies = training_data['PlayTennis'].value_counts()

        self.class_prior_probabilities = {
            class_label: class_count / total_samples
            for class_label, class_count in class_frequencies.items()
        }

        for class_label in class_frequencies.index:
            class_specific_data = training_data[training_data['PlayTennis'] == class_label]
            total_class_samples = len(class_specific_data)

            for feature_name in training_data.columns[:-1]:
                unique_feature_values = training_data[feature_name].unique()
                feature_value_counts = class_specific_data[feature_name].value_counts().to_dict()

                for feature_value in unique_feature_values:
                    self.conditional_probabilities[class_label][f"{feature_name}:{feature_value}"] = \
                        (feature_value_counts.get(feature_value, 0) + 1) / \
                        (total_class_samples + len(unique_feature_values))

        self._save_model()

    def predict(self, new_instance):
        model = self._load_model()
        class_scores = {
            class_label: model['class_prior_probabilities'][class_label]
            for class_label in model['class_prior_probabilities']
        }

        for class_label in class_scores:
            for feature_name, feature_value in new_instance.items():
                conditional_key = f"{feature_name}:{feature_value}"
                class_scores[class_label] *= model['conditional_probabilities'][class_label].get(
                    conditional_key, 1
                )

        return max(class_scores, key=class_scores.get)

    def _save_model(self):
        with open('naive_bayes_model.json', 'w') as model_file:
            json.dump({
                'class_prior_probabilities': self.class_prior_probabilities,
                'conditional_probabilities': self.conditional_probabilities
            }, model_file)

    def _load_model(self):
        with open('naive_bayes_model.json', 'r') as model_file:
            return json.load(model_file)

def load_dataset_from_json(file_path):
    return pd.read_json(file_path, lines=True)

def log_predictions(test_data, classifier, log_file='classification_log.txt'):
    with open(log_file, 'w') as log:
        correct_predictions = 0
        for _, row in test_data.iterrows():
            actual_class = row['PlayTennis']
            feature_values = row.drop('PlayTennis').to_dict()
            predicted_class = classifier.predict(feature_values)

            log.write(f"Features: {feature_values}\n")
            log.write(f"Actual Class: {actual_class}, Predicted Class: {predicted_class}\n\n")
            
            if predicted_class == actual_class:
                correct_predictions += 1

        accuracy = correct_predictions / len(test_data)
        log.write(f"Accuracy: {accuracy:.2%}\n")
        print(f"Accuracy: {accuracy:.2%}\n")


if __name__ == "__main__":
    dataset_path = 'play_tennis_dataset.json'
    training_data = load_dataset_from_json(dataset_path)

    naive_bayes_classifier = SimpleNaiveBayesClassifier()
    naive_bayes_classifier.fit(training_data)

    test_instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Weak'}
    predicted_class = naive_bayes_classifier.predict(test_instance)
    print(f"Predicted class: {predicted_class}")
    log_predictions(training_data, naive_bayes_classifier)
