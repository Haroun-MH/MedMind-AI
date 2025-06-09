# MedMind API

## Introduction

MedMind is a Flask-based web API designed to predict potential diseases based on symptoms provided by the user. It leverages a machine learning model trained on a dataset of symptoms and corresponding diseases. The API also provides additional information related to the predicted disease, including a description, common precautions, suggested medications, and workout recommendations.

This system aims to provide users with preliminary insights into their health conditions based on self-reported symptoms. **It is not a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Project Structure

```
MedMind/
├── Datasets/
│   ├── Symptom-severity.csv
│   ├── Training.csv
│   ├── description.csv
│   ├── diets.csv
│   ├── medications.csv
│   ├── precautions_df.csv
│   ├── symtoms_df.csv
│   └── workout_df.csv
├── Models/
│   └── svc_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

-   `Datasets/`: Contains all the CSV files used for training the model and providing supplementary information (descriptions, precautions, etc.).
-   `Models/`: Stores the trained machine learning model (`svc_model.pkl`).
-   `app.py`: The main Flask application file containing the API logic.
-   `requirements.txt`: Lists the Python dependencies required to run the project.
-   `README.md`: This file.

## Setup and Running the Application

Follow these steps to set up and run the MedMind API locally:

1.  **Prerequisites:**
    *   Python 3.x installed.
    *   `pip` (Python package installer) installed.

2.  **Clone the Repository (if applicable):**
    If you have this project in a Git repository, clone it to your local machine.
    ```bash
    git clone <repository_url>
    cd MedMind
    ```


3.  **Install Dependencies:**
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask Application:**
    Navigate to the project's root directory (where `app.py` is located) and run:
    ```bash
    python app.py
    ```
    The application will typically start on `http://127.0.0.1:5000/`.

## API Endpoints

The API currently exposes one main endpoint for disease prediction.

### 1. Predict Disease

*   **URL:** `/predict`
*   **Method:** `POST`
*   **Description:** Accepts a list of symptoms (either as a comma-separated string or natural language text) and returns a predicted disease along with associated information.
*   **Request Body (JSON):**
    ```json
    {
        "symptoms": "I have an itching and skin rash, and I also feel chills"
    }
    ```
    Alternatively, for comma-separated exact symptoms (less preferred now):
    ```json
    {
        "symptoms": "itching,skin_rash,shivering"
    }
    ```

*   **Success Response (200 OK):**
    *   **Content-Type:** `application/json`
    *   **Body Example:**
        ```json
        {
            "predicted_disease": "Fungal infection",
            "description": "Fungal infection is a common skin condition caused by fungi.",
            "precautions": [
                "bathe twice",
                "use dettol or neem in bathing water",
                "keep infected area dry",
                "wear clean cloths"
            ],
            "medications": [
                "Antifungal Cream",
                "Fluconazole",
                "Terbinafine",
                "Clotrimazole"
            ],
            "recommendations": [
                "Avoid sugary foods",
                "Consume probiotics",
                "Increase intake of garlic",
                "Include coconut oil in diet"
            ],
            "symptom_severity": {
                "itching": 1,
                "skin_rash": 3,
                "shivering": 3,
                "overall_severity_score": 7
            }
        }
        ```

*   **Error Response (e.g., 400 Bad Request if symptoms are missing or invalid):**
    *   **Content-Type:** `application/json`
    *   **Body Example (Symptoms missing):**
        ```json
        {
            "error": "'symptoms' field is required."
        }
        ```
    *   **Body Example (No known symptoms identified):**
        ```json
        {
            "error": "Could not identify any known symptoms from the input provided. Please describe your symptoms more clearly or check for typos.",
            "unknown_symptoms_provided": [] 
        }
        ```
