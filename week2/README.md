_You must build a browser-only machine learning web app using TensorFlow.js that performs binary classification on the Kaggle Titanic dataset, runs fully client-side (no backend), and is suitable for hosting on GitHub Pages._

**Use these CDNs:**

TensorFlow.js: https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest

tfjs-vis: https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest

TailwindCSS: https://cdn.tailwindcss.com


**Same section structure and icons:**

**1. Data Schema**

Target: Survived
Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked,...
Ignore: PassengerId, Name, Ticket, Cabin

**2. Load CSV (train + test)**

Load CSV using FileReader.
Handle commas inside quotes correctly.

**3. Data Inspection**

_**Display:**_

Preview table

Missing value percentages

Stats & Visualizations

_**Two tfjs-vis bar charts:**_

Survival by Sex

Survival by Pclass

**4. Preprocessing**

Must implement fully:

Convert all numeric fields

Impute:

Age → median

Fare → median

Embarked → mode

Standardize:

Age, Fare

One-hot encode:

Sex

Pclass

Embarked

Add engineered features (commented as optional):

FamilySize = SibSp + Parch + 1

IsAlone = FamilySize === 1

Display final tensor shape

**5. Training**

80/20 stratified split

50 epochs

Batch size 32

Live tfjs-vis charts

Early stopping (patience=5)

Implement Stop Training button

**6. Metrics**

Compute AUC manually

Provide threshold slider

Update:

Accuracy

Precision

Recall

F1-score

**7. Prediction + Export**

Predict on test.csv:

Output probabilities

Output binary predictions using threshold

_**Export:**_

predictions.csv

_**Allow saving model:**_

titanic-tfjs-model.json

titanic-tfjs-model.weights.bin

_**Link page:**_ https://123456789hien.github.io/nndl/week2/
