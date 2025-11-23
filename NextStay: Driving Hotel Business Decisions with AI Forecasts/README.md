_**Link Page**:_  https://123456789hien.github.io/nndl/NextStay:%20Driving%20Hotel%20Business%20Decisions%20with%20AI%20Forecasts/

_Build a fully client-side hotel cancellation forecasting tool using TensorFlow.js.
The app must support four modules: Data Upload & EDA, LSTM Training, Model Download, and Prediction Dashboard.
Everything runs in-browser; no backend or server storage._

**1️⃣ Module 1 — Interactive Data Upload & EDA**

Allow the user to upload two CSVs: train.csv and test.csv (semicolon-separated).

Automatically perform:
✓ missing value cleaning
✓ normalization of numeric fields
✓ merging train/test structure

Show:
✓ first 10 rows preview
✓ line chart (cancellation_rate over time)
✓ histogram (avg_room_price)
✓ correlation heatmap

Charts must update interactively based on the uploaded dataset.

**2️⃣ Module 2 — Train LSTM Model in Browser**

Train an LSTM neural network (3 layers × 50 neurons each) on the uploaded train dataset.

Inputs: normalized monthly features; Output: predicted cancellation_rate.

Allow adjustable hyperparameters: Epochs and Batch Size.

Show live metrics: Train RMSE and Test RMSE.

After training, automatically generate and download the model files:
• model.json (architecture)
• weights.bin (learned weights)

Ensure reproducibility: the same CSV → same trained model → same downloaded files.

**3️⃣ Module 3 — Predictive Decision Support**

Allow users to re-upload the trained model (model.json + weights.bin).

Once loaded, allow selecting:
• Room_Type
• Target Year
• Target Month

Compute next-month cancellation_rate using model.predict().

Assign risk levels:
• High > 50%
• Medium 20–50%
• Low < 20%

Display results in an interactive table summarizing risk across room types.

**4️⃣ Module 4 — Business Integration Layer**

Provide a business-facing explanation of each prediction:
• High risk → apply prepayment policies
• Medium risk → offer flexible cancellation discounts
• Low risk → promote upsell or long-stay offers

Present predictions as operational decision support for hotel managers.

Emphasize real-time retraining and fully in-browser operation (no backend).

**Overall Requirements**

Entire pipeline must run locally in browser: Upload → EDA → Train → Download → Predict.

No server calls, no external storage, no API usage.

Output must feel like a professional prototype app for real hotel chains.
