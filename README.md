# UK-Car-Price-Predication

## UK Used Cars Price Prediction

**Problem Statement:**

The objective is to develop a predictive model for used car prices in the UK market, helping both buyers and sellers make informed decisions by estimating fair market values based on various vehicle characteristics and market conditions.

**Dataset Column Descriptions:**

1. `title`: The manufacturer and model name of the vehicle (e.g., "SKODA Fabia", "Vauxhall Corsa"), representing the brand identity and specific model variant.

2. `Price`: The selling price of the used car in British Pounds (£), which is our target variable for prediction.

3. `Mileage(miles)`: The total distance the car has been driven in miles, a crucial factor affecting the vehicle's value and condition.

4. `Registration_Year`: The year when the car was first registered, indicating the age of the vehicle.

5. `Previous Owners`: The number of previous owners the vehicle has had, which can impact its market value and perceived reliability.

6. `Fuel type`: The type of fuel the vehicle uses (Petrol, Diesel, Hybrid, Electric), reflecting operating costs and environmental impact.

7. `Body type`: The style/design of the vehicle (Hatchback, Saloon, SUV, etc.), indicating its purpose and utility.

8. `Engine`: The engine size in liters (e.g., 1.4L, 2.0L), representing the vehicle's power capacity and fuel efficiency.

9. `Gearbox`: The transmission type (Manual or Automatic), which affects driving experience and maintenance costs.

10. Additional features include `Doors` (number of doors), `Seats` (seating capacity), `Emission Class` (Euro rating for emissions standards), and `Service history` (maintenance record status), all contributing to the overall value assessment of the vehicle.

#### Problem description
Used Car Prices in UK Dataset is a comprehensive collection of automotive information extracted from the popular automotive marketplace website, autotrader.co.uk. This dataset comprises 3,685 data points, each representing a unique vehicle listing, and includes thirteen distinct features providing valuable insights into the world of automobiles. 
- Want to use this dataset to predict the used car price. 
- Dataset: https://www.kaggle.com/datasets/muhammadawaistayyab/used-cars-prices-in-uk

Feature description:

- title
- Price : price of car in pounds
- Mileage(miles)
- Registration(year)
- Previous Owners
- Fuel Type
- Body Type
- Engine
- Gearbox
- Seats
- Doors
- Emission Class
- Service history

#### EDA

Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis) For images: analyzing the content of the images. For texts: frequent words, word clouds, etc

#### LinearRegression

```
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

Training MSE: 3572586.765435099
Validation MSE: 3584094.055822923
Test MSE: 3792559.4621593314

Training R2: 0.8219588410220839
Validation R2: 0.8281110867863548
Test R2: 0.8166468026624932

- For ROC curve, we need to convert the regression problem into a binary classification
- Let's use the median price as a threshold
```
median_price = y.median()
y_train_binary = (y_train > median_price).astype(int)
y_val_binary = (y_val > median_price).astype(int)
y_test_binary = (y_test > median_price).astype(int)
```
```
# Calculate ROC curves
fpr_train, tpr_train, _ = roc_curve(y_train_binary, y_train_pred)
fpr_val, tpr_val, _ = roc_curve(y_val_binary, y_val_pred)
fpr_test, tpr_test, _ = roc_curve(y_test_binary, y_test_pred)
```

```
# Calculate AUC scores
auc_train = auc(fpr_train, tpr_train)
auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)
```

AUC Scores:
Training AUC: 0.9725072349381741
Validation AUC: 0.9802731645550075
Test AUC: 0.9647207120817254

#### DecisionTreeRegressor:

```
# Split the data into train, validation, and test sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create and train decision trees with different parameters
max_depths = [3, 5, 7, 10, 15, None]
min_samples_splits = [2, 5, 10]
results = []

for depth in max_depths:
    for min_samples_split in min_samples_splits:
        # Create and train the model
        regressor = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        regressor.fit(X_train_scaled, y_train)
        
        # Make predictions on validation set
        y_val_pred = regressor.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        results.append({
            'max_depth': depth,
            'min_samples_split': min_samples_split,
            'val_mse': val_mse,
            'val_r2': val_r2,
            'model': regressor
        })
        
        print(f"\nDecision Tree (max_depth={depth}, min_samples_split={min_samples_split}):")
        print(f"Validation MSE: {val_mse:.2f}")
        print(f"Validation R2: {val_r2:.4f}")

# Find best model based on validation MSE
best_result = min(results, key=lambda x: x['val_mse'])
best_regressor = best_result['model']

print(f"\nBest model parameters:")
print(f"Max depth: {best_result['max_depth']}")
print(f"Min samples split: {best_result['min_samples_split']}")
print(f"Validation MSE: {best_result['val_mse']:.2f}")
print(f"Validation R2: {best_result['val_r2']:.4f}")

# Evaluate best model on test set
y_test_pred = best_regressor.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nTest Set Performance:")
print(f"MSE: {test_mse:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"R2: {test_r2:.4f}")

# Plot MSE comparison
plt.figure(figsize=(12, 6))
for min_samples_split in min_samples_splits:
    mse_values = [r['val_mse'] for r in results if r['min_samples_split'] == min_samples_split]
    plt.plot([str(d) for d in max_depths], mse_values, marker='o', label=f'min_samples_split={min_samples_split}')

plt.xlabel('Max Depth')
plt.ylabel('Validation MSE')
plt.title('Model Validation MSE vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_regressor.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()

# Scatter plot of predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices (Test Set)')
plt.tight_layout()
plt.show()
```

#### Dependency and environment management

- train.py
- predict.py
- Pipenv and Pipenv.lock
- Dockerfile

```
# Function to predict price for a new car
def predict_car_price(car_features):
    # Convert the input features to match the training data format
    car_df = pd.DataFrame([car_features])
    # Encode categorical variables
    for column in categorical_columns:
        if column in car_df.columns:
            car_df[column] = label_encoders[column].transform(car_df[column])
    
    # Scale features
    car_scaled = scaler.transform(car_df)
    
    # Make prediction
    return best_regressor.predict(car_scaled)[0]

# Example usage
example_car_1 = {
    'mileage': 12203,
    'registration_year': 2009,
    'previous_owners': 1,
    'engine': 2.0,
    'doors': 5,
    'seats': 5,
    'service_history': True
}
predicted_price = predict_car_price(example_car_1)
print(f"\nPredicted price for the example car: £{predicted_price:.2f}")
```


 
