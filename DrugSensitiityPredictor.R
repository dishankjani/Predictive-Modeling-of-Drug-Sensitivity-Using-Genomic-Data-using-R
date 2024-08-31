library("caret")
library("xgboost")
library("ggplot2")
library("dplyr")
library("randomForest")
#install.packages("randomForest")
library("xgboost")
install.packages("xgboost")

#Read the dataset, here because the csv file is in same folder there is no need of mentioning path otherwise we need to mention path as well
data <- read.csv("GDSC_DATASET.csv", header = TRUE, sep = ",")
head(data)

#Step1: EDA
summary(data)

hist(data$LN_IC50)

# Handling missing values
data <- na.omit(data)  # Remove rows with NA values
summary(data)

#Check for duplication
uplicated_rows <- data[duplicated(data), ]
print(uplicated_rows) #The output is 0, thus we do not have any duplicated rows

#shows 19 columns and 242035 rows
shape<-dim(data)
shape

ggplot(data, aes(x = Gene.Expression, y = LN_IC50)) +
  geom_point() +
  geom_smooth(method = "lm")

#Correlation score tells how much one variable is related to other variable
cor(data$Z_SCORE, data$LN_IC50)

#Step 2: Data Preprocessing
head(data)
# Handle missing values by imputing with median
preProcess_missingdata <- preProcess(data, method = 'medianImpute')
data_imputed <- predict(preProcess_missingdata, newdata = data)

# Separate features (X) and target (y)
X <- data_imputed[, !names(data_imputed) %in% c("LN_IC50")]
y <- data_imputed$LN_IC50

# Normalize the features
preProcess_normalize <- preProcess(X, method = c("center", "scale"))
X_normalized <- predict(preProcess_normalize, X)

# Split the data into training and test sets
set.seed(42)
trainIndex <- createDataPartition(y, p = .8, list = FALSE, times = 1)
X_train <- X_normalized[trainIndex,]
X_test <- X_normalized[-trainIndex,]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]


# Apply PCA for dimensionality reduction (optional)
pca_model <- preProcess(X_train, method = "pca", thresh = 0.95)
X_train_pca <- predict(pca_model, X_train)
X_test_pca <- predict(pca_model, X_test)


#Training Model using Random Forest
# Train a Random Forest model
rf_model <- randomForest(X_train_pca, y_train, ntree = 500, importance = TRUE)

# Evaluate the model using cross-validation
set.seed(42)
train_control <- trainControl(method = "cv", number = 10)
rf_cv <- train(X_train_pca, y_train, method = "rf", trControl = train_control, ntree = 500)

# Predict on the test set
y_pred <- predict(rf_model, X_test_pca)

# Calculate performance metrics
mse <- mean((y_pred - y_test)^2)
r_squared <- 1 - (sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2))

print(paste("MSE:", mse))
print(paste("R-squared:", r_squared))

#Now we can develop the similar model with XGBoost and compare performances
#Now we can develop the similar model with XGBoost and compare performances
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

# Set parameters for xgboost
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6,
  eta = 0.1
)

# Train the model
xgb_model <- xgb.train(params, dtrain, nrounds = 100)

# Calculate SHAP values for your training data
shap_values <- shap.values(xgb_model, X_train)

# Extract the SHAP values and the corresponding feature names
shap_scores <- shap_values$shap_score
colnames(shap_scores) <- colnames(X_train)

shap.plot.summary(shap_scores, X_train)
