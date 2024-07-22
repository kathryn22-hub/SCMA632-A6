options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Load necessary libraries
library(quantmod)
library(ggplot2)
library(forecast)
library(Metrics)
library(tseries)

# Fetch the data
# Load necessary libraries
library(quantmod)

# Set the ticker symbol
ticker <- "INFY"

# Set the start and end dates
start_date <- "2021-04-01"
end_date <- "2024-06-30"

# Download the data
getSymbols(ticker, src = "yahoo", from = start_date, to = end_date)

# The data will be stored in an object named after the ticker symbol
data <- get(ticker)

# Display the first few rows of the data
head(data)


# Select the target variable Adj Close
df <- data[, "INFY.Adjusted"]

# Check for missing values
cat("Missing values:\n")
print(sum(is.na(df)))

# Plot the data
ggplot(data = as.data.frame(df), aes(x = index(df), y = df)) +
  geom_line() +
  labs(title = "INFY Adj Close Price", x = "Date", y = "Adj Close Price") +
  theme_minimal()

# Decompose the time series
result <- decompose(ts(df, frequency = 12), type = "multiplicative")

# Plot the decomposed components
plot(result)


# Resample the data to monthly frequency
monthly_data <- to.monthly(df, indexAt = "lastof", OHLC = FALSE)

# Convert to a time series object with monthly frequency
ts_data <- ts(monthly_data, frequency = 12, start = c(2021, 4))

# Split the data into training and test sets
train_data <- window(ts_data, end = c(2023, 6))
test_data <- window(ts_data, start = c(2023, 7))

# Fit the Holt-Winters model
holt_winters_model <- hw(train_data, seasonal = "multiplicative", h = 12)

# Forecast for the next year (12 months)
holt_winters_forecast <- forecast(holt_winters_model, h = 12)

# Plot the forecast
autoplot(holt_winters_forecast) +
  labs(title = "Holt-Winters Forecast", x = "Date", y = " Adjusted  Close Price")

# Compute RMSE, MAE, MAPE, and R-squared for Holt-Winters
rmse_hw <- rmse(test_data, holt_winters_forecast$mean)
mae_hw <- mae(test_data, holt_winters_forecast$mean)
mape_hw <- mape(test_data, holt_winters_forecast$mean) * 100
r2_hw <- cor(test_data, holt_winters_forecast$mean)^2

cat(sprintf("RMSE: %.2f\nMAE: %.2f\nMAPE: %.2f\nR-squared: %.2f\n", rmse_hw, mae_hw, mape_hw, r2_hw))

# Fit auto.arima model
arima_model <- auto.arima(train_data, seasonal = TRUE)

# Print the model summary
summary(arima_model)

# Number of periods to forecast
n_periods <- length(test_data)

# Generate forecast
forecast_arima <- forecast(arima_model, h = n_periods)

# Plot the original data, fitted values, and forecast
autoplot(forecast_arima) +
  labs(title = "Auto ARIMA Forecasting", x = "Date", y = "Value")

# Compute RMSE, MAE, MAPE, and R-squared for ARIMA
rmse_arima <- rmse(test_data, forecast_arima$mean)
mae_arima <- mae(test_data, forecast_arima$mean)
mape_arima <- mape(test_data, forecast_arima$mean) * 100
r2_arima <- cor(test_data, forecast_arima$mean)^2

cat(sprintf("RMSE: %.2f\nMAE: %.2f\nMAPE: %.2f\nR-squared: %.2f\n", rmse_arima, mae_arima, mape_arima, r2_arima))

# Forecast for the next 60 days (daily data)
daily_data <- df
arima_model_daily <- auto.arima(daily_data, seasonal = TRUE)

# Generate in-sample predictions
fitted_values <- fitted(arima_model_daily)

# Number of periods to forecast
n_periods <- 60

# Generate forecast
forecast_daily <- forecast(arima_model_daily, h = n_periods)

# Create future dates index
future_dates <- seq(end(daily_data), by = "days", length.out = n_periods)

# Convert forecast to a DataFrame with future_dates as the index
forecast_df <- data.frame(Date = future_dates, Forecast = forecast_daily$mean)
conf_int_df <- data.frame(Date = future_dates, 
                          lower_bound = forecast_daily$lower[,2], 
                          upper_bound = forecast_daily$upper[,2])

# Plot the original data, fitted values, and forecast
ggplot() +
  geom_line(data = as.data.frame(daily_data), aes(x = index(daily_data), y = daily_data), color = 'blue') +
  geom_line(data = forecast_df, aes(x = Date, y = Forecast), color = 'green') +
  geom_ribbon(data = conf_int_df, aes(x = Date, ymin = lower_bound, ymax = upper_bound), alpha = 0.2) +
  labs(title = "Auto ARIMA Forecasting", x = "Date", y = "Value") +
  theme_minimal()


# Load necessary libraries
library(caret)
library(tidyverse)

install.packages("reticulate")
library(reticulate)


# Install Keras
install.packages("keras")
library(keras)

install.packages("tensorflow")
library(tensorflow)

library(randomForest)
library(Metrics)

# Set the ticker symbol
ticker <- "INFY"

# Set the start and end dates
start_date <- "2021-04-01"
end_date <- "2024-06-30"

# Download the data
getSymbols(ticker, src = "yahoo", from = start_date, to = end_date)

# The data will be stored in an object named after the ticker symbol
data <- get(ticker)

# Select the 'Adjusted' column
df <- data[, "INFY.Adjusted"]

# Convert to a data frame
df <- data.frame(Date = index(df), coredata(df))
colnames(df) <- c("Date", "Adj_Close")

# Initialize MinMaxScaler
scaler <- preProcess(df[, "Adj_Close", drop = FALSE], method = "range")

# Scale the data
scaled_df <- df %>% mutate(Adj_Close = predict(scaler, df[, "Adj_Close", drop = FALSE]))

# Create sequences
create_sequences <- function(data, target_col, sequence_length) {
  sequences <- list()
  labels <- numeric()
  for (i in 1:(nrow(data) - sequence_length)) {
    sequences[[i]] <- data[i:(i + sequence_length - 1), ]
    labels[i] <- data[i + sequence_length, target_col]
  }
  X <- array(unlist(sequences), dim = c(length(sequences), sequence_length, ncol(data)))
  y <- array(labels, dim = c(length(labels), 1))
  list(X = X, y = y)
}

# Convert data frame to matrix
data_matrix <- as.matrix(scaled_df[,-1])

# Define the target column index and sequence length
target_col <- ncol(data_matrix)
sequence_length <- 30

# Create sequences
seq_data <- create_sequences(data_matrix, target_col, sequence_length)
X <- seq_data$X
y <- seq_data$y

cat("Shape of X:", dim(X), "\n")
cat("Shape of y:", dim(y), "\n")

# Split the data into training and testing sets (80% training, 20% testing)
train_size <- floor(0.8 * dim(X)[1])
X_train <- X[1:train_size,,]
y_train <- y[1:train_size,]
X_test <- X[(train_size + 1):dim(X)[1],,]
y_test <- y[(train_size + 1):dim(X)[1],]

install_tensorflow()

# Build the LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(sequence_length, ncol(data_matrix))) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = "adam",
  loss = "mean_squared_error"
)

model %>% summary()

# Train the model
history <- model %>% fit(
  X_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(X_test, y_test),
  shuffle = FALSE
)

# Evaluate the model
loss <- model %>% evaluate(X_test, y_test)
cat("Test Loss:", loss, "\n")

# Predict on the test set
y_pred <- model %>% predict(X_test)

# Inverse transform the predictions and true values to get them back to the original scale
y_test_scaled <- predict(scaler, data.frame(Adj_Close = y_test))[, 1]
y_pred_scaled <- predict(scaler, data.frame(Adj_Close = y_pred))[, 1]

# Print some predictions and true values
cat("Predictions vs True Values:\n")
for (i in 1:10) {
  cat(sprintf("Prediction: %.2f, True Value: %.2f\n", y_pred_scaled[i], y_test_scaled[i]))
}

# Compute performance metrics
rmse <- rmse(y_test_scaled, y_pred_scaled)
mae <- mae(y_test_scaled, y_pred_scaled)
mape <- mape(y_test_scaled, y_pred_scaled)
r2 <- R2(y_test_scaled, y_pred_scaled)

cat(sprintf("RMSE: %.2f\nMAE: %.2f\nMAPE: %.2f\nR-squared: %.2f\n", rmse, mae, mape, r2))

# Plot the predictions vs true values
plot(y_test_scaled, type = "l", col = "blue", lwd = 2, xlab = "Time", ylab = "Close Price", main = "LSTM: Predictions vs True Values")
lines(y_pred_scaled, col = "red", lwd = 2)
legend("topright", legend = c("True Values", "LSTM Predictions"), col = c("blue", "red"), lty = 1, lwd = 2)

# Decision Tree and Random Forest Model
create_sequences_flatten <- function(data, target_col, sequence_length) {
  sequences <- list()
  labels <- numeric()
  for (i in 1:(nrow(data) - sequence_length)) {
    sequences[[i]] <- as.vector(data[i:(i + sequence_length - 1), ])
    labels[i] <- data[i + sequence_length, target_col]
  }
  X <- do.call(rbind, sequences)
  y <- labels
  list(X = X, y = y)
}

# Create sequences
seq_data_flat <- create_sequences_flatten(data_matrix, target_col, sequence_length)
X_flat <- seq_data_flat$X
y_flat <- seq_data_flat$y

# Split the data into training and testing sets (80% training, 20% testing)
train_size <- floor(0.8 * nrow(X_flat))
X_train_flat <- X_flat[1:train_size, ]
y_train_flat <- y_flat[1:train_size]
X_test_flat <- X_flat[(train_size + 1):nrow(X_flat), ]
y_test_flat <- y_flat[(train_size + 1):length(y_flat)]

# Train Decision Tree model
dt_model <- train(
  X_train_flat, y_train_flat,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10)
)

# Make predictions
y_pred_dt <- predict(dt_model, X_test_flat)

# Evaluate the model
rmse_dt <- rmse(y_test_flat, y_pred_dt)
mae_dt <- mae(y_test_flat, y_pred_dt)
mape_dt <- mape(y_test_flat, y_pred_dt)
r2_dt <- R2(y_test_flat, y_pred_dt)

cat(sprintf("Decision Tree - RMSE: %.2f\nMAE: %.2f\nMAPE: %.2f\nR-squared: %.2f\n", rmse_dt, mae_dt, mape_dt, r2_dt))

# Plot the predictions vs true values for Decision Tree
plot(y_test_flat, type = "l", col = "blue", lwd = 2, xlab = "Time", ylab = "Close Price", main = "Decision Tree: Predictions vs True Values")
lines(y_pred_dt, col = "red", lwd = 2)
legend("topright", legend = c("True Values", "Decision Tree Predictions"), col = c("blue", "red"), lty = 1, lwd = 2)

# Train Random Forest model
rf_model <- train(
  X_train_flat, y_train_flat,
  method = "rf",
  trControl = trainControl(method = "cv", number = 10)
)

# Make predictions
y_pred_rf <- predict(rf_model, X_test_flat)

# Evaluate the model
rmse_rf <- rmse(y_test_flat, y_pred_rf)
mae_rf <- mae(y_test_flat, y_pred_rf)
mape_rf <- mape(y_test_flat, y_pred_rf)
r2_rf <- R2(y_test_flat, y_pred_rf)

cat(sprintf("Random Forest - RMSE: %.2f\nMAE: %.2f\nMAPE: %.2f\nR-squared: %.2f\n", rmse_rf, mae_rf, mape_rf, r2_rf))

# Plot the predictions vs true values for Random Forest
plot(y_test_flat, type = "l", col = "blue", lwd = 2, xlab = "Time", ylab = "Close Price", main = "Random Forest: Predictions vs True Values")
lines(y_pred_rf, col = "red", lwd = 2)
legend("topright", legend = c("True Values", "Random Forest Predictions"), col = c("blue", "red"), lty = 1, lwd = 2)

# Plot both Decision Tree and Random Forest predictions together
plot(y_test_flat, type = "l", col = "blue", lwd = 2, xlab = "Time", ylab = "Close Price", main = "Decision Tree & Random Forest: Predictions vs True Values")
lines(y_pred_dt, col = "red", lwd = 2)
lines(y_pred_rf, col = "green", lwd = 2)
legend("topright", legend = c("True Values", "Decision Tree Predictions", "Random Forest Predictions"), col = c("blue", "red", "green"), lty = 1, lwd = 2)


