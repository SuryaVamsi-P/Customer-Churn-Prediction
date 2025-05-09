##########  GROUP - 10 Project   ##########

########## MEMBERS :- Surya vamsi Patiballa, Fonise Bratton ##########

##########   CUSTOMER CHURN PREDICTION   ##########


###   STEP :- 1 (Loading Packages)   ###


install.packages("dplyr")
library(dplyr)
install.packages("caret")
library(caret)
install.packages("ggcorrplot")
library(ggcorrplot)
library(tidyr)
library(data.table)
library(e1071)
library(randomForest)
library(xgboost)
library(catboost)
library(ROCR)
library(pROC)
library(rpart)            # (Decision Trees)
library(rpart.plot)       # (Plotting Decision Trees)
library(MLmetrics)
library(neuralnet)
library(gridExtra)
library(grid)
library(corrplot)
library(reshape2)

options(warn = -1)   # (Ignoring Warnings)

library(scales)        # (For Data Preprocessing)
library(labelEncoder)

library(ggcorrplot)     # (For Data Visualization)
library(gridExtra)
library(ggplot2)
install.packages("plotly")
library(plotly)

library(caret)         # (For Data train-test split and METRICS)

                        ###  Machine Learning Models  ###

library(rpart)             # (Decision tree)
library(randomForest)      # (Random Forest)
library(e1071)             # (SVM)
library(neuralnet)         # (Neural networks)
library(xgboost)           # (XGBoost)
library(catboost)          # (CatBoost)
library(glmnet)            # (Logistic Regression)


###   STEP :- 2 (Loading Data)   ###

url <- "https://raw.githubusercontent.com/SuryaVamsi-P/Customer-Churn-Prediction/main/Data%20Set.csv"


SURdf <- read.csv(url, stringsAsFactors = FALSE)


###   STEP :- 3 (Understanding Data)   ###


head(SURdf)

dim(SURdf)   #  (SHAPE of Data set)

str(SURdf)   #  (STRUCTURE of Data frame (SURdf))

summary(SURdf)   #  (Additional SUMMARY of each column)

cat("Column names :-", colnames(SURdf), "\n")  #  (Displays Column names)

sapply(SURdf, class)   #  (Gets Data Types of each COLUMN)

cat("Target Variable  :-  CHURN","\n")


###   STEP :- 4 (Visualizing Missing values as a MATRIX)   ###


install.packages("naniar")
library(naniar)

vis_miss(SURdf)   #  (Gives a Heat map - like Visualization)

cat("As you see here from the above Output, there are no black spots. There are NO DIRECT MISSING VALUES.")


###  STEP :- 5 (Data Preprocessing & Manipulation)


SURdf <- SURdf %>% select(-customerID)  #  (Dropping "customerID" column)
head(SURdf)
cat("We can find some indirect missingness in our data, which can be in form of blankspaces.")


SURdf$TotalCharges <- as.numeric(SURdf$TotalCharges) # (Check for missing values in each column)
colSums(is.na(SURdf))


missing_totalcharges <- SURdf[is.na(SURdf$TotalCharges), ] # (Filter rows where 'TotalCharges' is NA)
missing_totalcharges  # (View the rows with missing 'TotalCharges')


tenure_zero_indices <- which(SURdf$tenure == 0)  #  (Get the indices of rows where 'tenure' is equal to 0)
tenure_zero_indices


SURdf <- SURdf[SURdf$tenure != 0, ]  #  (Drop rows where 'tenure' is equal to 0)
which(SURdf$tenure == 0)


#  (Replace NA values in 'TotalCharges' with the mean of 'TotalCharges')
SURdf$TotalCharges[is.na(SURdf$TotalCharges)] <- mean(SURdf$TotalCharges, na.rm = TRUE)
head(SURdf)


colSums(is.na(SURdf))  # (Count the number of missing values (NA) in each column)


#  (Map values in the 'SeniorCitizen' column: 0 -> "No", 1 -> "Yes")
SURdf$SeniorCitizen <- ifelse(SURdf$SeniorCitizen == 0, "No", "Yes")
head(SURdf)


summary(SURdf$InternetService)  # (Summarize the 'InternetService' column)


#  (Summarize the specified numerical columns)
numerical_cols <- c("tenure", "MonthlyCharges", "TotalCharges")
summary(SURdf[, numerical_cols])


###   STEP :- 6 (Data Visualization and Plots)   ###


#  DONUT CHARTS


g_labels <- c("Male", "Female")
g_values <- table(SURdf$gender)

# Churn labels and values
c_labels <- c("No", "Yes")
c_values <- table(SURdf$Churn)

# Create the Gender donut chart
gender_pie <- plot_ly(
  labels = g_labels,
  values = as.numeric(g_values),
  type = 'pie',
  hole = 0.4,
  textinfo = 'percent+label',
  name = "Gender",
  marker = list(colors = c('#1f77b4', '#ff7f0e')) # Custom colors
)

# Create the Churn donut chart
churn_pie <- plot_ly(
  labels = c_labels,
  values = as.numeric(c_values),
  type = 'pie',
  hole = 0.4,
  textinfo = 'percent+label',
  name = "Churn",
  marker = list(colors = c('#2ca02c', '#9467bd')) # Custom colors
)

fig <- subplot(
  gender_pie,
  churn_pie,
  nrows = 1, 
  shareX = FALSE, 
  shareY = FALSE
)

fig <- fig %>%
  layout(
    title = "Gender and Churn Distributions",
    showlegend = TRUE,
    annotations = list(
      list(text = "Gender", x = 0.2, y = 0.5, font = list(size = 20), showarrow = FALSE),
      list(text = "Churn", x = 0.8, y = 0.5, font = list(size = 20), showarrow = FALSE)
    )
  )

fig


result <- SURdf %>%
  filter(Churn == "No") %>%       # (Filter rows where 'Churn' is "No")
  group_by(gender) %>%            # (Group by 'gender')
  summarise(count = n())          # (Count the number of rows in each group)

print(result)


res <- SURdf %>%
  filter(Churn == "Yes") %>%      # (Filter rows where 'Churn' is "Yes")
  group_by(gender) %>%            # (Group by 'gender')
  summarise(count = n())          # (Count the number of rows in each group)

print(res)


#  OUTER DONUT CHARTS


outer_labels <- c("Churn: Yes", "Churn: No")
outer_values <- c(1869, 5163)
outer_colors <- c('#ff6666', '#66b3ff') 

# Data for the inner layer (Gender within Churn)
inner_labels <- c("F", "M", "F", "M")
inner_values <- c(939, 930, 2544, 2619)
inner_colors <- c('#ff9999', '#ffb3e6', '#9999ff', '#b3d9ff')

# Outer Donut Chart (Churn)
outer_pie <- plot_ly(
  labels = outer_labels,
  values = outer_values,
  type = 'pie',
  hole = 0.5,
  textinfo = 'label+percent',
  textfont = list(size = 15),
  marker = list(colors = outer_colors),
  domain = list(x = c(0, 1), y = c(0, 1))
)

# Inner Donut Chart (Gender)
inner_pie <- plot_ly(
  labels = inner_labels,
  values = inner_values,
  type = 'pie',
  hole = 0.7, 
  textinfo = 'label',
  textfont = list(size = 13),
  marker = list(colors = inner_colors),
  domain = list(x = c(0, 1), y = c(0, 1))
)


nested_donut <- subplot(outer_pie, inner_pie, nrows = 1) %>%
  layout(
    title = list(
      text = "Churn Distribution w.r.t Gender: Male(M), Female(F)",
      font = list(size = 18)
    ),
    showlegend = FALSE
  )

nested_donut


#  HISTOGRAM (GROUPED)


fig <- plot_ly(
  data = SURdf, 
  x = ~Churn,    
  color = ~Contract,  # (Grouped by the Contract column)
  type = "histogram", 
  barmode = "group"   # (Grouped bars)
)

fig <- fig %>%
  layout(
    title = list(text = "<b>Customer Contract Distribution<b>", font = list(size = 18)),
    xaxis = list(title = "Churn"),
    yaxis = list(title = "Count"),
    width = 700,
    height = 500,
    bargap = 0.1  # (Gap between bars)
  )

fig


#  DONUT CHARTS


# (Extract labels (unique PaymentMethod) and values (counts of each PaymentMethod) )
labels <- unique(SURdf$PaymentMethod)          # (Unique PaymentMethod values)
values <- table(SURdf$PaymentMethod)          # (Counts of each PaymentMethod)


fig <- plot_ly(
  labels = names(values),                     # (PaymentMethod categories)
  values = as.numeric(values),                # (Counts of each category)
  type = 'pie',
  hole = 0.3                                
)

fig <- fig %>%
  layout(
    title = list(text = "<b>Payment Method Distribution</b>", font = list(size = 18))
  )

fig


#  HISTOGRAM (GROUPED)


fig <- plot_ly(
  data = SURdf,                 
  x = ~Churn,                    
  color = ~PaymentMethod,        
  type = "bar"                   
)


fig <- fig %>%
  layout(
    title = "<b>Customer Payment Method distribution w.r.t. Churn</b>",
    xaxis = list(title = "Churn"),
    yaxis = list(title = "Count"),
    barmode = "stack",           
    width = 700,
    height = 500,
    bargap = 0.1                 
  )

fig


# (Get unique values from the 'InternetService' column)
unique(SURdf$InternetService)


# (Filter rows where gender is "Male" and count combinations of InternetService and Churn)
result <- SURdf %>%
  filter(gender == "Male") %>%                 
  count(InternetService, Churn) %>%          
  arrange(desc(n))                             

print(result)


# (Filter rows where gender is "Female" and count combinations of InternetService and Churn)
result <- SURdf %>%
  filter(gender == "Female") %>%                 
  count(InternetService, Churn) %>%          
  arrange(desc(n))                             

print(result)


#  BAR GRAPHS


x <- list(
  list("Churn:No", "Churn:No", "Churn:Yes", "Churn:Yes"),  # Outer group (Churn)
  list("Female", "Male", "Female", "Male")                # Inner group (Gender)
)

y_dsl <- c(965, 992, 219, 240)              # (Y values for DSL)
y_fiber <- c(889, 910, 664, 633)            # (Y values for Fiber optic)
y_no_internet <- c(690, 717, 56, 57)        # (Y values for No Internet)


fig <- plot_ly()

fig <- fig %>%
  add_trace(
    x = x, y = y_dsl, type = "bar", name = "DSL"
  ) %>%
  add_trace(
    x = x, y = y_fiber, type = "bar", name = "Fiber optic"
  ) %>%
  add_trace(
    x = x, y = y_no_internet, type = "bar", name = "No Internet"
  )

fig <- fig %>%
  layout(
    title = "<b>Churn Distribution w.r.t. Internet Service and Gender</b>",
    xaxis = list(title = "Churn and Gender"),
    yaxis = list(title = "Count"),
    barmode = "group"  
  )

fig


#  HISTOGRAM (Grouped)


color_map <- c("Yes" = "#FF97FF", "No" = "#AB63FA")

# (Create the grouped histogram)
fig <- plot_ly(
  data = SURdf,                
  x = ~Churn,                  
  color = ~Dependents,         
  colors = color_map,          
  type = "bar",                
  barmode = "group"            
)


fig <- fig %>%
  layout(
    title = list(
      text = "<b>Dependents Distribution</b>",
      font = list(size = 18)
    ),
    xaxis = list(title = "Churn"),
    yaxis = list(title = "Count"),
    width = 700,               
    height = 500,              
    bargap = 0.1               
  )

fig


# Define the custom color mapping for Partner (Yes/No)
color_map <- c("Yes" = "#FFA15A", "No" = "#00CC96")


fig <- plot_ly(
  data = SURdf,              
  x = ~Churn,                
  color = ~Partner,          
  type = "bar",              
  barmode = "group",         
  colors = color_map         
)


fig <- fig %>%
  layout(
    title = "<b>Churn Distribution w.r.t. Partners</b>",  
    xaxis = list(title = "Churn"),                       
    yaxis = list(title = "Count"),                       
    width = 700,                                         
    height = 500,                                        
    bargap = 0.1                                         
  )

fig


#  BAR CHARTS


# Define the custom color mapping for OnlineSecurity
color_map <- c("Yes" = "#FF97FF", "No" = "#AB63FA")  # (Mapping for OnlineSecurity categories)

fig <- plot_ly(
  data = SURdf,                      
  x = ~Churn,                        
  color = ~OnlineSecurity,           # Grouped by OnlineSecurity column
  type = "bar",                      
  colors = color_map,                
  barmode = "group"                  
)


fig <- fig %>%
  layout(
    title = "<b>Churn w.r.t Online Security</b>",  
    xaxis = list(title = "Churn"),                
    yaxis = list(title = "Count"),                
    width = 700,                                  
    height = 500,                                 
    bargap = 0.1                                  
  )

fig


# Define the custom color mapping for PaperlessBilling with red, yellow, and black
color_map <- c("Yes" = "red", "No" = "yellow")


fig <- plot_ly(
  data = SURdf,                      
  x = ~Churn,                         
  color = ~PaperlessBilling,          # Grouped by PaperlessBilling column
  type = "bar",                       # Use bar chart
  colors = color_map                  
)

fig <- fig %>%
  layout(
    title = "<b>Churn Distribution w.r.t. Paperless Billing</b>",  
    xaxis = list(title = "Churn"),                                
    yaxis = list(title = "Count"),                                
    width = 700,                                                 
    height = 500,                                                 
    bargap = 0.1                                                  
  )


fig


# Define the custom color mapping for TechSupport
color_map <- c("Yes" = "pink", "No" = "blue", "NA" = "brown") 


fig <- plot_ly(
  data = SURdf,                     
  x = ~Churn,                        
  color = ~TechSupport,              # Grouped by TechSupport column
  type = "bar",                      # Use bar chart
  colors = color_map,                
  barmode = "group"                  
)

fig <- fig %>%
  layout(
    title = "<b>Churn Distribution w.r.t. TechSupport</b>",  
    xaxis = list(title = "Churn"),                          
    yaxis = list(title = "Count"),                          
    width = 700,                                            
    height = 500,                                           
    bargap = 0.1                                            
  )

fig


# Define the custom color mapping for PhoneService
color_map <- c("Yes" = "green", "No" = "yellow", "No phone service" = "red")

fig <- plot_ly(
  data = SURdf,                      
  x = ~Churn,                        
  color = ~PhoneService,             # Grouped by PhoneService column
  type = "bar",                      # Use bar chart
  colors = color_map,                
  barmode = "group"                  
)

fig <- fig %>%
  layout(
    title = "<b>Churn Distribution w.r.t. Phone Service</b>",  
    xaxis = list(title = "Churn"),                            
    yaxis = list(title = "Count"),                            
    width = 700,                                              
    height = 500,                                             
    bargap = 0.1                                              
  )

fig


#  DENSITY PLOT


ggplot(SURdf, aes(x = MonthlyCharges, fill = Churn)) +
  geom_density(alpha = 0.5) +            # Add density plots with transparency
  scale_fill_manual(values = c("No" = "black", "Yes" = "pink")) + 
  labs(
    title = "Distribution of Monthly Charges by Churn",
    x = "Monthly Charges",
    y = "Density",
    fill = "Churn"                      
  ) +
  theme_minimal() +                      
  theme(
    text = element_text(size = 12),     
    legend.position = "top"             
  )


#  DENSITY PLOTS


ggplot(SURdf, aes(x = TotalCharges, fill = Churn)) +
  geom_density(alpha = 0.5) +            # Add density plots with transparency
  scale_fill_manual(values = c("No" = "orange", "Yes" = "red")) + 
  labs(
    title = "Distribution of Total Charges by Churn",
    x = "Total Charges",
    y = "Density",
    fill = "Churn"                      
  ) +
  theme_minimal() +                     
  theme(
    text = element_text(size = 12),     
    legend.position = "top"             
  )


#  BOX PLOTS


fig <- plot_ly(
  data = SURdf,               
  x = ~Churn,                  
  y = ~tenure,                 
  type = "box",                # Specify a boxplot
  marker = list(color = "black")  
)


fig <- fig %>%
  layout(
    title = list(
      text = "<b>Tenure vs Churn</b>", 
      font = list(size = 25, family = "Courier")  
    ),
    xaxis = list(title = "Churn"),               
    yaxis = list(title = "Tenure (Months)"),     
    width = 750,                                 
    height = 600                                 
  )

fig


#  CORRELATION MATRIX


# Convert categorical variables to numeric factors
numeric_df <- as.data.frame(lapply(SURdf, function(x) {
  if (is.factor(x) || is.character(x)) {
    as.numeric(as.factor(x))  # Convert factors or characters to numeric
  } else {
    x
  }
}))

# Compute the correlation matrix
corr <- cor(numeric_df, use = "pairwise.complete.obs")

# Create a clearer heatmap with better formatting
ggcorrplot(corr, 
           method = "square", 
           lab = TRUE,                      # Add correlation coefficients
           lab_size = 2.5,                    # Adjust text size of coefficients
           colors = c("blue", "white", "red"), 
           outline.color = "white", 
           ggtheme = theme_minimal()) +
  labs(title = "Correlation Heatmap", 
       subtitle = "Factorized Numeric Data") +
  theme(
    plot.title = element_text(size = 10, face = "bold", hjust = 0.5),   
    plot.subtitle = element_text(size = 10, hjust = 0.5),              
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7),      
    axis.text.y = element_text(size = 7)                              
  ) +
  scale_x_discrete(expand = expansion(mult = c(0.01, 0.01))) +         
  scale_y_discrete(expand = expansion(mult = c(0.01, 0.01)))  



###   STEP :- 7 (Test and Train Data Split)   ###


#   Splitting data into Train and Testing


# (Function to convert character or factor columns to integers)

object_to_int <- function(column) {
  if (is.character(column) || is.factor(column)) {
    column <- as.numeric(as.factor(column)) - 1  # (Convert to numeric and subtract 1 to start from 0)
  }
  return(column)
}

SURdf <- SURdf %>%
  mutate(across(where(is.character), object_to_int))  # (Applying to all character columns)


# (Define the object_to_int function)

object_to_int <- function(column) {
  if (is.character(column) || is.factor(column)) {
    column <- as.numeric(as.factor(column)) - 1  # (Converting to numeric starting from 0)
  }
  return(column)
}

SURdf <- as.data.frame(lapply(SURdf, object_to_int))

head(SURdf)


# (Ensuring 'Churn' is numeric for correlation calculation)

if (!is.numeric(SURdf$Churn)) {
  SURdf$Churn <- as.numeric(as.factor(SURdf$Churn)) - 1 
}

                                       # (Select only numeric columns for correlation calculation)
numeric_cols <- sapply(SURdf, is.numeric)
numeric_df <- SURdf[, numeric_cols]  

                                        # (Calculate correlations for numeric columns)
correlations <- cor(numeric_df, use = "pairwise.complete.obs")

                                         # (Extract correlations with 'Churn')
if ("Churn" %in% colnames(correlations)) {
  churn_correlations <- correlations[, "Churn"]
  
  sorted_correlations <- sort(churn_correlations, decreasing = TRUE)

  print(sorted_correlations)
} else {
  print("The 'Churn' column is not in the correlation matrix. Check column names or Data Preprocessing!!!")
}


# (Defining the dependent variable (y) as 'Churn' and independent variables (X) by dropping 'Churn')

y <- SURdf$Churn  

X <- SURdf %>% select(-Churn)


# (Combining X and y into a single dataframe for stratification)

data <- cbind(X, Churn = y)

                                                   # (Random seed for reproducibility)
set.seed(40)

                                      # (Split the data into training (70%) and testing (30%) sets)
trainIndex <- createDataPartition(data$Churn, p = 0.7, list = FALSE, times = 1)

                                            # (I made training and testing datasets)
trainData <- data[trainIndex, ]             # (Training set)
testData <- data[-trainIndex, ]             # (Testing set)

                               # (Separate predictors (X) and target (y) in training and testing sets)
X_train <- trainData %>% select(-Churn)
y_train <- trainData$Churn
X_test <- testData %>% select(-Churn)
y_test <- testData$Churn


#  DISTPLOT FUNCTION

distplot <- function(feature, frame, color = "yellow") {
  ggplot(frame, aes_string(x = feature)) +
    geom_density(fill = color, alpha = 0.5) +
    labs(
      title = paste("Distribution for", feature),
      x = feature,
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10)
    )
}

distplot("MonthlyCharges", SURdf, color = "blue")
distplot("tenure", SURdf, color = "green")
distplot("TotalCharges", SURdf, color = "gold")


#   Since the numerical features are distributed over different value ranges, I will use standard scalar to scale them down to the same range.


# (Standardizing numerical columns)


numerical_cols <- c("MonthlyCharges", "TotalCharges", "tenure")

df_std <- as.data.frame(scale(SURdf[numerical_cols]))  # (Standardize and convert to dataframe)
colnames(df_std) <- numerical_cols

distplot <- function(feature, frame, color = "cyan") {
  ggplot(frame, aes_string(x = feature)) +
    geom_density(fill = color, alpha = 0.5) + 
    labs(
      title = paste("Distribution for", feature),
      x = feature,
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10)
    )
}

for (feat in numerical_cols) {
  print(distplot(feat, df_std, color = "cyan"))
}


#  (STANDARDIZATION, One-Hot ENCODING, LABEL ENCODING)

numerical_cols <- c("MonthlyCharges", "TotalCharges", "tenure") 

                                                                   # (Define the categorical columns for one-hot encoding)
cat_cols_ohe <- c("PaymentMethod", "Contract", "InternetService")  # (Categorical columns for one-hot encoding)

                                                                   # (Those that need label encoding)
cat_cols_le <- setdiff(
  colnames(SURdf),                 
  c(numerical_cols, cat_cols_ohe)                                  # (Exclude numerical and one-hot encoding columns)
)

SURdf[cat_cols_le] <- lapply(SURdf[cat_cols_le], as.factor)

cat_cols_ohe   
cat_cols_le    
numerical_cols


#  (Ensuring the required numerical columns are defined)

numerical_cols <- c("MonthlyCharges", "TotalCharges", "tenure") 

                                                 # Standardize numerical columns in the training set
X_train[numerical_cols] <- as.data.frame(scale(X_train[numerical_cols]))

                                                 # Standardize numerical columns in the test set using training data statistics
train_means <- colMeans(X_train[numerical_cols])
train_sds <- apply(X_train[numerical_cols], 2, sd)
X_test[numerical_cols] <- as.data.frame(
  scale(X_test[numerical_cols], center = train_means, scale = train_sds)
)

                                                 # View the transformed datasets
head(X_train[numerical_cols])              # Standardized training set
head(X_test[numerical_cols])               # Standardized test set




###   STEP :- 8 (Machine Learning Model Evaluations and Predictions)   ###


install.packages("class", dependencies = TRUE)
library(class)


#   KNN(K-Nearest Neighbours) Algorithm   #


                          # Define the number of neighbors
k <- 11

                          # Train the k-NN model and predict on test data
predicted_y <- knn(
  train = X_train,            # Training data (features)
  test = X_test,              # Test data (features)
  cl = y_train,               # Training labels
  k = k                       # Number of neighbors
)

                                             # Calculate accuracy
accuracy_knn <- mean(predicted_y == y_test)  # Compare predictions with actual test labels

print(paste("KNN Accuracy :-", accuracy_knn))

                                            # Convert y_test and predicted_y to factors (if they are not already)
y_test <- as.factor(y_test)
predicted_y <- as.factor(predicted_y)

                                            # Create a confusion matrix
conf_matrix <- confusionMatrix(predicted_y, y_test)

                                            # Print classification report
print(conf_matrix)

 
#   SVM (Support Vector Machine) Algorithm   #


install.packages("e1071")
library(e1071)

# (Train an SVM model)
svc_model <- svm(
  x = X_train,            # Training features
  y = as.factor(y_train), # Training labels (as factor)
  kernel = "linear",      # Kernel type (linear, polynomial, radial, sigmoid)
  probability = TRUE,     # Enable probability predictions
  random.seed = 1         # Set random state equivalent
)

                                           # Predict on test data
predict_y <- predict(svc_model, X_test)

                                           # Calculate accuracy
accuracy_svc <- mean(predict_y == y_test)  # Compare predictions with actual labels

print(paste("SVM Accuracy :- ", accuracy_svc))

# (Convert y_test and predict_y to factors (if not already) )
y_test <- as.factor(y_test)
predict_y <- as.factor(predict_y)

              # Create a confusion matrix
conf_matrix <- confusionMatrix(predict_y, y_test)

              # Print the classification report
print(conf_matrix)


#   RANDOM FOREST Algorithm   #


install.packages("randomForest")
library(randomForest)

# Train a Random Forest model
model_rf <- randomForest(
  x = X_train,                    # Training features
  y = as.factor(y_train),         # Training labels (as factor)
  ntree = 500,                    # Number of trees
  mtry = sqrt(ncol(X_train)),     # Number of features to consider at each split (equivalent to "auto")
  maxnodes = 30,                  # Maximum number of leaf nodes
  strata = as.factor(y_train),    # Stratified sampling by class
  sampsize = c(573, 573),         # Balanced sample size
  importance = TRUE,              # Calculate feature importance
  oob.score = TRUE,               # Use out-of-bag samples for error estimation
  #do.trace = 50                   # Progress output
)

# Print OOB score
print(paste("Out-of-Bag (OOB) Score :- ", model_rf$err.rate[nrow(model_rf$err.rate), 1]))

# Make predictions on the test set
prediction_test <- predict(model_rf, X_test)

accuracy <- mean(prediction_test == y_test)
print(paste("RANDOM FOREST Accuracy :- ", accuracy))


# (Classification Report)

y_test <- as.factor(y_test)
prediction_test <- as.factor(prediction_test)

# Create a confusion matrix
conf_matrix <- confusionMatrix(prediction_test, y_test)

# Print the classification report
print(conf_matrix)


#  (RANDOM FOREST Confusion Matrix)

conf_matrix <- table(Predicted = prediction_test, Actual = y_test)

conf_df <- as.data.frame(as.table(conf_matrix))

ggplot(conf_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "black", linewidth = 0.8) + 
  geom_text(aes(label = Freq), size = 5, color = "white") +  
  scale_fill_gradient(low = "lightblue", high = "blue") +  
  labs(
    title = "RANDOM FOREST CONFUSION MATRIX",
    x = "Predicted",
    y = "Actual"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10)
  )


#   Random Forest ROC Curve   #


install.packages("pROC")
library(pROC)

# Predict probabilities for the test set
y_rfpred_prob <- predict(model_rf, X_test, type = "prob")[, 2]  # Probability for class '1'

# Calculate ROC curve
roc_rf <- roc(y_test, y_rfpred_prob)

# Extract False Positive Rate (FPR) and True Positive Rate (TPR)
fpr_rf <- 1 - roc_rf$specificities
tpr_rf <- roc_rf$sensitivities

roc_df <- data.frame(FPR = fpr_rf, TPR = tpr_rf)

# Plot the ROC curve using ggplot2
ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "red", size = 1) +                                           # ROC curve in red
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +  # Diagonal reference line
  labs(
    title = "Random Forest ROC Curve",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12)
  )



#   LOGISTIC REGRESSION   #

install.packages("caret")
install.packages("glmnet")
install.packages("ROSE")                   # For Over Sampling / Under Sampling

library(caret)
library(glmnet)
library(ROSE)

#  (Data Preprocessing (Scaling))  #

                                  # Scale the data (numerical features only)
X_train_scaled <- as.data.frame(scale(X_train))
X_test_scaled <- as.data.frame(scale(X_test))

                                  # Encode categorical variables (One-Hot Encoding)
X_train_encoded <- model.matrix(~ . - 1, data = X_train)  # Remove intercept
X_test_encoded <- model.matrix(~ . - 1, data = X_test)


#  (Checking Class Imbalance and balancing DATA)  #


                                  # Check class distribution
print(table(y_train))

                                  # Apply oversampling using ROSE
balanced_train <- ovun.sample(
  Churn ~ ., 
  data = cbind(X_train, Churn = y_train), 
  method = "over", 
  N = 2 * table(y_train)[1]
)$data

                                  # Split balanced data back into X_train and y_train
X_train_balanced <- balanced_train[, colnames(X_train)]
y_train_balanced <- balanced_train$Churn


#  (Model Training)  #


# Prepare the data for glmnet
X_train_matrix <- as.matrix(scale(X_train_balanced))  
y_train_binary <- as.numeric(as.factor(y_train_balanced)) - 1 

# Perform cross-validation to find the best lambda
cv_fit <- cv.glmnet(
  X_train_matrix, 
  y_train_binary, 
  alpha = 0,  # L2 regularization
  family = "binomial", 
  nfolds = 10
)

# Extract the best lambda
best_lambda <- cv_fit$lambda.min
print(paste("Best Lambda :- ", best_lambda))


#  (Model Prediction and Evaluation)  #


# Train the final model using the best lambda
final_model <- glmnet(
  X_train_matrix, 
  y_train_binary, 
  alpha = 0, 
  family = "binomial", 
  lambda = best_lambda
)

# Predict probabilities on the test set
X_test_matrix <- as.matrix(scale(X_test))
y_pred_prob <- predict(final_model, newx = X_test_matrix, type = "response")

# Convert probabilities to class predictions
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)

# Calculate accuracy
accuracy <- mean(y_pred == y_test)
print(paste("Logistic Regression (GLM) Accuracy :- ", accuracy))


#  (Classification Report)  #

# Ensure y_pred and y_test are factors
y_pred <- as.factor(y_pred)
y_test <- as.factor(y_test)

# Generate confusion matrix and classification report
conf_matrix <- confusionMatrix(y_pred, y_test)

# Print the classification report
print(conf_matrix)


#  (Logistic Regression - Confusion Matrix)  #


conf_matrix <- table(Predicted = y_pred, Actual = y_test)
                                      # Convert the confusion matrix to a data frame for ggplot
conf_df <- as.data.frame(as.table(conf_matrix))

# Create the heatmap using ggplot2
ggplot(conf_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "black", linewidth = 1) +                
  geom_text(aes(label = Freq), size = 5, color = "white") + 
  scale_fill_gradient(low = "blue", high = "red") +          
  labs(
    title = "Logistic Regression Confusion Matrix",
    x = "Predicted",
    y = "Actual"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )


#  (Logistic Regression - ROC Curve)  #


# Calculate the ROC curve
roc_curve <- roc(y_test, y_pred_prob)  # y_test (true labels) and y_pred_prob (predicted probabilities)

# Extract False Positive Rate (1 - Specificity) and True Positive Rate (Sensitivity)
fpr <- 1 - roc_curve$specificities
tpr <- roc_curve$sensitivities

# Create a data frame for ggplot
roc_df <- data.frame(FPR = fpr, TPR = tpr)

# Plot the ROC curve using ggplot2
ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "red", size = 1) +                          
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +  
  labs(
    title = "Logistic Regression ROC Curve",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )


#  DECISION TREE  #

 
install.packages("rpart")
install.packages("caret")                      # For accuracy and evaluation
library(rpart)
library(caret)

# Train the Decision Tree model
dt_model <- rpart(
  Churn ~ .,                              # Formula: Churn as a function of all features
  data = cbind(X_train, Churn = y_train), # Combine features and labels
  method = "class"                        # Classification method
)

# Predict on the test set
predictdt_y <- predict(dt_model, newdata = X_test, type = "class")

# Calculate accuracy
accuracy_dt <- mean(predictdt_y == y_test)
print(paste("Decision Tree Accuracy :- ", accuracy_dt))


#  (CLASSIFICATION Report)  #


# Ensure predictions and true labels are factors
predictdt_y <- as.factor(predictdt_y)
y_test <- as.factor(y_test)

# Generate the confusion matrix and classification report
conf_matrix <- confusionMatrix(predictdt_y, y_test)

# Print the classification report
print(conf_matrix)



#   XG BOOST CLASSIFIER   #


install.packages("xgboost")
library(xgboost)

# Convert the training and test data to matrices
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

# Convert labels to numeric (required by xgboost)
y_train_numeric <- as.numeric(as.factor(y_train)) - 1  # Convert to 0/1
y_test_numeric <- as.numeric(as.factor(y_test)) - 1    # Convert to 0/1


# Create DMatrix objects for training and testing
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train_numeric)
dtest <- xgb.DMatrix(data = X_test_matrix, label = y_test_numeric)

# Train the model
a_model <- xgboost(
  data = dtrain, 
  max_depth = 1,               # Depth of 1 mimics AdaBoost's decision stumps
  eta = 1,                     # High learning rate to mimic AdaBoost
  nrounds = 50,                # Number of boosting iterations
  objective = "binary:logistic", # Binary classification
  verbose = 1                  # Show training progress
)


# Predict probabilities
a_preds_prob <- predict(a_model, dtest)

# Convert probabilities to binary predictions
a_preds <- ifelse(a_preds_prob > 0.5, 1, 0)

# Calculate accuracy
accuracy <- mean(a_preds == y_test_numeric)
print(paste("AdaBoost-Like Model Accuracy :-", accuracy))


#  (Classification Report)  #


# Ensure predictions and true labels are factors
a_preds <- as.factor(a_preds)
y_test <- as.factor(y_test_numeric)

# Generate confusion matrix and classification report
conf_matrix <- confusionMatrix(a_preds, y_test)

# Print the classification report
print(conf_matrix)


#  (Confusion Matrix)  #


# Generate the confusion matrix
conf_matrix <- table(Predicted = a_preds, Actual = y_test)

# Convert the confusion matrix to a data frame for ggplot
conf_df <- as.data.frame(as.table(conf_matrix))

# Create the heatmap using ggplot2
ggplot(conf_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "black", linewidth = 1) +                # Add gridlines to heatmap
  geom_text(aes(label = Freq), size = 5, color = "white") +  # Annotate the cells
  scale_fill_gradient(low = "blue", high = "red") +          # Gradient color for cells
  labs(
    title = "XGBoost Confusion Matrix",
    x = "Predicted",
    y = "Actual"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )


##########            END OF CODE          ##########