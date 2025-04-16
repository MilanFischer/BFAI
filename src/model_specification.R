#-------------------------------------------------------------------------------
# Linear Regression Model with LASSO/Ridge (Elastic Net)
# Used for linear regression with regularization controlled by penalty and mixture
linear_reg_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet")

# Neural Network Model
# A multilayer perceptron (MLP) model with tuning for hidden units and epochs
nnet_spec <- 
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |> 
  set_engine("nnet", MaxNWts = 2600) |>  
  set_mode("regression")

# Multivariate Adaptive Regression Splines (MARS)
# Captures nonlinear interactions between variables
mars_spec <- 
  mars(prod_degree = tune()) |> #<- use GCV to choose terms
  set_engine("earth") |>  
  set_mode("regression")

# Support Vector Machine with Radial Basis Function Kernel
# Suitable for capturing complex, nonlinear relationships
svm_r_spec <- 
  svm_rbf(cost = tune(), rbf_sigma = tune()) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

# Support Vector Machine with Polynomial Kernel
# Another nonlinear model using polynomial basis functions
svm_p_spec <- 
  svm_poly(cost = tune(), degree = tune()) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

# k-Nearest Neighbors (KNN) Regression
# Distance-based model with tunable neighbors and weighting
knn_spec <- 
  nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) |> 
  set_engine("kknn") |> 
  set_mode("regression")

# Decision Tree Regression
# A simple decision tree model for regression with tunable complexity
cart_spec <- 
  decision_tree(cost_complexity = tune(), min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

# Bagged Decision Trees
# Ensemble method using multiple decision trees
bag_cart_spec <- 
  bag_tree() |> 
  set_engine("rpart", times = 50L) |> 
  set_mode("regression")

# Random Forest Regression
# Ensemble method using multiple decision trees with randomness
rf_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |> 
  set_engine("ranger", importance = "impurity") |> 
  set_mode("regression")

# Extreme Gradient Boosting (XGBoost)
# A powerful boosting model with extensive tuning options
xgb_spec <- 
  boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
             min_n = tune(), sample_size = tune(), trees = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

# Cubist Rule-Based Regression
# A rule-based model with committees and neighbors for prediction
cubist_spec <- 
  cubist_rules(committees = tune(), neighbors = tune()) |> 
  set_engine("Cubist") 

nnet_param <- 
  nnet_spec |> 
  extract_parameter_set_dials() |> 
  update(hidden_units = hidden_units(c(1, 27)))

# Support Vector Regressor with Linear Kernel
svm_linear_spec <- svm_linear(cost = tune()) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

# Bayesian Additive Regression Trees (BART)
bart_spec <- bart(mode = "regression") |> 
  set_engine("dbarts")

# Traditional GLM
glm_spec <- linear_reg() |> 
  set_engine("glm") |> 
  set_mode("regression")

