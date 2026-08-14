# WEEK 6 module - capstone development 

#Importing all the libraries
#---------------------------
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg") #Prevent matpotlib to open new window

import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#Giving folder paths
#-------------------
datafolder = "DATA"
modelfolder = "MODELS"
outputfolder = "OUTPUTS"

#LOADING DATASET
#---------------
print("\nLoading dataset....")

#EXTRACTING ZIPFILE
#------------------
file = "DATA/House price india.zip"  #file name

with zipfile.ZipFile(file) as zr:
    zr.extractall(datafolder)

# Finding CSV file (we can simply use name but this is more scalable)
#-------------------------------------------------------------------
csvfile = None

for file in os.listdir(datafolder):   #creating loop to find file

    if file.lower().endswith(".csv"):
        csvfile = os.path.join(datafolder, file)
        break

if csvfile is None:                   #if there is no file break
    print("CSV file not found.")
    exit()

df = pd.read_csv(csvfile)

print("\nDataset loaded successfully.")
print("-----------------------------")

#FINDING SHAPE OF DATASET
#------------------------
print("Rows and columns:", df.shape)

# BASIC INFORMATION IN DATASET
#-----------------------------
print("\nFirst 5 rows:")
print("---------------")

print(df.head())
print("-----------------------------------------------------------------")

print("\nColumn names:")
print("--------------")

print(df.columns.tolist())
print("------------------------------------------------------------------")

print("\nMissing values:")
print(df.isnull().sum().sum())
print("---------------------")


#Removing ID as it does not describe the house
#---------------------------------------------
df.drop("id", axis=1, inplace=True)

print("ID column removed because it does't provide much useful info.")
print("--------------------------")


#Handling MISSING VALUES
#-----------------------
df.fillna(df.median(numeric_only=True),inplace=True)
print("Missing values handled.")
print("-----------------------")

# SEPARATING FEATURES AND TARGET
#-------------------------------
X = df.drop("Price", axis=1)
y = df["Price"]

# Convert text columns into numbers
#----------------------------------
X = pd.get_dummies( X , drop_first=True )  

print("Number of features:", X.shape[1])

# SPLITING DATA INTO TRAINING AND TESTING SET

# SPLIT INTO 3 SETS(testing, training and validation)
#----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=65)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.20,random_state=65)


print("\nTraining rows:", X_train.shape[0])
print("Validation rows:", X_val.shape[0])  
print("Testing rows :", X_test.shape[0])

#scaling 
#----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)      
X_test = scaler.transform(X_test)



# Baseline random forest
#-----------------------
print("\n----------------------------")
print("RANDOM FOREST BASELINE")
print("----------------------------")

randomforest = RandomForestRegressor(n_estimators=200, random_state=65 ,n_jobs=-1)

randomforest.fit(X_train,y_train)

baselineprediction = randomforest.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baselineprediction)

baseline_rmse = np.sqrt(mean_squared_error(y_test, baselineprediction))

baseline_r2 = r2_score(y_test,baselineprediction)

print("Mean Absolute Error :", round(baseline_mae, 4))
print("Root Mean Squared Error:", round(baseline_rmse, 4))
print("R2 score :", round(baseline_r2, 4))


#Settings of genetic algorithm
#-----------------------------
populationsize = 30
generations = 20
mutationrate = 0.17
numoffeatures = X_train.shape[1]

print("\n-----------------------------")
print("GENETIC ALGORITHM SETTINGS")
print("-----------------------------")

print("Population size:", populationsize)
print("Generations:", generations)
print("Mutation rate:", mutationrate)
print("Number of features:", numoffeatures ,"\n")
print("------------------------------------------")
print("We are now going generation by generation: ")
print("------------------------------------------\n")


#Taking random initial population
#-------------------------------- 
population = np.random.randint(0,2,size=(populationsize, numoffeatures))


#fitness function
#----------------
def fitness(chromosome):
    selected = np.where(chromosome == 1)[0]
    
    if len(selected) == 0:
        return 0

    traindata = X_train[:, selected]
    valdata = X_val[:, selected]      
    
    model = RandomForestRegressor(n_estimators=40, random_state=65, n_jobs=-1)
    model.fit(traindata, y_train)
    prediction = model.predict(valdata)
    
    score = r2_score(y_val, prediction)  
    return score


#Selection
#---------

def selection(population, scores):

    sortedindices = np.argsort(scores)[::-1]

    # Keep the best 10 chromosomes
    bestindices = sortedindices[:10]

    return population[bestindices]

# CROSSOVER
#----------

def crossover(parent1, parent2):

    point = np.random.randint(1 , numoffeatures)

    child1 = np.concatenate((parent1[:point],parent2[point:]))

    child2 = np.concatenate((parent2[:point],parent1[point:]))

    return child1, child2

# MUTATION
#---------
def mutation(chromosome):

    for i in range(numoffeatures):

        if np.random.random() < mutationrate:

            # 0 becomes 1 and 1 becomes 0
            chromosome[i] = 1 - chromosome[i]

    return chromosome



# Running genetic algorithm
#--------------------------
best_history = []
best_score = -999
bestchromosome = None

for generation in range(generations):

    scores = []

    # Calculating fitness
    for chromosome in population:

        score = fitness(chromosome)

        scores.append(score)

    scores = np.array(scores)

    # Find best chromosome
    best_index = np.argmax(scores)

    generation_best = scores[best_index]

    # Save the best chromosome
    if generation_best > best_score:

        best_score = generation_best

        bestchromosome = population[best_index].copy()

    best_history.append(generation_best)

    print("Generation",generation + 1,"| Best R2:", round(generation_best, 4), "| Average R2:",round(np.mean(scores), 4))

    # Select best parents
    parents = selection(population,scores)

    # Create new population
    newpopulation = []

    newpopulation.append(population[best_index].copy())

    while len(newpopulation) < populationsize:

        i, j = np.random.choice(len(parents),2,replace=False)

        parent1 = parents[i]
        parent2 = parents[j]

        # Crossover
        child1, child2 = crossover(parent1,parent2)

        # Mutation
        child1 = mutation(child1)
        child2 = mutation(child2)

        newpopulation.append(child1)

        if len(newpopulation) < populationsize:

            newpopulation.append(child2)

    population = np.array(newpopulation)

#Finding best features
#---------------------
selected_indices = np.where(bestchromosome == 1)[0]

selectedfeatures = X.columns[selected_indices].tolist()

print("\n-----------------------------")
print("GENETIC ALGORITHM RESULTS")
print("-------------------------------")

print("Best GA R2 SCORE:",round(best_score, 4))

print("Number of selected features:",len(selectedfeatures))

print("\nSelected features:\n")

for feature in selectedfeatures:
    print("*", feature)

# Final random forest
#--------------------
X_train_selected = X_train[:,selected_indices]

X_test_selected = X_test[:,selected_indices]

finalmodel = RandomForestRegressor(n_estimators=100,random_state=65,n_jobs=-1)

finalmodel.fit(X_train_selected,y_train)

final_prediction = finalmodel.predict(X_test_selected)


# Final results
#--------------

final_mae = mean_absolute_error(y_test,final_prediction)

final_rmse = np.sqrt(mean_squared_error(y_test,final_prediction))

final_r2 = r2_score(y_test,final_prediction)

print("\n--------------------------------")
print("FINAL RESULTS")
print("----------------------------------")

print("\nBaseline Random Forest")
print("-----------------------")
print("Mean Absolute Error:", round(baseline_mae, 4))
print("Root Mean Squared Error:", round(baseline_rmse, 4))
print("R2 Score:", round(baseline_r2, 4))

print("\nGA Optimized Random Forest")
print("---------------------------")
print("Mean Absolute Error:", round(final_mae, 2))
print("Root Mean Squared Error:", round(final_rmse, 2))
print("R2 Score :", round(final_r2, 4))

# GA GRAPHS SAVING AS PNG
#------------------------
plt.figure(figsize=(8, 5))

plt.plot(range(1, generations + 1),best_history,marker="o")

plt.axhline(baseline_r2,linestyle="--",label="Baseline R2")

plt.xlabel("Generation")
plt.ylabel("Best R2")
plt.title("Genetic Algorithm Progress")

plt.legend()

plt.savefig("OUTPUTS/ga_progress.png")

plt.close()

# R2 comparison graph between baseline and ga optimized model
#-----------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.bar(["Baseline RF", "GA Optimized RF"],[baseline_r2, final_r2])

plt.ylabel("R2 Score")
plt.title("Random Forest R2 Comparison")

plt.savefig("OUTPUTS/r2_comparison.png")

plt.close()

# Price Distribution Graph
#-------------------------
plt.figure(figsize=(8, 5))

sns.histplot(df["Price"],kde=True)

plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")

plt.savefig("OUTPUTS/price_distribution.png")

plt.close()

# CORRELATION MATRIX
#-------------------

numeric_data = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 10))

sns.heatmap(numeric_data.corr(),cmap="coolwarm")

plt.title("Correlation Matrix")

plt.savefig("OUTPUTS/correlation_matrix.png")

plt.close()

# Saving model
#-------------
model_data = {
    "model": finalmodel,
    "scaler": scaler,
    "selected_features": selected_indices.tolist(),
    "selected_feature_names": selectedfeatures,
    "all_feature_names": X.columns.tolist()
}

joblib.dump(model_data,"MODELS/house_price_ga_random_forest.joblib")

# Results
#--------
print("\n-----------------------------")
print("PROJECT COMPLETED")
print("-----------------------------")

print("Baseline R2:",round(baseline_r2, 4))

print("Optimized R2:",round(final_r2, 4))
print("----------------------------------")

print("Selected features:",len(selectedfeatures),"\n")

print("Model saved in MODELS folder Successfully.")
print("-------------------------------------------")
print("Graphs saved in OUTPUTS folder Successfully.")
print("-------------------------------------------")