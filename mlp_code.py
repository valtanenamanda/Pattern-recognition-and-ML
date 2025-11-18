import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mlp(traindata, trainclass, testdata, maxEpochs=100000):
    N = traindata.shape[1]
    d = traindata.shape[0]
    classes = np.max(trainclass)

    # Initialisation
    hidden=10
    J = []  # List to store loss function values
    rho=0.1
    
    # Initialize the training output matrix
    trainOutput = np.zeros((classes, N))
    for i in range(N):
        trainOutput[trainclass[i] - 1, i] = 1  # Adjust for 0-based indexing

    # Add bias term to the input data
    extendedInput = np.vstack((traindata, np.ones(N)))

    # Initialize weight matrices with small random values
    wHidden = (np.random.rand(d + 1, hidden) - 0.5) / 10
    wOutput = (np.random.rand(hidden + 1, classes) - 0.5) / 10

    fig, ax = plt.subplots()
    t = 0

    while t < maxEpochs:
        t += 1

        # Feed-forward operation
        vHidden = np.dot(wHidden.T, extendedInput) # hidden layer net activation
        yHidden = 1 / (1 + np.exp(-vHidden)) # hidden layer activation function


        extendedHidden = np.vstack((yHidden, np.ones(N))) # hidden layer extended output
        vOutput = np.dot(wOutput.T, extendedHidden) # output layer net activation
        yOutput = vOutput  # Linear output

        # Calculate the loss function value (You need to define the loss function)
        loss = 0.5 * np.sum((yOutput - trainOutput) ** 2) / N  # Fill in the loss function calculation
        J.append(loss)

        if t % 1000 == 0:
            fig, ax = plt.subplots()
            ax.semilogy(range(1, len(J) + 1), J)
            ax.set_title(f"Training (epoch {t})")
            ax.set_ylabel("Training error")
            ax.set_xlabel("Epoch")
            fig.tight_layout()
            fig.savefig("training_error2.png")

            


        # You need to define the stopping conditions here
        if loss < 1e-3:  # Check if the learning is good enough
            break

        if t > maxEpochs:  # Check if too many epochs would be done
            break

        if t > 1:
            if abs(J[t-1] - J[t-2]) < 1e-10:  # Check if the improvement is small enough
                break


        # Update sensitivities and weights (You need to define the backpropagation equations)
        deltaOutput = yOutput - trainOutput
        deltaHidden = np.dot(wOutput[:-1, :], deltaOutput) * yHidden * (1 - yHidden)
        deltawHidden = -rho * np.dot(extendedInput, deltaHidden.T) / N
        deltawOutput = -rho * np.dot(extendedHidden, deltaOutput.T) / N

        wOutput = wOutput + deltawOutput
        wHidden = wHidden + deltawHidden

    # mean training loss over all epochs
    mean_loss = float(np.mean(J))    

    # Testing with the test data
    N = testdata.shape[1]
    extendedInput = np.vstack((testdata, np.ones(N)))

    vHidden = np.dot(wHidden.T, extendedInput) # hidden layer net activation
    yHidden = 1 / (1 + np.exp(-vHidden)) # hidden layer activation function

    extendedHidden = np.vstack((yHidden, np.ones(N))) # hidden layer extended output
    vOutput =  np.dot(wOutput.T, extendedHidden) # hidden layer net activation
    yOutput = vOutput
    testclass = np.argmax(yOutput, axis=0) + 1  # Adjust for 0-based indexing

    return testclass, t, wHidden, wOutput, mean_loss


# Usage

# Load data
df1 = pd.read_csv('data1.csv', header=None) 
df2 = pd.read_csv('data2.csv', header=None)
df3 = pd.read_csv('data3.csv', header=None)
trainclass = df1.iloc[:, -1].values
traindata = df1.iloc[:, :-1].values.T
testdata = df3.iloc[:, :-1].values.T
trueclass = df3.iloc[:, -1].values

maxEpochs = 100000

# Plot the first two columns of all three datasets
plt.figure(figsize=(6, 5))

plt.scatter(df1.iloc[:, 0], df1.iloc[:, 1], s=10, label="data1", alpha=0.7)
plt.scatter(df2.iloc[:, 0], df2.iloc[:, 1], s=10, label="data2", alpha=0.7)
plt.scatter(df3.iloc[:, 0], df3.iloc[:, 1], s=10, label="data3", alpha=0.7)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Datasets Scatter Plot")
plt.legend()
plt.tight_layout()
plt.savefig("datasets_scatter.png")
plt.show()


testclass, t, wHidden, wOutput, mean_loss = mlp(traindata, trainclass, testdata, maxEpochs)
print("Training completed in {} epochs".format(t))
print("Mean training loss:", mean_loss)
accuracy = np.mean(testclass == trueclass)
print("Accuracy:", accuracy)

# I tried every set as a training set and a test set:

# When training with data1 and testing with data2:
# Average training loss: 0.0338
# Accuracy: 0.5

# When training with data1 and testing with data3:
# Average training loss: 0.02815
# Accuracy: 0.976

# When training with data2 and testing with data1:
# Average training loss: 0.268
# Accuracy: 0.5

# When training with data2 and testing with data3:
# Average training loss: 0.250
# Accuracy: 0.5

# When training with data3 and testing with data1:
# Average training loss: 0.0259
# Accuracy: 0.833

# When training with data3 and testing with data2:
# Average training loss: 0.0262
# Accuracy: 0.5

# The results indicate that the model performs well when trained on data1 or data3, but struggles with data2.
# This could be explained when looking at the scatter plot of the datasets, where data2 appears to be located in a completely different region of the feature space compared to data1 and data3.
# If the model is intended to perform well on data2, the training set must include samples similar to data2.

# The model learns datasets 1 and 3 well and achieves high accuracy when tested on these (n.o. 1 or 3) sets.
# The MLP works correctly: it learns well when training and test data belong to the same distribution, but it cannot generalize to completely different data (data2). 
# Improving performance requires either better training data coverage or increasing model capacity, data standardization, balancing data distribution or applying preprocessing techniques.