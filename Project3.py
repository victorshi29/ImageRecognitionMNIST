import numpy as np
import matplotlib.pyplot as plt

train_images_np = np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np = np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np = np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np = np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np = np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np = np.load('./Project3_Data/MNIST_test_labels.npy')


##Template MLP code
# modified to be numerically stable because of overflow errors
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    # return np.exp(x)/np.sum(np.exp(x))


# modified sigmoid function to be numerically stable because of overflow errors
def sigmoid(x):
    a = x
    for i in range(0, len(a)):
        if a[i] < 0:
            a[i] = np.exp(a[i]) / (1 + np.exp(a[i]))
        else:
            a[i] = 1 / (1 + np.exp(-a[i]))
    return a


def CrossEntropy(y_hat, y):
    return -np.dot(y, np.log(y_hat))


class MLP():

    def __init__(self):
        # Initialize all the parametres
        # Uncomment and complete the following lines
        self.W1 = np.random.normal(loc=0, scale=0.1, size=(64, 784))
        self.b1 = np.zeros(64)
        self.W2 = np.random.normal(loc=0, scale=0.1, size=(10, 64))
        self.b2 = np.zeros(10)
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        # Feed data through the network
        # Uncomment and complete the following lines
        self.x = x
        self.W1x = np.matmul(self.W1, self.x)
        self.a1 = self.W1x + self.b1
        self.f1 = sigmoid(self.a1)
        self.W2x = np.matmul(self.W2, self.f1)
        self.a2 = self.W2x + self.b2
        self.y_hat = softmax(self.a2)
        return self.y_hat

    def update_grad(self, y):
        # Compute the gradients for the current observation y and add it to the gradient estimate over the entire batch
        # Uncomment and complete the following lines
        dA2db2 = np.identity(10)

        dA2dW2 = np.zeros((10, 640))
        for row in range(0, 10):
            for col in range(row * 64, row * 64 + 64):
                dA2dW2[row, col] = self.f1[col - row * 64]

        dA2dF1 = self.W2
        dF1dA1 = np.diag(self.f1 * (1 - self.f1))
        dA1db1 = np.identity(64)
        dF1db1 = dF1dA1  # np.matmul(dF1dA1, dA1db1) # does nothing

        dA1dW1 = np.zeros((64, 64 * 784))
        for row in range(0, 64):
            for col in range(row * 784, row * 784 + 784):
                dA1dW1[row, col] = self.x[col - row * 784]

        dLdA2 = self.y_hat - y
        dLdW2 = np.matmul(dLdA2, dA2dW2)
        dLdb2 = dLdA2  # np.matmul(dLdA2, dA2db2), dA2db2 is identity, so does nothing
        dLdF1 = np.matmul(dLdA2, dA2dF1)
        dLdA1 = np.matmul(dLdF1, dF1dA1)
        dLdW1 = np.matmul(dLdA1, dA1dW1)
        dLdb1 = np.matmul(dLdF1, dF1db1)

        self.W2_grad = self.W2_grad + dLdW2
        self.b2_grad = self.b2_grad + dLdb2
        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + dLdb1

    def update_params(self, learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad.reshape((10, 64))
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad.reshape((64, 784))
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)


## Init the MLP
myNet = MLP()

learning_rate = 1e-3
n_epochs = 0

rng = np.random.default_rng(0)
accuracies = []
epochs = []
val_accuracies = []

labels = train_labels_np
inputs = train_images_np

val_labels = val_labels_np
val_inputs = val_images_np

## Training code
for iter in range(n_epochs):
    # Code to train network goes here
    indexes = np.arange(0, len(labels), 1)
    val_indexes = np.arange(0, 5000, 1)
    rng.shuffle(indexes)
    rng.shuffle(val_indexes)

    if iter != 0 and iter % 5 == 0:
        learning_rate = learning_rate / 2
    max_iter = int(np.ceil(len(labels) / 256))
    total_accuracies = 0
    val_total_accuracies = 0

    for n in range(0, max_iter):
        loss_sum = 0
        correct = 0
        val_correct = 0

        if n != max_iter - 1:
            start = n * 256
            end = n * 256 + 256
        else:
            start = (max_iter - 1) * 256
            end = len(labels)

        myNet.reset_grad()
        for i in range(start, end):
            x = train_images_np[indexes[i]]
            b = np.random.randint(0,5000)
            val_x = val_images_np[b]

            val_y_hat = myNet.forward(val_x)
            val_classification = val_labels_np[b]
            val_y = np.zeros(10)
            val_y[val_classification] = 1

            y_hat = myNet.forward(x)
            classification = train_labels_np[indexes[i]]
            y = np.zeros(10)
            y[classification] = 1

            myNet.update_grad(y)

            loss = CrossEntropy(y_hat, y)
            loss_sum = loss_sum + loss
            y_hat_index = np.where(y_hat == np.amax(y_hat))
            if y_hat_index[0] == classification:
                correct = correct + 1

            val_y_hat_index = np.where(val_y_hat == np.amax(val_y_hat))
            if val_y_hat_index[0] == val_classification:
                val_correct = val_correct + 1

        myNet.update_params(learning_rate)

        print("Average loss : " + str(loss_sum / (end - start)))
        print("Epoch: " + str(iter) + " Accuracy of batch " + str(n) + ": " + str(correct / (end - start) * 100))
        total_accuracies = total_accuracies + (correct / (end - start) * 100)
        val_total_accuracies = val_total_accuracies + (val_correct / (end - start) * 100)
    accuracies.append(total_accuracies / max_iter)
    val_accuracies.append(val_total_accuracies / max_iter)
    epochs.append(iter)

plt.plot(epochs, accuracies)
plt.plot(epochs, val_accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("MLP Using the 50000 Images")
plt.show()

#############################################

# TESTING PART

myNet = MLP()
w1 = np.load("w1.npy")
w2 = np.load("w2.npy")
b1 = np.load("b1.npy")
b2 = np.load("b2.npy")

myNet.W2 = w2
myNet.W1 = w1
myNet.b1 = b1
myNet.b2 = b2

labels = test_labels_np
inputs = test_images_np

val_labels = val_labels_np
val_inputs = val_images_np

## Training code

confusion_matrix = np.zeros((10,10))
correct = 0
max_iter = int(np.ceil(len(labels) / 256))
for n in range(0, 5000):
    x = train_images_np[n]
    classification = train_labels_np[n]
    y = np.zeros(10)
    y[classification] = 1
    y_hat = myNet.forward(x)
    y_hat_index = np.where(y_hat == np.amax(y_hat))
    confusion_matrix[int(y_hat_index[0])][classification] = confusion_matrix[int(y_hat_index[0])][classification] + 1
    if y_hat_index[0] == classification:
        correct = correct + 1

print(correct / 5000 * 100)

##############################################################################################
##############################################################################################

# PYTORCH PART

## Template for ConvNet Code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

train_images_np = np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np = np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np = np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np = np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np = np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np = np.load('./Project3_Data/MNIST_test_labels.npy')


class ConvNet(nn.Module):
    # From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Your training and testing code goes here

net = ConvNet()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters())

#labels = torch.tensor(train_labels_np[0:2000]).long()
#inputs = torch.tensor(train_images_np[0:2000]).float()

labels = torch.tensor(train_labels_np).long()
inputs = torch.tensor(train_images_np).float()

val_labels = torch.tensor(val_labels_np).long()
val_inputs = torch.tensor(val_images_np).float()

m = nn.Softmax(dim=1)
train_acc_arr = []
val_acc_arr = []

for epoch in range(0, 100):  # loop over the dataset multiple times

    indexes = np.arange(0, len(labels))
    val_indexes = np.arange(0, len(val_labels))
    np.random.shuffle(indexes)
    np.random.shuffle(val_indexes)

    running_loss = 0.0
    total_train_acc = 0
    total_valid_acc = 0

    max_iter = int(np.ceil(len(labels) / 256))
    for i in range(0, max_iter):

        if i != max_iter - 1:
            start = i * 256
            end = i * 256 + 256
        else:
            start = (max_iter - 1) * 256
            end = len(labels)

        labels_batch = labels[indexes[start:end]]
        inputs_batch = inputs[indexes[start:end]]
        val_labels_batch = val_labels
        val_inputs_batch = val_inputs

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs_batch)
        val_outputs = net(val_inputs_batch)

        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        correct = 0
        labels_one_hot = F.one_hot(torch.tensor(labels_batch).long())
        outputs_one_hot = F.one_hot(torch.argmax(m(outputs), dim=1), 10)
        temp = (labels_one_hot == outputs_one_hot).float().sum(dim=1)
        for j in range(0, len(temp)):
            if temp[j] == 10:
                correct = correct + 1

        val_correct = 0
        val_labels_one_hot = F.one_hot(torch.tensor(val_labels_batch).long())
        val_outputs_one_hot = F.one_hot(torch.argmax(m(val_outputs), dim=1), 10)
        temp2 = (val_labels_one_hot == val_outputs_one_hot).float().sum(dim=1)
        for k in range(0, len(temp2)):
            if temp2[k] == 10:
                val_correct = val_correct + 1

        # print statistics
        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 256:.3f}')
        running_loss = 0.0
        accuracy = 100 * correct / (end - start)
        val_accuracy = 100 * val_correct / (5000)
        print("Accuracy = {}".format(accuracy))
        print("Validation Accuracy = {}".format(val_accuracy))
        total_train_acc = total_train_acc + accuracy
        total_valid_acc = total_valid_acc + val_accuracy

    val_acc_arr.append(total_valid_acc / max_iter)
    train_acc_arr.append((total_train_acc / max_iter))
print('Finished Training')

epochs = np.arange(1, 101, 10)
plt.plot(epochs, train_acc_arr, color="blue", label="Training")
plt.plot(epochs, val_acc_arr, color="red", label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("CNN Using 50000 Images")
plt.legend()
plt.show()

##############################################################

# TESTING CNN PART

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

train_images_np = np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np = np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np = np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np = np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np = np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np = np.load('./Project3_Data/MNIST_test_labels.npy')

labels = torch.tensor(test_labels_np).long()
inputs = torch.tensor(test_images_np).float()
class ConvNet(nn.Module):
    # From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Your training and testing code goes here
m = nn.Softmax(dim=1)
net = ConvNet()
net = torch.load("cnn.pt")

correct = 0

labels_batch = labels
inputs_batch = inputs

outputs = net(inputs_batch)

labels_one_hot = F.one_hot(torch.tensor(labels_batch).long())
outputs_one_hot = F.one_hot(torch.argmax(m(outputs), dim = 1), 10)
temp = (labels_one_hot == outputs_one_hot).float().sum(dim = 1)
for j in temp:
  if j == 10:
    correct = correct + 1

# print statistics

print(100 * correct / len(labels))

print('Finished')

