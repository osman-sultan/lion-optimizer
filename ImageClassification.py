import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from lion_pytorch import Lion
import kagglehub
import torchvision
from matplotlib import pyplot as plt
import time
import os

#Model
def train(model, optimizer, train_loader, val_loader, num_epochs, name="Model", typeOptimizer = "Adam", dataset="MNIST", folder_name=""):
    criterion = nn.CrossEntropyLoss()
    start = time.time()

    train_accuracy, val_accuracy = [], []
    total_batches = len(train_loader)
    #Train each epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress = (batch_idx + 1) / total_batches * 100
            print(f"\rEpoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{total_batches} - {progress:.2f}% completed", end="")

        train_acc = train_accuracy.append(get_accuracy(model, train_loader))
        val_acc = val_accuracy.append(get_accuracy(model, val_loader))
        print(f"{name} - Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {val_accuracy[-1]}")
    timeTaken = time.time() - start
    print(f"Training Time: {timeTaken}")

    #Print accuracies to a graph
    plt.title("Train Data Accuracy")
    plt.plot(range(num_epochs), train_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt_path=os.path.join(folder_name, f'{name}_{typeOptimizer}_{dataset}_Training.pdf')
    plt.savefig(plt_path, format="pdf", bbox_inches='tight')

    plt.title("Validation Data Accuracy")
    plt.plot(range(num_epochs), val_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt_path=os.path.join(folder_name, f'{name}_{typeOptimizer}_{dataset}_Validation.pdf')
    plt.savefig(plt_path, format="pdf", bbox_inches='tight')
    return timeTaken

# Define validation accuracy function
def get_accuracy(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def TrainAndOptimize(storage, model, hyperparameters, train_load, val_load, name="Model", dataset="MNist"):
  #Save data to a respective folder
  folder_name = f"{name}_batchSize_{hyperparameters["BatchSize"]}_learningRate_{hyperparameters["LearningRate"]}_weightDecay_{hyperparameters["WeightDecay"]}_NumEpochs_{hyperparameters["NumEpochs"]}"
  os.makedirs(folder_name, exist_ok=True)

  # Train with Lion
  optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparameters["LearningRate"], weight_decay=hyperparameters["WeightDecay"])
  timeLion = train(model=model, optimizer=optimizer, train_loader=train_load, val_loader=val_load, num_epochs=hyperparameters["NumEpochs"], name=name + " Lion", typeOptimizer="Lion", dataset=dataset, folder_name=folder_name)
  val_accL = get_accuracy(model, val_load)
  print(f"{name} Validation Set Accuracy - Lion: {val_accL}")
  model_path=os.path.join(folder_name, f'{name}_Lion_model_weights_{dataset}.pth')
  torch.save(model.state_dict(), model_path)

  # Train with Adam
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparameters["LearningRate"], weight_decay=hyperparameters["WeightDecay"])
  timeAdam = train(model=model, optimizer=optimizer, train_loader=train_load, val_loader=val_load, num_epochs=hyperparameters["NumEpochs"], name=name + " Adam", typeOptimizer="Adam", dataset=dataset, folder_name=folder_name)
  val_accA = get_accuracy(model, val_load)
  print(f"{name} Validation Set Accuracy - Adam: {val_accA}")

  model_path=os.path.join(folder_name, f'{name}_Adam_model_weights_{dataset}.pth')
  torch.save(model.state_dict(), model_path)

  #Write accurcies to a text file
  model_path = os.path.join(folder_name, f'{name}_{dataset}_Accuracy.txt')
  with open(model_path, 'w') as file:
    file.write(f"{name} Validation Set Accuracy - Lion: {val_accL} - Time Taken: {timeLion}\n")
    file.write(f"{name} Validation Set Accuracy - Adam: {val_accA} - Time Taken: {timeAdam}")
  
  final_name = f"{name}_{dataset}_batchSize_{hyperparameters["BatchSize"]}_learningRate_{hyperparameters["LearningRate"]}_weightDecay_{hyperparameters["WeightDecay"]}_NumEpochs_{hyperparameters["NumEpochs"]}"
  storage.append(f"{final_name} Validation Set Accuracy - Lion: {val_accL} - Time Taken: {timeLion}")
  storage.append(f"{final_name} Validation Set Accuracy - Adam: {val_accA} - Time Taken: {timeAdam}")

  

if __name__ == "__main__":
    #Cuda
    if torch.cuda.is_available():
        print("Cuda Avaliable - Using GPU")
    else:
        print("Using CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Download card dataset
    data_path_cards = kagglehub.dataset_download('gpiosenka/cards-image-datasetclassification')
    
    #Data Transformation
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])

    batch_size = [16, 32, 64, 128]
    learning_rate = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    weight_decay = [0.001, 0.01, 0.1, 1]
    num_epochs = [5, 10]

    storage = []
    # Uncomment if training weight decay and num epochs
    # for wd in weight_decay:
    #     for ne in num_epochs:
    for bs in batch_size:
        for lr in learning_rate:
            #Set hyperparameters
            hyperparameters = {"BatchSize": bs, "NumWorkers":4, "LearningRate": lr, "NumEpochs": 5, "WeightDecay": 1e-2}
            # Uncomment if training weight decay and num epochs
            #hyperparameters = {"BatchSize": 16, "NumWorkers":4, "LearningRate": 0.01, "NumEpochs": ne, "WeightDecay": wd}

            #Load MNIST data
            train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            val_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            train_load_mnist = DataLoader(train_mnist, batch_size=hyperparameters["BatchSize"], num_workers=hyperparameters["NumWorkers"], shuffle=True)
            val_load_mnist = DataLoader(val_mnist, batch_size=hyperparameters["BatchSize"], num_workers=hyperparameters["NumWorkers"], shuffle=True)
            
            #Load Card Dataset
            train_path_cards = os.path.join(data_path_cards, "train")
            test_path_cards = os.path.join(data_path_cards, "test")
            train_data_cards = datasets.ImageFolder(root=train_path_cards, transform=transform)
            val_data_cards = datasets.ImageFolder(root=test_path_cards, transform=transform)
            train_load_cards = DataLoader(train_data_cards, batch_size=hyperparameters["BatchSize"], num_workers=hyperparameters["NumWorkers"], shuffle=True)
            val_load_cards = DataLoader(val_data_cards, batch_size=hyperparameters["BatchSize"], num_workers=hyperparameters["NumWorkers"], shuffle=True)

            #Models
            resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(device)
            resnet34 = torchvision.models.resnet34(weights='IMAGENET1K_V1').to(device)
            resnet50 = torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device)
            squeezeNet = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True).to(device)
            alexNet = torchvision.models.alexnet(weights='IMAGENET1K_V1').to(device)
            effNet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True).to(device)

            #Train Models
            models = {"ResNet18":resnet18, "ResNet34":resnet34, "ResNet50":resnet50, "SqueezeNet":squeezeNet, "AlexNet":alexNet, "EfficientNet":effNet}
            for key in models:
                model = models[key]

                #This is responsible for freezing 90% of the weights
                num_frozen = int(len(list(model.parameters())) * 0.9)
                for weight in list(model.parameters())[:num_frozen]:
                    weight.requires_grad = False
                
                TrainAndOptimize(storage, model, hyperparameters, train_load_mnist, val_load_mnist, key, "MNIST")
                TrainAndOptimize(storage, model, hyperparameters, train_load_cards, val_load_cards, key, "Cards")
            storage.append("#######################################################")
    
    with open("AllAccuracies.txt", 'w') as file:
        for line in storage:
            file.write(line + "\n")