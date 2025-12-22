import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

def _to_tensor(arr, device):
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    return arr.to(device=device, dtype=torch.float32)

def _reshape_mask(t, H, W):
    """
    Ensures t is shape [1,1,H,W] (mask) or [1,C,H,W] (delta)
    and resizes to (H,W) with nearest‑neighbour.
    """
    if t.dim() == 2:          # H×W
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:        # C×H×W
        t = t.unsqueeze(0)
    return F.interpolate(t, size=(H, W), mode="nearest")

def add_trigger_to_batch(data, m, delta, trigger_size=None):
    """
    data  : torch tensor [B,C,H,W]
    m     : numpy or torch, any shape
    delta : numpy or torch, any shape
    """
    B, C, H, W = data.shape
    device     = data.device

    m     = _reshape_mask(_to_tensor(m,     device), H, W)   # [1,1,H,W]
    delta = _reshape_mask(_to_tensor(delta, device), H, W)   # [1,C,H,W]

    triggered_data = data * (1 - m) + delta * m
    return triggered_data


def count_filter_activation_layer(model, x, layer_idx, filt_id):
    """
    Returns #samples whose activation of features[layer_idx][filt_id] > 0.
    """
    with torch.no_grad():
        for k in range(layer_idx + 1):
            x = model.features[k](x)
        act = (x[:, filt_id] > 0).any(dim=(1, 2))   # [B] bool
        return act.float().sum().item()


def compute_filter_activation_counts(model, test_loader, filter_idx, m, delta, device, trigger_size=5):
    """
    Computes and prints the activation counts for the targeted filter on clean and triggered images.
    
    Args:
        model: The neural network model.
        test_loader: DataLoader for the test set.
        filter_idx: The index of the target filter/ neuron to monitor.
        m (numpy.ndarray or Tensor): The trigger mask used to embed the trigger.
        delta (numpy.ndarray or Tensor): The trigger pattern.
        device: The device (e.g., "cuda:0").
        trigger_size: The trigger patch size (for consistency).
    
    Returns:
        A tuple (normal_activation, triggered_activation) containing the activation counts.
    """
    model.eval()
    total_samples = 0
    normal_activation = 0
    triggered_activation = 0
    # Loop over test batches.
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            batch_size = data.size(0)
            total_samples += batch_size
            # Count activations on the clean images.
            n_clean = count_filter_activation(model, data, device, filter_idx)
            normal_activation += n_clean
            # Create triggered images.
            data_triggered = add_trigger_to_batch(data, m, delta, trigger_size)
            n_triggered = count_filter_activation(model, data_triggered, device, filter_idx)
            triggered_activation += n_triggered

    print(f"Out of {total_samples} samples:")
    print(f"  Normal activation count for filter {filter_idx}: {normal_activation}")
    print(f"  Triggered activation count for filter {filter_idx}: {triggered_activation}")
    return normal_activation, triggered_activation

def model_testing(model, test_loader, test_type="Test ACC", y_tc=None, m=None, delta=None, device = 'cuda:0'):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if test_type == "ASR" and y_tc is not None:
                images, labels = add_trigger(images, labels, y_tc, m, delta)
            images = images.to(device)
            # outputs = get_embedding_resnet18(model, images)
            # get_embedding_resnet18_pretrain(model, images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        acc = 100 * correct / total
        print(f"{test_type}: {acc}%")

def count_filter_activation(model, images, device, filter_idx, threshold = 0.0):
    """
    Returns the number of samples in `images` for which the activation of filter `filter_idx`
    in the first convolutional layer is greater than zero.
    """
    model.eval()
    with torch.no_grad():
        # Support various models:
        if hasattr(model, 'cnn'):
            conv_layer = model.cnn[0]
        elif hasattr(model, 'conv1'):
            conv_layer = model.conv1
        elif hasattr(model, 'features'):
            conv_layer = model.features[0]  # For VGG or similar models.
        else:
            raise ValueError("Model does not have a recognized first conv layer attribute.")
    
        # Forward pass through the first conv layer
        activations = conv_layer(images.float().to(device))
        # activations shape: [B, out_channels, H, W]
        sample_active = (activations[:, filter_idx, :, :] > threshold).any(dim=(1, 2))
        count = sample_active.sum().item()
        
    return count

def get_filter_activation(model, images, device, mask, filter_idx):
    """
    Returns the number of samples in `images` for which the activation of filter `filter_idx`
    in the first convolutional layer is greater than zero.
    """
    model.eval()
    all_activation = []
    mask_size = int(np.sqrt(len(mask.nonzero())))
    with torch.no_grad():
        # Support various models:
        if hasattr(model, 'cnn'):
            conv_layer = model.cnn[0]
        elif hasattr(model, 'conv1'):
            conv_layer = model.conv1
        elif hasattr(model, 'features'):
            conv_layer = model.features[0]  # For VGG or similar models.
        else:
            raise ValueError("Model does not have a recognized first conv layer attribute.")
    
        # Forward pass through the first conv layer
        activations = conv_layer((images * mask).to(device))
        # activations shape: [B, out_channels, H, W]
        sample_active = (activations[:, filter_idx, :, :] > 0).any(dim=(1, 2))
        indices = mask.bool().nonzero(as_tuple=False)  # Shape [25, 2]
        activation = F.conv2d(images[:, :, indices[:, 0], indices[:, 1]].view(images.shape[0], images.shape[1], mask_size, mask_size).to(device), weight = conv_layer.weight.data, bias = conv_layer.bias.data)
        
        activation = activation[:, filter_idx, :, :].view(images.shape[0], )

        all_activation.extend(activation.tolist())
        count = sample_active.sum().item()
        
    return all_activation

def set_seeds(seed=0):
    import torch.backends.cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Some extra flags to make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model_fcn(args, model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.)
    n_total_step = len(train_loader)
    print_step = n_total_step // 4
    for epoch in range(args.epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % print_step == 0:
                print(
                    f'epoch {epoch + 1}/{args.epochs}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (test_images_set, test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(args.device)
                test_labels_set = test_labels_set.to(args.device)

                y_predicted = model(test_images_set)
                labels_predicted = y_predicted.argmax(axis=1)
                number_corrects += (labels_predicted == test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')

def train_model_cnn(args, model, train_loader, test_loader):
    iter = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.)

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.to(args.device)
                labels = labels.to(args.device)
            else:
                images = Variable(images)
                labels = Variable(labels)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    if torch.cuda.is_available():
                        images = images.to(args.device)
                        labels = labels.to(args.device)
                    else:
                        images = Variable(images)
                    # Forward pass only to get logits/output
                    outputs = model(images)
                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)
                    # Total number of labels
                    total += labels.size(0)

                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()
                torch.save(model, "CNN_MODEL.pth")
                accuracy = 100 * correct / total
                print(f'epoch: {epoch}, test ACC: {float(accuracy)}')

def train_model_resnet(model, train_loader, device, epochs=10, lr=0.01):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if (i+10) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {running_loss/len(train_loader):.4f}")
        running_loss = 0.0
    print("Finished training ResNet")


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    model = copy.deepcopy(model)
    model.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


