import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # todo The code defines the initializer method (__init__) for the ResidualBlock class. This method runs automatically when a new object of the class is created.
# !def __init__(self, in_channels, out_channels, stride=1):
# ? This defines the constructor for the ResidualBlock class.
# ? self: A reference to the specific instance of the ResidualBlock that is being created.
# ? in_channels: The number of input channels or feature maps that the block will receive. This is a crucial parameter for setting up the convolutional layers.
# ? out_channels: The number of output channels or feature maps the block should produce. This may be different from in_channels, especially if the block changes the number of feature maps.
# * stride=1: An optional parameter that defaults to 1. The stride determines the step size of the convolution operation and can be used to reduce the spatial dimensions (height and width) of the input, a technique known as downsampling.
#! super().__init__()
# ? super(): This function returns a temporary object of the parent class. In this case, the parent class is nn.Module.
# ? .: The dot operator is used to access methods of the temporary parent object.
# ? __init__(): This calls the constructor of the parent class, nn.Module.
# * Purpose of super().__init__() in this context:
# ? Calling the parent's constructor is a required step when building custom modules in PyTorch. The nn.Module initializer sets up important, internal features of the module, such as:
# ? A mechanism to automatically track and register all sub-modules (the layers you will define next) and their parameters (weights and biases).
# ? Hooks for the forward and backward passes during training.
# ? Functionality to move the model to different devices, like a GPU.
#! Without this line, the ResidualBlock would not function correctly as a PyTorch module.

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, padding=1, bias=False)

# todo         Setting bias=False is a common and important optimization when using Batch Normalization.

# * What a Bias Term Does
# ? A bias is a learnable number that is added to every element of the output after the convolution operation.
# Think of the linear equation y = mx + b.
# The convolution (m*x) is the main feature extraction step.
# The bias (b) acts like the y-intercept. It's a simple shift that allows the network to adjust the output up or down, giving it more flexibility.

# * Why bias=False is Used with Batch Normalization
# ? As the diagram you sent shows, a BatchNorm2d layer almost always follows a Conv2d layer. The key is that the Batch Normalization layer has its own learnable bias, which makes the one in the convolution redundant.

# * Here's how BatchNorm works:
# Normalize: It first normalizes its input by subtracting the mean and dividing by the standard deviation of the batch. This centers the data around zero.
# Scale and Shift: It then multiplies the result by a learnable parameter gamma (γ) and adds another learnable parameter, beta (β).
# This beta (β) in the BatchNorm layer serves the exact same purpose as the bias in the Conv2d layer—it adds a learnable shift to the output.

# ? If the Conv2d layer adds its bias, the very next BatchNorm layer will immediately subtract the mean, effectively canceling out the bias's effect. The BatchNorm layer's own beta parameter will then apply the necessary shift.
# ? Since the convolution's bias is made redundant by the Batch Normalization layer, we set bias=False to save a few parameters and computations. 🧠

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x, fmap_dict=None, prefix=""):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        out = torch.relu(out_add)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

        return out


class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])


# * nn.ModuleList - The Smart List
# First, the most important component here is nn.ModuleList. It looks like a regular Python list, but it has one critical difference: it properly registers all the layers it contains with the parent model.
# This means when you call model.parameters() or model.to('cuda'), PyTorch will automatically find all the ResidualBlocks inside the ModuleList. If you were to use a plain Python list [...], PyTorch would not see the layers inside it, and they would not be trained.

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# *         nn.AdaptiveAvgPool2d((1,1))
# This is a type of global average pooling. It takes the feature map from the last convolutional layer, which might have any height and width (e.g., 8x8), and reduces each channel to a single number by averaging all the values in that channel.
# The "Adaptive" part is the key. Instead of you specifying a kernel size, you specify the desired output size. By setting it to (1,1), you are telling the layer, "I don't care what the input size is, just average it down until the height and width of each feature map is 1x1."
# This makes the network flexible and able to handle input images of different sizes.

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        if not return_feature_maps:
            x = self.conv1(x)
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x
        else:
            feature_maps = {}
            x = self.conv1(x)
            feature_maps["conv1"] = x

            for i, block in enumerate(self.layer1):
                x = block(x, feature_maps, prefix=f"layer1.block{i}")
            feature_maps["layer1"] = x

            for i, block in enumerate(self.layer2):
                x = block(x, feature_maps, prefix=f"layer2.block{i}")
            feature_maps["layer2"] = x

            for i, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix=f"layer3.block{i}")
            feature_maps["layer3"] = x

            for i, block in enumerate(self.layer4):
                x = block(x, feature_maps, prefix=f"layer4.block{i}")
            feature_maps["layer4"] = x

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x, feature_maps
