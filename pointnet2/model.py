import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class PointNet(nn.Module): 
    def __init__(self): 
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, 1) 
        self.bn4 = nn.BatchNorm1d(128) 

        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024) 

        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.7)

        self.fc3 = nn.Linear(256, 1)

        # initialize 
        for m in self.modules():
            if isinstance(m, nn.Conv1d): 
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.Linear): 
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)

    # TODO: 
    # - make sure weights are initialized with Xavier 
    # - we do bn then relu 
    def forward(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = torch.max(x, 2, True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.fc3(self.dp1(x))

        return x