from torch.nn.modules.activation import LeakyReLU
class MesoNet(nn.Module):
	def __init__(self,input_features:int, output_features:int):
		super().__init__()

		self.convBlock1 = nn.Sequential(
				nn.Conv2d(in_channels=3,out_channels=100, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(100),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=100, out_channels=250, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(250),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(250),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(250),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=250, out_channels=8, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(8),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1),
				nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(8),
				nn.ReLU(inplace=True),
				nn.LeakyReLU(0.1)
		)
	
		self.convBlock2 = nn.Sequential(
				nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2, bias=False),
				nn.BatchNorm2d(8),
				nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, bias=False),
				nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2, bias=False),
				nn.MaxPool2d(kernel_size=2),
				nn.MaxPool2d(kernel_size=4)
		)

		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(1024, 16)
		self.fc2 = nn.Linear(16, output_features)
		self.leakRelu= nn.LeakyReLU(0.1)

	def forward(self,x):
		x = self.convBlock1(x)
		x = self.convBlock2(x)
		x = x.view(x.size(0),-1)
		x = self.dropout(x)
		x = self.fc1(x)
		x = self.leakRelu(x)
		x=  self.dropout(x)
		x = self.fc2(x)
	
		return x
