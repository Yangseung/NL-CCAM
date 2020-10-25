import torch
import torch.nn as nn

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mix_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mix_conv.weight.data.fill_(0.0)
        # self.mix_conv.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.relu(self.bn(self.mix_conv(out)))
        out = (self.bn(self.mix_conv(out)))
        out = out + x
        # print(self.gamma)
        return out

class Self_Attn_low(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn_low, self).__init__()
        self.channel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mix_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mix_conv.weight.data.fill_(0.0)
        # self.mix_conv.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.relu(self.bn(self.mix_conv(out)))
        out = (self.bn(self.mix_conv(out)))
        out = out + x
        # print(self.gamma)
        return out

class CNN(nn.Module):
    """
    Simple CNN NETWORK
    """
    def __init__(self, pretrain=True, num_classes=200):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
	    # 3 x 128 x 128
	    nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2),
	    # nn.BatchNorm2d(64),
        nn.MaxPool2d(2,2),
        # Self_Attn(64),
	    # 32 x 128 x 128
	    nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2),
	    # nn.BatchNorm2d(128),
        nn.MaxPool2d(2,2),
        Self_Attn(128),
	    # 64 x 128 x 128
	    nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.MaxPool2d(2,2),
        Self_Attn(256),
	    # 64 x 64 x 64
        nn.Conv2d(256, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.MaxPool2d(2, 2),
		Self_Attn(512),
	    # 128 x 64 x 64
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2),
        # nn.MaxPool2d(2, 2),
        Self_Attn(512)
	)
	# 256 x 32 x 32
        self.avg_pool = nn.AvgPool2d(14)
        # 256 x 1 x 1

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
        )

        if pretrain:
            self.weights_pretrain()
            # self.weights_init()
            print('pretrained weight load complete..')

    def weights_pretrain(self):
        k = 0
        pretrained_weights = torch.load('model/vgg16_imgnet.pth')
        pretrained_list = pretrained_weights.keys()
        for i, layer_name in enumerate(pretrained_list):
            layer_num = int(layer_name.split('.')[1])
            layer_group = layer_name.split('.')[0]
            layer_type = layer_name.split('.')[-1]
            if layer_num >= 10:
                layer_num = layer_num + 1
            if layer_num >= 17:
                layer_num = layer_num + 1
            if layer_num >= 24:
                layer_num = layer_num + 1

            if layer_group != "features":
                break

            if layer_type == 'weight':
                assert self.conv[layer_num].weight.data.size() == pretrained_weights[
                    layer_name].size(), "size error!"
                self.conv[layer_num].weight.data = pretrained_weights[layer_name]
            else:  # layer type == 'bias'
                assert self.conv[layer_num].bias.size() == pretrained_weights[layer_name].size(), "size error!"
                self.conv[layer_num].bias.data = pretrained_weights[layer_name]

    def forward(self, x):
        features = self.conv(x)
        self.feature_map = features
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        self.pred = output
        return output, features

    def get_cam(self):
        return self.feature_map, self.pred

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_net(num_classes):
    net = CNN(num_classes=num_classes)
    # net.apply(weight_init)
    print("INIT NETWORK")
    return net

def load_net(num_classes, model_name):
    net = CNN(num_classes=num_classes)
    net.load_state_dict(torch.load(model_name))
    return net

def check():
    input = torch.rand(1,3,224,224)
    net = CNN()
    output = net(input)
    print(output)

if __name__ == "__main__":
    check()