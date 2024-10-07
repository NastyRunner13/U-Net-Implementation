import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True), use_dropout=False):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, 
                padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation
        ]

        if use_dropout:
            layers.append(nn.Dropout(p=0.5))

        layers.extend([
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, 
                padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation
        ])

        if use_dropout:
            layers.append(nn.Dropout(p=0.5))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features=None, use_dropout=False):
        super(UNET, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(
                DoubleConv(
                    in_channels=in_channels,
                    out_channels=feature,
                    use_dropout=use_dropout
                ),
            )
            in_channels = feature
        
        self.bottleNeck = DoubleConv(
            in_channels=features[-1],
            out_channels=features[-1]*2,
            use_dropout=use_dropout
        )

        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(
                    in_channels=feature*2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.decoder.append(
                DoubleConv(
                    in_channels=feature*2,
                    out_channels=feature,
                    use_dropout=use_dropout
                )
            )


        self.finalConv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleNeck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)

        return self.finalConv(x)
    
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn((3, 1, 160, 160)).to(device)
    model = UNET(in_channels=1, out_channels=1).to(device)
    preds = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {preds.shape}")
    assert preds.shape == x.shape, "Output shape does not match input shape"

if __name__ == "__main__":
    test()
