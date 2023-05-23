# fetch pre-trained teacher models

#mkdir -p save/models/

cd save/models

mkdir -p resent50_imagenet_vanilla
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
mv resnet50-19c8e357.pth resent50_imagenet_vanilla/

mkdir -p resent34_imagenet_vanilla
wget https://download.pytorch.org/models/resnet34-333f7ec4.pth
mv resnet34-333f7ec4.pth resent34_imagenet_vanilla/


cd ../..