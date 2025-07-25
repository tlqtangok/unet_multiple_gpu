readme
===
this is a patch for the Pytorch-UNet project, to use multiple GPUs and ONNX inference, include a cpp inference program



common run commands
===

train
---
python train.py --epochs 10 --batch-size 600 --learning-rate 1e-6 --scale 1 --validation 5 -c 2 --amp --load ./checkpoints/checkpoint_epoch6.pth


predict
---
python $m/predict.py -s 1 --classes 2 --model /home/jd/t/Pytorch-UNet-master/checkpoints/checkpoint_epoch2.pth -i /home/jd/t/Pytorch-UNet-master/data/infer/imgs/*.png


pth to onnx
---
python export_onnx.py --checkpoint ./checkpoints/checkpoint_epoch1.pth  --output ./checkpoints/checkpoint_epoch1.onnx


view model in detail
---
python show_model_detail.py  --model ./checkpoints/checkpoint_epoch1.onnx


build and run cpp
---
cd build && cmake .. && cmake --build . --config Release -v &&  ./unet_inference   --model $m/checkpoints/checkpoint_epoch1.onnx  --gpu  --input $m/data/infer/imgs --output $m/data/infer/imgs  --batch 2

