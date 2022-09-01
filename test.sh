#python3 evaluation.py --data_path /mnt/bd/aurora-mtrc-data/data/giana/segmentation/test/hdtest --model_upsample_num 5 --result_save_path ./result --model_path ./checkpoint/segformer/20210922_184747/checkpoint.pth.tar --model segformer 

python3 evaluation.py --data_path /mnt/bd/aurora-mtrc-arnold/data/miccai/test/PolypHD --model_upsample_num 5 --result_save_path ./hight --model_path ./checkpoint/unet/20220228_203342/model_best.pth.tar --model unet --crop_black true
