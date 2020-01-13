#多进程训练
#1.准备imagenet训练数据,放到目录./data/ILSVRC2012
#2.运行下面命令,启动训练,日志在./mylog.all输出
CUDA_VISIBLE_DEVICES=6,7 python -m paddle.distributed.launch --log_dir ./mylog.all mobilenet_v1.py --use_data_parallel 1 --batch_size=256     --reader_thread=8    --total_images=1281167    --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output/ --lr_strategy=piecewise_decay --lr=0.1   --data_dir=./data/ILSVRC2012
