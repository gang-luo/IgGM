python src/train_iggm_lightning.py --config config/test_debug.yaml

python src/train_iggm_lightning.py --config config/train_0306.yaml

python src/train_iggm_lightning.py --config config/train_0507_2.yaml

python src/train_iggm_lightning.py --config config/train_0507.yaml




# nohup python src/train_iggm_lightning.py --config config/train_0507.yaml > log_runout/train_0507.log 2>&1 &
# nohup python src/train_iggm_lightning.py --config config/train_0518.yaml > log_runout/train_0518-v6.log 2>&1 &
# nohup python src/train_iggm_lightning.py --config config/train_0518_signleGPU.yaml > log_runout/train_0518_signleGPU_allloss.log 2>&1 &
nohup python src/train_iggm_lightning.py --config config/train_0518.yaml > log_runout/train_0518_S3.log 2>&1 &

python src/train_iggm_lightning.py --config config/train_0526_signle.yaml

python src/train_iggm_lightning.py --config config/train_0526_signle2.yaml

python src/train_iggm_lightning.py --config config/train_finally.yaml


# # 环境
# conda env update -f envirs.yaml



nohup python src/train_iggm_lightning.py --config config/train_0526_signle.yaml > log_runout/train_0526_signle1.log 2>&1 &


# 终版debug+training
# python src/train_iggm_lightning.py --config config/train_finally.yaml

python src/train_iggm_lightning.py --config config/train_finally_debug.yaml

python src/train_iggm_lightning.py --config config/train_finally_debug_overfit.yaml


python src/train_iggm_lightning.py --config config/train_overfit_0708.yaml
python src/train_iggm_lightning.py --config config/train_overfit_070802.yaml