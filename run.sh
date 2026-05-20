python src/train_iggm_lightning.py --config config/test_debug.yaml

python src/train_iggm_lightning.py --config config/train_0306.yaml

python src/train_iggm_lightning.py --config config/train_0507_2.yaml

python src/train_iggm_lightning.py --config config/train_0507.yaml




nohup python src/train_iggm_lightning.py --config config/train_0507.yaml > log_runout/train_0507.log 2>&1 &
nohup python src/train_iggm_lightning.py --config config/train_0518.yaml > log_runout/train_0518-v6.log 2>&1 &
nohup python src/train_iggm_lightning.py --config config/train_0518_signleGPU.yaml > log_runout/train_0518_signleGPU_allloss520.log 2>&1 &
# nohup python src/train_iggm_lightning.py --config config/train_0518_signleGPU.yaml > log_runout/train_0518_signleGPU_nobbloss.log 2>&1 &

# # 环境
# conda env update -f envirs.yaml
