import kagglehub

kagglehub.login()
# Download latest version
path = kagglehub.model_download("migueldilalla/lego_bricks_machinevisonyolofinetune/pyTorch/default")

print("Path to model files:", path)