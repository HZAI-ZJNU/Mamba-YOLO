from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/mamba-yolo.yaml")
    model.train(
        data="ultralytics/cfg/datasets/VisDrone.yaml",
        epochs=10,
        workers=8,
        batch=16,
        optimizer="SGD",
        device=0,
        amp=True,
        project="./output_dir/VisDrone",
        name='mambayolo_n',
    )