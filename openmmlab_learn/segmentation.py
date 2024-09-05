import subprocess

from mmseg.apis import inference_model, init_model, show_result_pyplot

from Vision import VISION_ROOT


def main():
    subprocess.run(
        ["mim", "download", "mmsegmentation", "--config", "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024", "--dest", "."]
    )
    config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
    checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    model = init_model(config_file, checkpoint_file, device='cpu')
    img = str(VISION_ROOT / r'data/Ahri/Popstar Ahri.jpg')
    result = inference_model(model, img)
    show_result_pyplot(model, img, result, show=True)


if __name__ == "__main__":
    main()
