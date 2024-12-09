import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET  # full size version 173.6 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def process_image(pred):
    """
    Processes a prediction to crop blank space and resize to (64, 64).
    Returns the processed image as a NumPy array.
    """
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    # Normalize the image to 0-255
    im = Image.fromarray((predict_np * 255).astype(np.uint8))

    # Crop the image to remove blank space
    bbox = im.getbbox()  # Get bounding box of non-zero regions
    if bbox:
        im = im.crop(bbox)

    # Resize to 64x64
    im = im.resize((64, 64), resample=Image.BILINEAR)

    # Convert back to NumPy array
    return np.array(im, dtype=np.uint8)


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_images' + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '_human_seg', model_name + '_human_seg.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    processed_images = []
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("Inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # Process image and append to the list
        processed_image = process_image(pred)
        processed_images.append(processed_image)

        del d1, d2, d3, d4, d5, d6, d7

    # Convert the list of processed images to a NumPy array (N, 64, 64)
    processed_images_np = np.stack(processed_images, axis=0)

    # Save the result as a NumPy array
    npy_output_path = os.path.join(prediction_dir, 'processed_images.npy')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    np.save(npy_output_path, processed_images_np)

    print(f"Processed images saved as NumPy array: {npy_output_path}")
    print(f"Shape of processed array: {processed_images_np.shape}")


if __name__ == "__main__":
    main()
