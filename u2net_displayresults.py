import numpy as np
processed_images = np.load('test_data/test_human_images_results/processed_images.npy')
print(processed_images.shape)  # Should print (N, 64, 64)
