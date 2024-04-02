# coding : utf-8
# Author : yuxiang Zeng

if __name__ == '__main__':
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import *
    # function
    def process_image(inputs):
        imageTranfer, image_name = inputs
        file_name = '../BigDataSource/Teddy2024/附件2/ImageData/' + image_name
        image = Image.open(file_name)
        features = imageTranfer.image2tensor(image)
        return features


    # input
    inputList = [(imageTranfer, imageNames) for imageNames in allImages]

    # Execute
    allImagesFeatures = []
    allImages = os.listdir('../BigDataSource/Teddy2024/附件2/ImageData')
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_image, inputs) for inputs in inputList]
        for future in tqdm(as_completed(futures), total=len(allImages)):
            allImagesFeatures.append(future.result())
    print('Done!')