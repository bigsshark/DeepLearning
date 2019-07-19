import numpy as np
from tensorflow import keras
from utils import *

class RPNGererator(keras.utils.Sequence):
    """
    Generates data for Keras

            def generator_batch_triplet(data_list, dic_data_list, nbr_class_one=250, nbr_class_two=7,
                    batch_size=32, return_label=True, mode='train',
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):

    """

    def __init__(self, data_list, dic_data_list,nbr_class_model=250, nbr_class_color=7, batch_size=32,return_label=True,
                 mode='train',crop_method=center_crop,scale_ratio=1.0,random_scale=False,img_width=299,
                 img_height=299,shuffle=True,save_to_dir=None,save_network_input=None,augment=False):
        
        self.data_list = data_list
        self.dic_data_list = dic_data_list
        self.nbr_class_model = nbr_class_model
        self.nbr_class_color = nbr_class_color
        self.batch_size = batch_size
        self.return_label = True
        self.mode = mode
        self.crop_method = crop_method
        self.scale_ratio = scale_ratio
        self.random_scale = random_scale
        self.img_width = 299
        self.img_height = 299
        self.shuffle = shuffle
        self.save_to_dir = save_to_dir
        self.save_network_input = save_network_input
        self.augment = augment
        self.on_epoch_end()
        

    def __len__(self):
        
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        data_list_temp = [self.data_list[k] for k in indexes]

        # Generate data  y1 is model  y2 is color
        mini_batch = self.__data_generation(data_list_temp)

        return mini_batch

    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.data_list))
        
        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    def __data_generation(self, data_list_temp):


        """

        def generator_batch_triplet(data_list, dic_data_list, nbr_class_one=250, nbr_class_two=7,
                    batch_size=32, return_label=True, mode='train',
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    '''
    A generator that yields a batch of ([anchor, positive, negative], [class_one, class_two, pseudo_label]).

    Input:
        data_list  : a list of [img_path, vehicleID, modelID, colorID]
        dic_data_list: a dictionary: {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size
        mode       : generator used as for 'train', 'val'.
                     if mode is set 'train', dic_data_list has to be specified.
                     if mode is set 'val', dic_data_list could be a null dictionary: { }.
                     if mode is et 'feature_extraction', then return (X_anchor)


    Output:
        ([anchor, positive, negative], [class_one, class_two, pseudo_label]
    '''
        """
        # Initialization
        current_batch_size = len(data_list_temp)

        X_batch = np.zeros((current_batch_size, self.img_width, self.img_height, 3))
        Y_model_batch = np.zeros((current_batch_size, self.nbr_class_model))
        Y_color_batch = np.zeros((current_batch_size, self.nbr_class_color))
        Y_vid_batch = np.zeros((current_batch_size, 1))

        for i in np.arange(current_batch_size):

            line_result = data_list_temp[i].strip().split(' ')
            #print line
            if self.return_label:
                #         # line imagepath  vid model color
                label = (int(line_result[-3]),int(line_result[-2]),int(line_result[-1]))
            img_path = line_result[0]

            if self.random_scale:
                self.scale_ratio = np.random.uniform(0.9, 1.1)

            img = scale_byRatio(img_path, ratio=self.scale_ratio, return_width=self.img_width,
                                crop_method=self.crop_method)

            X_batch[i] = img

            if self.return_label:
                Y_model_batch[i , label[1]] = 1
                Y_color_batch[i , label[2]] = 1
                Y_vid_batch[i] = label[0]

        if self.augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if self.save_to_dir:
            for i in range(current_batch_size):
                tmp_path = data_list_temp[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(self.save_to_dir, image_name)
                img = array_to_img(X_batch[i ])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        

        img = X_batch[0,:,:,:]
        img = np.reshape(img, (-1))

        if self.return_label:
            return (X_batch,[Y_model_batch,Y_color_batch,Y_vid_batch])
        else:
            return X_batch
