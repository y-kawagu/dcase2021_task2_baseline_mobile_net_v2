########################################################################
# import default libraries
########################################################################
import os
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
import keras_model
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = com.file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data


########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue
        
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)
        # pickle file for storing section names
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                    machine_type=machine_type)
        # pickle file for storing anomaly score distribution
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                machine_type=machine_type)
        
        # get section names from wave file names
        section_names = com.get_section_names(target_dir, dir_name="train")
        unique_section_names = np.unique(section_names)
        n_sections = unique_section_names.shape[0]
        
        # make condition dictionary
        joblib.dump(unique_section_names, section_names_file_path)

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        # number of wave files in each section
        # required for calculating y_pred for each wave file
        n_files_ea_section = []
        
        data = np.empty((0, param["feature"]["n_frames"] * param["feature"]["n_mels"]), float)
        
        for section_idx, section_name in enumerate(unique_section_names):

            # get file list for each section
            # all values of y_true are zero in training
            files, y_true = com.file_list_generator(target_dir=target_dir,
                                                    section_name=section_name,
                                                    dir_name="train",
                                                    mode=mode)

            n_files_ea_section.append(len(files))

            data_ea_section = file_list_to_data(files,
                                                msg="generate train_dataset",
                                                n_mels=param["feature"]["n_mels"],
                                                n_frames=param["feature"]["n_frames"],
                                                n_hop_frames=param["feature"]["n_hop_frames"],
                                                n_fft=param["feature"]["n_fft"],
                                                hop_length=param["feature"]["hop_length"],
                                                power=param["feature"]["power"])

            data = np.append(data, data_ea_section, axis=0)

        # number of all files
        n_all_files = sum(n_files_ea_section)
        # number of vectors for each wave file
        n_vectors_ea_file = int(data.shape[0] / n_all_files)

        # make one-hot vector for conditioning
        condition = np.zeros((data.shape[0], n_sections), float)
        start_idx = 0
        for section_idx in range(n_sections):
            n_vectors = n_vectors_ea_file * n_files_ea_section[section_idx]
            condition[start_idx : start_idx + n_vectors, section_idx : section_idx + 1] = 1
            start_idx += n_vectors

        # 1D vector to 2D image
        data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)

        # train model
        print("============== MODEL TRAINING ==============")
        model = keras_model.get_model(param["feature"]["n_frames"], 
                                      param["feature"]["n_mels"],
                                      n_sections,
                                      param["fit"]["lr"])

        model.summary()

        history = model.fit(x=data,
                            y=condition,
                            epochs=param["fit"]["epochs"],
                            batch_size=param["fit"]["batch_size"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"])

        # calculate y_pred for fitting anomaly score distribution
        y_pred = []
        start_idx = 0
        for section_idx in range(n_sections):
            for file_idx in range(n_files_ea_section[section_idx]):
                p = model.predict(data[start_idx : start_idx + n_vectors_ea_file, : , :, :])[:, section_idx : section_idx + 1]
                y_pred.append(np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon) 
                                      - np.log(np.maximum(p, sys.float_info.epsilon)))))
                start_idx += n_vectors_ea_file

        # fit anomaly score distribution
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
        gamma_params = [shape_hat, loc_hat, scale_hat]
        joblib.dump(gamma_params, score_distr_file_path)
        
        visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        visualizer.save_figure(history_img)
        model.save(model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")

        del data
        del condition
        del model
        keras_model.clear_session()
        gc.collect()
        
