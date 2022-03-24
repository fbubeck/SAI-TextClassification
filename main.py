from algorithms import Algorithm_1
from algorithms import Algorithm_2
from algorithms import Algorithm_3
from data import DataProvider
from data import DataPreprocessing
import json
from matplotlib import pyplot as plt
import pandas as pd


def main():
    print("Starting...")
    print("")

    # read config.json
    with open('config/config.json') as file:
        config = json.load(file)

    # Get Parameters from config file
    algo1_a_lr = config["Algorithm 1 - Run A"]["learning_rate"]
    algo1_a_epochs = config["Algorithm 1 - Run A"]["n_epochs"]
    algo1_a_id = config["Algorithm 1 - Run A"]["id"]
    algo1_a_opt = config["Algorithm 1 - Run A"]["opt"]
    algo1_b_lr = config["Algorithm 1 - Run B"]["learning_rate"]
    algo1_b_epochs = config["Algorithm 1 - Run B"]["n_epochs"]
    algo1_b_id = config["Algorithm 1 - Run B"]["id"]
    algo1_b_opt = config["Algorithm 1 - Run B"]["opt"]
    algo1_c_lr = config["Algorithm 1 - Run C"]["learning_rate"]
    algo1_c_epochs = config["Algorithm 1 - Run C"]["n_epochs"]
    algo1_c_id = config["Algorithm 1 - Run C"]["id"]
    algo1_c_opt = config["Algorithm 1 - Run C"]["opt"]
    algo2_a_epochs = config["Algorithm 2 - Run A"]["n_epochs"]
    algo2_a_solver = config["Algorithm 2 - Run A"]["solver"]
    algo2_a_c = config["Algorithm 2 - Run A"]["C"]
    algo2_a_penalty = config["Algorithm 2 - Run A"]["penalty"]
    algo2_a_id = config["Algorithm 2 - Run A"]["id"]
    algo2_b_epochs = config["Algorithm 2 - Run B"]["n_epochs"]
    algo2_b_solver = config["Algorithm 2 - Run A"]["solver"]
    algo2_b_c = config["Algorithm 2 - Run A"]["C"]
    algo2_b_penalty = config["Algorithm 2 - Run A"]["penalty"]
    algo2_b_id = config["Algorithm 2 - Run A"]["id"]
    algo2_c_epochs = config["Algorithm 2 - Run C"]["n_epochs"]
    algo2_c_solver = config["Algorithm 2 - Run A"]["solver"]
    algo2_c_c = config["Algorithm 2 - Run A"]["C"]
    algo2_c_penalty = config["Algorithm 2 - Run A"]["penalty"]
    algo2_c_id = config["Algorithm 2 - Run C"]["id"]
    algo3_a_lr = config["Algorithm 3 - Run A"]["learning_rate"]
    algo3_a_epochs = config["Algorithm 3 - Run A"]["n_epochs"]
    algo3_a_id = config["Algorithm 3 - Run A"]["id"]
    algo3_a_opt = config["Algorithm 3 - Run A"]["opt"]
    algo3_b_lr = config["Algorithm 3 - Run B"]["learning_rate"]
    algo3_b_epochs = config["Algorithm 3 - Run B"]["n_epochs"]
    algo3_b_id = config["Algorithm 3 - Run B"]["id"]
    algo3_b_opt = config["Algorithm 3 - Run B"]["opt"]
    algo3_c_lr = config["Algorithm 3 - Run C"]["learning_rate"]
    algo3_c_epochs = config["Algorithm 3 - Run C"]["n_epochs"]
    algo3_c_id = config["Algorithm 3 - Run C"]["id"]
    algo3_c_opt = config["Algorithm 3 - Run C"]["opt"]

    # Get Data (Preprocessed)
    Preprocessor = DataPreprocessing.DataPreprocessing()
    train_data, test_data = Preprocessor.clean_text()

    # Creating Algorithm Objects
    algo1_a = Algorithm_1.TensorFlow_CNN(train_data, test_data, algo1_a_lr, algo1_a_epochs, algo1_a_id, algo1_a_opt)
    algo1_b = Algorithm_1.TensorFlow_CNN(train_data, test_data, algo1_b_lr, algo1_b_epochs, algo1_b_id, algo1_b_opt)
    algo1_c = Algorithm_1.TensorFlow_CNN(train_data, test_data, algo1_c_lr, algo1_c_epochs, algo1_c_id, algo1_c_opt)

    algo2_a = Algorithm_2.TextClassifier_DBOW(train_data, test_data, algo2_a_epochs, algo2_a_solver, algo2_a_c, algo2_a_penalty, algo2_a_id)
    algo2_b = Algorithm_2.TextClassifier_DBOW(train_data, test_data, algo2_b_epochs, algo2_b_solver, algo2_b_c, algo2_b_penalty,algo2_b_id)
    algo2_c = Algorithm_2.TextClassifier_DBOW(train_data, test_data, algo2_c_epochs, algo2_c_solver, algo2_c_c, algo2_c_penalty,algo2_c_id)

    algo3_a = Algorithm_3.NeuralNetworkEmbeddingLayer(train_data, test_data, algo3_a_lr, algo3_a_epochs, algo3_a_id, algo3_a_opt)
    algo3_b = Algorithm_3.NeuralNetworkEmbeddingLayer(train_data, test_data, algo3_b_lr, algo3_b_epochs, algo3_b_id, algo3_b_opt)
    algo3_c = Algorithm_3.NeuralNetworkEmbeddingLayer(train_data, test_data, algo3_c_lr, algo3_c_epochs, algo3_c_id, algo3_c_opt)

    # Universal Sentence Encoder
    algo1_a_trainingDuration, algo1_a_trainingError = algo1_a.train()
    algo1_a_testDuration, algo1_a_testError = algo1_a.test()
    algo1_a.plot()
    algo1_b_trainingDuration, algo1_b_trainingError = algo1_b.train()
    algo1_b_testDuration, algo1_b_testError = algo1_b.test()
    algo1_c_trainingDuration, algo1_c_trainingError = algo1_c.train()
    algo1_c_testDuration, algo1_c_testError = algo1_c.test()
    algo1_c.plot()

    # DBOW + LogReg
    algo2_a_trainingDuration, algo2_a_trainingError = algo2_a.train()
    algo2_a_testDuration, algo2_a_testError = algo2_a.test()
    algo2_b_trainingDuration, algo2_b_trainingError = algo2_b.train()
    algo2_b_testDuration, algo2_b_testError = algo2_b.test()
    algo2_c_trainingDuration, algo2_c_trainingError = algo2_c.train()
    algo2_c_testDuration, algo2_c_testError = algo2_c.test()

    # NeuralNetworkEmbeddingLayer
    algo3_a_trainingDuration, algo3_a_trainingError = algo3_a.train()
    algo3_a_testDuration, algo3_a_testError = algo3_a.test()
    algo3_a.plot()
    algo3_b_trainingDuration, algo3_b_trainingError = algo3_b.train()
    algo3_b_testDuration, algo3_b_testError = algo3_b.test()
    algo3_b.plot()
    algo3_c_trainingDuration, algo3_c_trainingError = algo3_c.train()
    algo3_c_testDuration, algo3_c_testError = algo3_c.test()
    algo3_c.plot()

    # Plots
    px = 1 / plt.rcParams['figure.dpi']

    training1 = {'error': [algo1_a_testError, algo1_b_testError, algo1_c_testError],
                 'duration': [algo1_a_trainingDuration, algo1_b_trainingDuration, algo1_c_trainingDuration],
                 'Run': ["A", "B", "C"]
                 }

    training2 = {'error': [algo2_a_testError, algo2_b_testError, algo2_c_testError],
                 'duration': [algo2_a_trainingDuration, algo2_b_trainingDuration, algo2_c_trainingDuration],
                 'Run': ["A", "B", "C"]
                 }

    training3 = {'error': [algo3_a_testError, algo3_b_testError, algo3_c_testError],
                 'duration': [algo3_a_trainingDuration, algo3_b_trainingDuration, algo3_c_trainingDuration],
                 'Run': ["A", "B", "C"]
                 }

    inference1 = {'error': [algo1_a_testError, algo1_b_testError, algo1_c_testError],
                  'duration': [algo1_a_testDuration, algo1_b_testDuration, algo1_c_testDuration],
                  'Run': ["A", "B", "C"]
                  }

    inference2 = {'error': [algo2_a_testError, algo2_b_testError, algo2_c_testError],
                  'duration': [algo2_a_testDuration, algo2_b_testDuration, algo2_c_testDuration],
                  'Run': ["A", "B", "C"]
                  }

    inference3 = {'error': [algo3_a_testError, algo3_b_testError, algo3_c_testError],
                  'duration': [algo3_a_testDuration, algo3_b_testDuration, algo3_c_testDuration],
                  'Run': ["A", "B", "C"]
                  }

    data_training1 = pd.DataFrame(training1)
    data_training2 = pd.DataFrame(training2)
    data_training3 = pd.DataFrame(training3)
    data_inference1 = pd.DataFrame(inference1)
    data_inference2 = pd.DataFrame(inference2)
    data_inference3 = pd.DataFrame(inference3)
    data_training1.sort_values(by=['duration'], inplace=True)
    data_training2.sort_values(by=['duration'], inplace=True)
    data_training3.sort_values(by=['duration'], inplace=True)
    data_inference1.sort_values(by=['duration'], inplace=True)
    data_inference2.sort_values(by=['duration'], inplace=True)
    data_inference3.sort_values(by=['duration'], inplace=True)

    fig = plt.figure(figsize=(1200 * px, 800 * px))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Duration [in seconds]')
    ax1.set_ylabel('Model-Performance (Accuracy)')
    ax2.set_xlabel('Duration [in seconds]')
    ax2.set_ylabel('Model-Performance (Accuracy)')
    fig.suptitle('Efficiency of different ML-Algorithms and Parametersets')
    ax1.plot(data_training1["duration"], data_training1["error"], '-o', c='blue', alpha=0.6)
    ax1.plot(data_training2["duration"], data_training2["error"], '-o', c='green', alpha=0.6)
    ax1.plot(data_training3["duration"], data_training3["error"], '-o', c='red', alpha=0.6)
    ax2.plot(data_inference1["duration"], data_inference1["error"], '-o', c='blue', alpha=0.6)
    ax2.plot(data_inference2["duration"], data_inference2["error"], '-o', c='green', alpha=0.6)
    ax2.plot(data_inference3["duration"], data_inference3["error"], '-o', c='red', alpha=0.6)
    ax1.title.set_text('Training')
    ax2.title.set_text('Inference')
    plt.legend(["Universal Sentence Encoder + Neural Network", "DBOW Model + Logistic Regression",
                "Neural Network with Embedding Layer"], loc='lower center', ncol=4, bbox_transform=fig.transFigure,
               bbox_to_anchor=(0.5, 0))
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    for i in range(3):
        ax1.annotate(data_training1["Run"][i], xy=(data_training1["duration"][i], data_training1["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax1.annotate(data_training2["Run"][i], xy=(data_training2["duration"][i], data_training2["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax1.annotate(data_training3["Run"][i], xy=(data_training3["duration"][i], data_training3["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax2.annotate(data_inference1["Run"][i], xy=(data_inference1["duration"][i], data_inference1["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax2.annotate(data_inference2["Run"][i], xy=(data_inference2["duration"][i], data_inference2["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax2.annotate(data_inference3["Run"][i], xy=(data_inference3["duration"][i], data_inference3["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
    plt.savefig('plots/Algorithms_Evaluation.png')
    # plt.show()
    print("Evaluation Plot saved...")
    print("")


if __name__ == "__main__":
    main()
