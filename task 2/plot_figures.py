import argparse
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--losses_dir", type=str, default="resultsNLP/losses", 
                        help="directory containing the losses of the models")
    args = parser.parse_args()
    
    steps = []
    losses = []
    files = ["baseline_LSTM", "baseline_GRU", "bahdanau", "dot", "convolutional", "residual", "residualTriple", "residual1024", "residual2048_2layers", "residual1024triple"]
    models = files
    for file in files:
        steps_model = []
        losses_model = []
        filename = "{}/{}.csv".format(args.losses_dir, file)
        with open(filename, "r", encoding="utf8") as file:
            for line in file:
                if line.startswith("Wall"):
                    continue
                tokens = line.strip().split(",")
                steps_model.append(int(tokens[1]))
                losses_model.append(float(tokens[2]))
        steps.append(steps_model)
        losses.append(losses_model)
    print("models: " + str(models))
    print("steps: " + str(steps))
    print("losses: " + str(losses))

    max_step = 45000
    ticks = range(0, max_step, 5000)
    #colors = [(27,158,119), (217,95,2), (117,112,179), (231,41,138), (102,166,30), (230,171,2)]
    colors = [(166,206,227), (31,120,180), (178,223,138), (51,160,44), (251,154,153), (227,26,28), (253,191,111), (255,127,0), (202,178,214), (106,61,154), (255,255,153), (177,89,40)]
    colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in colors]
    lines = ["-", "--", "-.", ":"]
    
    plt.figure()
    for i in range(len(models)):
        plt.plot(steps[i], losses[i], linestyle=lines[i%len(lines)], color=colors[i%len(colors)], label=models[i])
    plt.legend(loc="best")
    plt.xticks(ticks)
    plt.xlim(xmax=max_step)
    plt.grid()
    plt.xlabel("Training steps")
    plt.ylabel("Evaluation loss")
    plt.savefig(args.losses_dir + "/losses.png")
