import argparse
import numpy as np



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment_name", type=str,
                        help="experiment containing the target file")
    parser.add_argument("--threshold", type=int, default= 10,
                        help="threshold for frequent sentences")
    parser.add_argument("--most_freq", type=int, default= 5,
                        help="how many of the most frequent sentences to print")

    args = parser.parse_args()

    f = open("resultsNLP/FinalResults/"+args.experiment_name+"/predictions.txt")
    sentence_count = {}
    for line in f.readlines():
        if line in sentence_count.keys():
            sentence_count[line] += 1
        else:
            sentence_count[line] = 1
    print("number of distinct sentences: ",len(sentence_count.values()))
    frequent_sentences = [sentence_count[s] for s in sentence_count.keys() if sentence_count[s] >= args.threshold]
    print("number of frequent sentences: ", len(frequent_sentences))
    most_freq = [s for s, v in sentence_count.items()
                 if v in sorted(sentence_count.values(), reverse=True)[:args.most_freq]]
    print("most frequent sentences: ")
    for s in most_freq:
        print s

