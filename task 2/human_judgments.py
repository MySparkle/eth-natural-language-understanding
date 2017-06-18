import os
import argparse
import random
import glob
import numpy as np

def create_sample(judgments_dir, source_file, target_file, pred_file, sample_size):
    """
        Args:
            judgments_dir:  String, the directory containing the human judgments
            source_file:    String, the files containing the source sentences
            target_file:    String, the files containing the target sentences
            pred_file:      String, the files containing the predicted sentences
            sample_size:    int, number of sentences to be included in the sample
        
        Creates a file of sampled sentences to be annotated. Each entry is of the form: source sentence, target sentence, predicted sentence, empty line for annotation
    """
    f_s = open(source_file, "r", encoding="utf8")
    source_sentences = f_s.readlines()
    f_s.close()
    f_t = open(target_file, "r", encoding="utf8")
    target_sentences = f_t.readlines()
    f_t.close()
    f_p = open(pred_file, "r", encoding="utf8")
    pred_sentences = f_p.readlines()
    f_p.close()
    
    sample_file = judgments_dir + "/sampled_sentences.txt"
    f_sample = open(sample_file, "w", encoding="utf8")
    random.seed(42)
    samples = random.sample(range(len(source_sentences)), sample_size)
    for i in samples:
        f_sample.write("Source: " + source_sentences[i])
        f_sample.write("Target: " + target_sentences[i])
        f_sample.write("Prediction: " + pred_sentences[i])
        f_sample.write("Score: \n")
        f_sample.write("=======================================\n")
        f_sample.flush()
    f_sample.close()

def fleiss_kappa(M):
    """
        Args:
            M:              numpy matrix, A matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
        
        Computes `Fleiss' Kappa
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators    
    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)    
    kappa = (Pbar - PbarE) / (1 - PbarE)    
    return kappa

def evaluate(annotations_dir):
    """
        Args:
            annotations_dir:    String, the directory containing the annotated files
        
        Computes the evaluation scores (mean score, count for each score, agreement) based on the annotated files in judgments_dir.
    """
    
    scores = []
    for filename in glob.glob(annotations_dir + "/*.txt"):
        scores_part = []
        with open(filename, "r", encoding="utf8") as file:
            for line in file:
                if line.startswith("Score: "):
                    score = line.strip().split()[1]
                    scores_part.append(int(score))
        scores.append(scores_part)
    print("Scores: " + str(scores))
    flat_scores = [score for sublist in scores for score in sublist]
    mean_score = sum(flat_scores)/len(flat_scores)
    score_pct = {}
    score_pct[0] = flat_scores.count(0)/len(flat_scores)
    score_pct[1] = flat_scores.count(1)/len(flat_scores)
    score_pct[2] = flat_scores.count(2)/len(flat_scores)
    print("Mean score: " + str(mean_score))
    print("Score precentages: " + str(score_pct))
    scores_matrix = np.array(scores).T
    scores_matrix = np.array([[(s == i).sum() for i in range(3)] for s in np.array(scores).T])
    print("Scores matrix: \n" +str(scores_matrix))
    kappa = fleiss_kappa(scores_matrix)
    print("Fleiss' kappa: " + str(kappa))
    print(mean_score, score_pct, kappa)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--judgments_dir", type=str, default="human_judgments", 
                        help="directory containing the human judgments")
    parser.add_argument("--data_dir", type=str, default="validation", 
                        help="directory containing the source and target datasets")
    parser.add_argument("--pred_dir", type=str, default="model/pred", 
                        help="directory containing the predictions file")
    parser.add_argument("--create_sample", action="store_true", 
                        help="create a file of sampled sentences to be annotated")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="number of sentences in the sample")
    parser.add_argument("--annotations_dir", type=str, default="human_judgments", 
                        help="directory containing the annotated files")
    parser.add_argument("--evaluate", action="store_true", 
                        help="compute evaluation scores based on the annotated files")
    args = parser.parse_args()
    
    os.makedirs(args.judgments_dir, exist_ok=True)
    if args.create_sample:
        source_file = args.data_dir + "/source.txt"
        target_file = args.data_dir + "/target.txt"
        pred_file = args.pred_dir + "/predictions.txt"
        create_sample(args.judgments_dir, source_file, target_file, pred_file, args.sample_size)
    if args.evaluate:
        evaluate(args.annotations_dir)
