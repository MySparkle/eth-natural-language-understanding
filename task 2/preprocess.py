import os
import argparse
import re
import ast
import nltk

def construct_vocabulary(train_dir, train_files, vocab_size):
    """
        Args:
            train_dir:      String, directory to store the vocabulary
            train_files:    List[String], filenames of the training data
            vocab_size:     int, size of the vocabulary
        
        Constructs the vocabulary of the training set
    """
    print("Constructing wordToID dictionary...")
    word_frequencies = {}
    
    for file_name in train_files:    
        file = open(file_name, "r", encoding="utf8")
        i = 0
        for line in file:
            if i % 10000 == 0:
                print("at line:", i)
            for word in line.split():
                if word in word_frequencies.keys():
                    word_frequencies[word] += 1
                else:
                    word_frequencies[word] = 1
            i += 1
        file.close()
    
    output_file = train_dir + "/vocab.txt"
    file = open(output_file, "w", encoding="utf8")
    # Sort words with respect to frequency to pick the vocab_size most frequent words
    for word in sorted(word_frequencies, key=word_frequencies.get, reverse=True)[:vocab_size]:
        file.write("{}\t{}\n".format(word, word_frequencies[word]))
        file.flush()
    file.write("<utterer_change>\t1\n")
    file.flush()
    file.close()
    
def transform_CMC(data_dir):
    """
        Args:
            data_dir:      String, directory containing the Cornell Movie Corpus dataset
        
        Transforms the Cornell Movie Corpus to match the MovieTriples dataset structure
    """
    print("Transforming the Cornell Movie Corpus...")
    lines_file = data_dir + "/CMC_movie_lines.txt"
    f_l = open(lines_file, "r", encoding="utf8")
    lines = {}
    i = 0
    for line in f_l:
        if i % 10000 == 0:
            print("at line:", i)
        fields = line.split(" +++$+++ ")
        lineID = fields[0]
        lineText = fields[-1]
        # Preprocess line to match the MovieTriples dataset preprocessing
        lineText = lineText.strip()
        lineText = lineText.replace("<u>", "")
        lineText = lineText.replace("</u>", "")
        lineText = lineText.replace("<b>", "")
        lineText = lineText.replace("</b>", "")
        lineText = lineText.replace("<i>", "")
        lineText = lineText.replace("</i>", "")
        tokens = nltk.word_tokenize(lineText)
        pos_tags = nltk.pos_tag(tokens)
        ne_tree = nltk.ne_chunk(pos_tags)
        for subtree in ne_tree.subtrees():
            if subtree.label() == "PERSON":
                name, _ = zip(*subtree.leaves())
                name = " ".join(name)
                #print("PERSON: " + name)
                lineText = lineText.replace(name, "<person>")
        lineText = re.sub("([.,!?():'\"])", r" \1 ", lineText)
        tokens = nltk.word_tokenize(lineText)
        tokens = ["<number>" if token.isdigit() else token for token in tokens]
        finalLine = " ".join(tokens).lower()
        finalLine = finalLine.replace("< person >", "<person>")
        finalLine = finalLine.replace("< number >", "<number>")
        finalLine = finalLine.replace("<number> , <number>", "<number>")
        lines[lineID] = finalLine
        i += 1
    f_l.close()
    
    conversations_file = data_dir + "/CMC_movie_conversations.txt"
    f_c = open(conversations_file, "r", encoding="utf8")
    preprocessed_file = data_dir + "/CMC_preprocessed.txt"
    f_p = open(preprocessed_file, "w", encoding="utf8")
    i = 0
    for line in f_c:
        if i % 10000 == 0:
            print("at line:", i)
        conversation_IDs = ast.literal_eval(line.split(" +++$+++ ")[-1])
        conversation = "\t".join([lines[lineID] for lineID in conversation_IDs])
        f_p.write(conversation + "\n")
        f_p.flush()
        i += 1
    f_c.close()
    f_p.close()
    
def preprocess(dir, files, exploit_triple=False):
    """
        Args:
            dir:            String, directory to store the preprocessed data
            files:          List[String], filenames of the files containing the raw data
            exploit_triple: Boolean, use the first turn when predicting the third
        
        Preprocesses some dataset files and saves the result
    """
    print("Preprocessing sentences from {}...".format(files))
    source_file = dir + "/source.txt"
    target_file = dir + "/target.txt"
    f_s = open(source_file, "w", encoding="utf8")
    f_t = open(target_file, "w", encoding="utf8")
    
    for file_name in files:
        file = open(file_name, "r", encoding="utf8")
        i = 0
        for line in file:
            if i % 10000 == 0:
                print("at line:", i)
            # Correct mistonekizations
            line = line.replace("<person> ' t", "don ' t")
            line = line.replace("<person>s", "<person>")
            line = line.replace("<person>", " <person> ")
            line = line.replace("<number>", " <number> ")
            line = line.replace("<continued_utterance>", " <continued_utterance> ")
            
            sentences = line.split("\t")
            sentences = [" ".join(s.split()) for s in sentences]
            for sentence_no in range(len(sentences) - 1):
                f_s.write(sentences[sentence_no] + "\n")
                f_s.flush()
                f_t.write(sentences[sentence_no + 1] + "\n")
                f_t.flush()
                if exploit_triple and sentence_no >= 1:
                    source_sentence = sentences[sentence_no - 1] + " <utterer_change> " + sentences[sentence_no]
                    f_s.write(source_sentence + "\n")
                    f_s.flush()
                    f_t.write(sentences[sentence_no + 1] + "\n")
                    f_t.flush()
            i += 1
        file.close()
    f_s.close()
    f_t.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="directory containing the datasets")
    parser.add_argument("--train_dir", type=str, default="train", 
                        help="directory to store the training set")
    parser.add_argument("--validation_dir", type=str, default="validation", 
                        help="directory to store the validation set")
    parser.add_argument("--vocab_size", type=int, default=20000,
                        help="size of the vocabulary")
    parser.add_argument("--transform_CMC", action="store_true", 
                        help="transform the Cornell Movie Corpus to match the MovieTriples dataset structure")
    parser.add_argument("--original_only", action="store_true", 
                        help="preprocess only the MovieTriples dataset")
    parser.add_argument("--exploit_triple", action="store_true", 
                        help="exploit the three-turn structure of the dataset, using the first turn when predicting the third")
    parser.add_argument("--run_test", action="store_true", 
                        help="preprocess only the MovieTriples dataset")
    args = parser.parse_args()
    

    if args.run_test:
        os.makedirs(args.train_dir, exist_ok=True)
        train_files = [args.data_dir]
        preprocess(args.train_dir, train_files, args.exploit_triple)
    else:
        os.makedirs(args.train_dir, exist_ok=True)
        os.makedirs(args.validation_dir, exist_ok=True)
        if args.transform_CMC:
            transform_CMC(args.data_dir)
        if args.original_only:
            train_files = [args.data_dir + "/Training_Shuffled_Dataset.txt"]
        else:
            train_files = [args.data_dir + "/Training_Shuffled_Dataset.txt", args.data_dir + "/CMC_preprocessed.txt"]
        val_files = [args.data_dir + "/Validation_Shuffled_Dataset.txt"]
        construct_vocabulary(args.train_dir, train_files, args.vocab_size)
        preprocess(args.train_dir, train_files, args.exploit_triple)
        preprocess(args.validation_dir, val_files, args.exploit_triple)
