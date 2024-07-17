import nnet
import os

# Params
workers_prepare = -1 # Set to -1 for nproc
mean_face_path = "media/20words_mean_face.npy"
tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"


# lrs2_username = "lrs2008" # Set to your lrs2 username
# lrs2_password = "eT7uonoh" # Set to your lrs2 password
# os.environ["LRS2_USERNAME"] = lrs2_username
# os.environ["LRS2_PASSWORD"] = lrs2_password

print("Download and Prepare EGO4D")
datasets = nnet.get_datasets()
ego4d_dataset = nnet.datasets.EGO4D(None, None, version="EGO4D", download=True, prepare=True, tokenizer_path=tokenizer_path, mean_face_path=mean_face_path, workers_prepare=2, mode="train")


print("Create Corpora")
ego4d_dataset.create_corpus(mode="train")
ego4d_dataset.create_corpus(mode="val")
ego4d_dataset.create_corpus(mode="test")

filenames = ["datasets/EGO4D/corpus_pretrain.txt", "datasets/EGO4D/corpus_train.txt", "datasets/EGO4D/corpus_val.txt", "datasets/EGO4D/corpus_test.txt", "datasets/EGO4D/corpus_trainval.txt"]
with open("datasets/EGO4D/corpus_ego4d_train+val.txt", "w") as fw:
    for filename in filenames:
        try:
            with open(filename, "r") as fr:
                for line in fr.readlines():
                    fw.write(line)
        except FileNotFoundError:
            print(f"File {filename} not found, skipping.")
