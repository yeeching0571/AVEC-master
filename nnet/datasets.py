# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
import torchvision
import torchaudio
from torchvision.datasets.utils import extract_archive

# Other
import os
import glob
from tqdm import tqdm
import sentencepiece as spm
import numpy as np
import requests
import pickle
import gdown
import multiprocessing
import zipfile
import time
from google.colab import drive
import psutil
import ipdb;

# NeuralNets
from nnet import layers
from nnet import transforms
from nnet import collate_fn
from nnet import transforms
from nnet import collate_fn

###############################################################################
# Datasets
###############################################################################

class Dataset(torch.utils.data.Dataset):

    def __init__(self, batch_size=16, collate_fn=collate_fn.Collate(), root="datasets", shuffle=True):
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.root = root
        self.shuffle = shuffle

class MultiDataset(Dataset):

    def __init__(self, batch_size, collate_fn, datasets, shuffle=True):
        super(MultiDataset, self).__init__(batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=None)

        self.datasets = datasets

    def __len__(self):

        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, n):

        ctr = 0
        for dataset in self.datasets:
            ctr_prev = ctr
            ctr += len(dataset)
            if n < ctr:
                return dataset.__getitem__(n - ctr_prev)
            
class EGO4D(Dataset):

    def __init__(self, batch_size, collate_fn, version="EGO4D", img_mean=(0.5,), img_std=(0.5,), crop_mouth=True, root="datasets", shuffle=True, ascending=False, mode="pretrain+train+val", load_audio=True, load_video=True, video_transform=None, audio_transform=None, download=False, prepare=False, workers_prepare=2, video_max_length=None, audio_max_length=None, label_max_length=None, tokenizer_path="datasets/LRS3/tokenizerbpe256.model", mean_face_path="media/20words_mean_face.npy", align=False):
        super(EGO4D, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle and not ascending)

        assert version in ["EGO4D", "LRS2", "LRS3"]

        # Params
        self.version = version
        self.mode = mode
        self.ascending = ascending
        self.load_audio = load_audio
        self.load_video = load_video
        self.video_max_length = video_max_length
        self.audio_max_length = audio_max_length
        self.label_max_length = label_max_length
        self.workers_prepare = multiprocessing.cpu_count() if workers_prepare==-1 else workers_prepare
        self.tokenizer_path = tokenizer_path
        self.crop_mouth = crop_mouth
        self.mean_face_path = mean_face_path
        self.align = align

        # Download Dataset
        if download:
            self.download()

        # Prepare Dataset
        if prepare:
            self.prepare()

        # EGO4D
        if version == "EGO4D":

            # Mode
            assert mode in ["train+val", "pretrain+train", "pretrain", "train", "val", "test"]

            # Paths
            self.paths = []
            if "pretrain" in mode:
                with open(os.path.join(root, "EGO4D", "pretrain.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "EGO4D", "pretrain_set", line.replace("\n", "")))
            if "train" in mode:
                with open(os.path.join(root, "EGO4D", "train.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "EGO4D", "train_set", line.replace("\n", "")))
            if "val" in mode:
                with open(os.path.join(root, "EGO4D", "val.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "EGO4D", "val_set", line.replace("\n", "")))
            if "test" in mode:
                with open(os.path.join(root, "EGO4D", "test.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "EGO4D", "test_set", line.split()[0]))

        # Video Transforms
        self.video_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            layers.Permute(dims=(1, 0, 2, 3)),
            torchvision.transforms.Grayscale(),
            layers.Permute(dims=(1, 0, 2, 3)),
            transforms.NormalizeVideo(mean=img_mean, std=img_std),
            video_transform if video_transform != None else nn.Identity()
        ])

        # Audio Transforms
        self.audio_preprocessing = audio_transform if audio_transform != None else nn.Identity() 

        # Filter Dataset
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.filter()
                n_elt = [len(self.paths)]
            else:
                n_elt = [None]

            # Broadcast number of elements
            torch.distributed.barrier()
            torch.distributed.broadcast_object_list(n_elt, src=0)

            # Broadcast path list
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0:
                self.paths = [None for _ in range(n_elt[0])]
            torch.distributed.broadcast_object_list(self.paths, src=0)
        else:
            self.filter()

    def create_corpus(self, mode):

        print("Enter create_corpus")
        corpus_path = os.path.join(self.root, self.version, "corpus_{}.txt".format(mode))
        #ipdb.set_trace()
        if not os.path.isfile(corpus_path):

            print("Create Corpus File: {} {}".format(self.version, mode))
            corpus_file = open(corpus_path, "w")

            # EGO4D
            #ipdb.set_trace()
            if self.version == "EGO4D":
                if "train" == mode:
                    with open(os.path.join(self.root, "EGO4D", "train.txt")) as f:
                        for line in tqdm(f.readlines()):
                            with open(os.path.join(self.root, "EGO4D", "train_set", line.replace("\n", "") + ".txt"), "r") as f:
                                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                                # ipdb.set_trace()
                                corpus_file.write(line)

                if "val" == mode:
                    with open(os.path.join(self.root, "EGO4D", "val.txt")) as f:
                        for line in tqdm(f.readlines()):
                            with open(os.path.join(self.root, "EGO4D", "val_set", line.replace("\n", "") + ".txt"), "r") as f:
                                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                                corpus_file.write(line)

                if "test" == mode:
                    with open(os.path.join(self.root, "EGO4D", "test.txt")) as f:
                        for line in tqdm(f.readlines()):
                            with open(os.path.join(self.root, "EGO4D", "test_set", line.split()[0] + ".txt"), "r") as f:
                                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                                corpus_file.write(line)

                # LRS3
                elif self.version == "LRS3":

                    for file_path in tqdm(glob.glob(os.path.join(self.root, "LRS3", mode, "*", "*.txt"))):
                        with open(file_path, "r") as f:
                            line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                            corpus_file.write(line)

            
    class FilterDataset:

        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            file_path = self.paths[idx]
            return file_path, torch.load(file_path + ".pt")
      
    def filter(self):

        if self.video_max_length==None and self.audio_max_length==None and self.label_max_length==None:
            return
        else:
            video_max_length = self.video_max_length if self.video_max_length != None else float("inf")
            audio_max_length = self.audio_max_length if self.audio_max_length != None else float("inf")
            label_max_length = self.label_max_length if self.label_max_length != None else float("inf")
            print("Dataset Filtering")
            print("Video maximum length : {} / Audio maximum length : {} / Label sequence maximum length : {}".format(video_max_length, audio_max_length, label_max_length))
            
            filename = os.path.join(self.root, self.version, "mode_{}_video_max_length_{}_audio_max_length_{}_label_max_length_{}_paths.pt".format(self.mode, video_max_length, audio_max_length, label_max_length))
            if not os.path.isfile(filename):

                # Create Dataloader
                dataloader = torch.utils.data.DataLoader(
                    self.FilterDataset(self.paths),
                    batch_size=16,
                    num_workers=self.workers_prepare,
                    collate_fn=collate_fn.Collate(),
                )

                # filter
                paths = []
                lengths = []
                for batch in tqdm(dataloader):
                    path, infos = batch[0]
                    if infos["video_len"] <= video_max_length and infos["audio_len"] <= audio_max_length and infos["label_len"] <= label_max_length:
                        paths.append(path)
                        lengths.append(infos["audio_len"])
                    
                self.paths = paths
                torch.save(self.paths, filename)

            else:
                self.paths = torch.load(filename)

            # sort_by_length
            if self.ascending:
                paths = [elt[1] for elt in sorted(zip(lengths, paths))]

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, n):

        # Load Video
        if self.load_video:
            if self.crop_mouth:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + "_mouth.mp4")
            else:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + ".mp4")
        else:
            video = None

        # Load Audio
        if self.load_audio:
            audio = torchaudio.load(self.paths[n] + ".flac")[0]
        else:
            audio = None

        # Load Infos
        infos = torch.load(self.paths[n] + ".pt")
        label, video_len, audio_len, label_len = infos["label"], infos["video_len"], infos["audio_len"], infos["label_len"]

        # Audio Preprocessing
        if self.load_audio:
            audio = self.audio_preprocessing(audio[:1]).squeeze(dim=0)

        # Video Preprocessing
        if self.load_video: 
            video = video.permute(3, 0, 1, 2)
            video = self.video_preprocessing(video)
            if self.align:
                video = transforms.align_video_to_audio(video.permute(1, 2, 3, 0), audio)
                video_len = video.shape[0]
            else:
                video = video.permute(1, 2, 3, 0)

        # Infos Preprocessing
        video_len = torch.tensor(video_len, dtype=torch.long)
        audio_len = torch.tensor(audio_len, dtype=torch.long)
        label_len = torch.tensor(label_len, dtype=torch.long)

        return video, audio, label, video_len, audio_len, label_len



    def download_file(self, url, path):
            with requests.get(url, stream=True) as r:
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)

    def download_ego4d(self):
        # file_id = "1UUeskiOoMS-5nn4PEZQeWiZWC9Vfhdfk"
        # base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        root_path = os.path.join(self.root, "EGO4D", "train_set")
    
        if not os.path.exists(root_path):
            os.makedirs(root_path)

         # 下载文件。如果文件已存在，则不下载。设置标志文件，判断是否已解压。
        file_name = "train_set1.zip"
        file_path = os.path.join(root_path, file_name)
        extract_flag = os.path.join(root_path, "train_set1_extracted.flag")
        
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            gdown.download(base_url, file_path, quiet=False)
        else:
            print(f"File {file_path} already exists, skipping download.")
        
        if not os.path.exists(extract_flag):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(root_path)
                print(f"Extracted {file_path} successfully.")
                with open(extract_flag, 'w') as f:
                    f.write('Extraction completed')
            except Exception as e:
                print(f"Failed to extract {file_path}: {e}")
        else:
            print(f"File {file_path} already extracted, skipping extraction.")
            
        # 定义 Google Drive 文件的 URL 和对应的文件名
        google_drive_files = [
            ("15dqRtGiFlqVXMHCFCZZesc2FiGdDRauz", "test.txt"),
            ("1I6_Z5r6TMVXbMsXjXZsWjFmzuULDCvSc", "train.txt"),
            ("1xeZt3rpLubtNhIDLlpXDA7xjt8VCZev5", "val.txt"),
        ]

        # 下载每个文件
        for file_id, filename in google_drive_files:
            file_path = os.path.join(self.root,"EGO4D", filename)
            print(f"Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)

        # Download Landmarks from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
        # landmarks_url = "https://drive.google.com/uc?id=1G2-rEUNeGotJ9EtTIj0UzqbvCSbn6CJy"
        # landmarks_path = os.path.join(self.root, "EGO4D", "LRS2_landmarks.zip")
        # extract_to_path = os.path.join(self.root, "EGO4D")
        # check_file_path = os.path.join(extract_to_path, "path_to_check_file")

    
        # if not os.path.exists(landmarks_path):
        #     print(f"Downloading LRS2_landmarks.zip from Google Drive...")
        #     try:
        #         gdown.download(landmarks_url, landmarks_path, quiet=False)
        #     except Exception as e:
        #         print(f"Failed to download LRS2_landmarks.zip: {e}")
        # else:
        #     print(f"File {landmarks_path} already exists, skipping download.")

      
        # if not os.path.exists(check_file_path):
        #     print(f"Extracting LRS2_landmarks.zip to {extract_to_path}...")
        #     try:
        #         extract_archive(
        #             from_path=landmarks_path,
        #             to_path=extract_to_path
        #         )
        #     except Exception as e:
        #         print(f"Failed to extract LRS2_landmarks.zip: {e}")
        # else:
        #     print(f"File {check_file_path} already exists, skipping extraction.")




    def download(self):

        # Print
        print("Download Dataset")
        os.makedirs(os.path.join(self.root, self.version), exist_ok=True)

        # EGO4D
        if self.version == "EGO4D":
            self.download_ego4d()
        
        # LRS3
        elif self.version == "LRS3":
            self.download_lrs3()

    # download LSR2/LRS3 files
    # def download_file(self, url, path):

    #     # Download, Open and Write
    #     with requests.get(url, auth=(os.getenv("{}_USERNAME".format(self.version)), os.getenv("{}_PASSWORD".format(self.version))), stream=True) as r:
    #         with open(path, 'wb') as f:
    #             for chunk in r.iter_content(chunk_size=1024):
    #                 f.write(chunk)

    class PrepareDataset:

        def __init__(self, paths, tokenizer, mean_face_path, version):
            self.paths = paths
            self.tokenizer = tokenizer
            self.lip_crop = transforms.LipDetectCrop(mean_face_landmarks_path=mean_face_path)
            self.version = version

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):

            file_path = self.paths[idx]

            # Read and Encode
            with open(file_path, "r") as f:
                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower().replace("\n", "")
                label = torch.LongTensor(self.tokenizer.encode(line))

            # Load Video
            video, audio, info = torchvision.io.read_video(file_path.replace(".txt", ".mp4"))

            # Save Audio
            torchaudio.save(file_path.replace(".txt", ".flac"), audio, sample_rate=16000)

            # Extract Landmarks  File datasets/EGO4D/LRS2_landmarks.zip
            if self.version == "EGO4D":
                landmarks_pathname = file_path.replace(".txt", ".pkl").replace("train_set", "ego4d_landmarks")
            elif self.version == "LRS3":
                landmarks_pathname = file_path.replace(".txt", ".pkl").replace("LRS3", "LRS3/LRS3_landmarks")
            with open(landmarks_pathname, "br") as f:
                landmarks = pickle.load(f)

            # Interpolate Landmarks
            preprocessed_landmarks = self.lip_crop.landmarks_interpolate(landmarks)

            # Crop
            if not preprocessed_landmarks:
                video = torchvision.transforms.functional.resize(video.permute(3, 0, 1, 2), size=(self.lip_crop.crop_height, self.lip_crop.crop_width)).permute(1, 2, 3, 0)
            else:
                video = self.lip_crop.crop_patch(video.numpy(), preprocessed_landmarks)
                assert video is not None
                video = torch.tensor(video)
         
            # Save Video
            torchvision.io.write_video(filename=file_path.replace(".txt", "_mouth.mp4"), video_array=video, fps=info["video_fps"], video_codec="libx264")

            # Save Infos
            infos = {"label": label, "video_len": video.shape[0], "audio_len": audio.shape[1], "label_len": label.shape[0]}
            torch.save(infos, file_path.replace(".txt", ".pt"))
            
            return file_path, infos
        
 
    def prepare(self):

        # Remove from corpus
        # {NS} ~ non scripted
        # {LG} ~ Laughter

        if self.version == "EGO4D":
            paths_txt = glob.glob(os.path.join(self.root, "EGO4D", "*", "*", "*.txt"))
        elif self.version == "LRS3":
            paths_txt = glob.glob(os.path.join(self.root, "LRS3", "*", "*", "*.txt"))

        # Create Corpus File
        corpus_path = os.path.join(self.root, self.version, "corpus.txt")
        if not os.path.isfile(corpus_path):
            print("Create Corpus File")
            corpus_file = open(corpus_path, "w")
            for file_path in tqdm(paths_txt):
                with open(file_path, "r") as f:
                    line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                    corpus_file.write(line)

        # Load Tokenizer
        tokenizer = spm.SentencePieceProcessor(self.tokenizer_path)

        # Prepare
        print("Prepare Dataset")
        dataloader = torch.utils.data.DataLoader(
            self.PrepareDataset(
                paths=paths_txt,
                tokenizer=tokenizer,
                mean_face_path=self.mean_face_path,
                version=self.version
            ),
            batch_size=16,
            num_workers=self.workers_prepare,
            collate_fn=collate_fn.Collate(),
        )

        try:
            for batch in tqdm(dataloader):
                # 处理数据
                print_memory_usage()  # 打印内存使用情况
                pass
        except RuntimeError as e:
            print(f"RuntimeError: {e}")

class CorpusLM(Dataset):

    def __init__(self, batch_size, collate_fn, root="datasets", shuffle=True, download=False, tokenizer_path="datasets/LRS3/tokenizerbpe1024.model", max_length=None, corpus_path="datasets/LibriSpeechCorpus/librispeech-lm-norm.txt"):
        super(CorpusLM, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle)

        # Params
        self.root = root
        self.max_len = max_length

        if isinstance(tokenizer_path, str):
            self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        else:
            self.tokenizer = tokenizer_path
        self.corpus = open(corpus_path, 'r').readlines()

    def __getitem__(self, i):

        if self.max_len:
            while len(self.tokenizer.encode(self.corpus[i].replace("\n", "").lower())) > self.max_len:
                i = torch.randint(0, self.__len__(), [])

        label = torch.LongTensor(self.tokenizer.encode(self.corpus[i].replace("\n", "").lower()))

        return label,

    def __len__(self):
        return len(self.corpus)


    """ Lip Reading in the Wild (LRW) Dataset : https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

    The dataset consists of up to 1000 utterances of 500 different words, spoken by hundreds of different speakers. 
    All videos are 29 frames (1.16 seconds) in length, and the word occurs in the middle of the video.

    Infos:
        488,766 train samples
        25,000 val samples
        test samples
        (29, 256, 256, 3) videos
        (1, 19456) audios
    
    """

    def __init__(self, batch_size, collate_fn, root="datasets", shuffle=True, mode="train", img_mean=(0.5,), img_std=(0.5,), crop_mouth=True, load_audio=True, load_video=True, video_transform=None, download=False, prepare=False, mean_face_path="media/20words_mean_face.npy", workers_prepare=-1):
        super(LRW, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle)

        # Params
        self.workers_prepare = multiprocessing.cpu_count() if workers_prepare==-1 else workers_prepare
        self.crop_mouth = crop_mouth
        self.mean_face_path = mean_face_path
        self.load_audio = load_audio
        self.load_video = load_video

        # Download Dataset
        if download:
            self.download()

        # Prepare Dataset
        if prepare:
            self.prepare()

        # Mode
        assert mode in ["train", "val", "test"]

        # Class Dict
        self.class_dict = {}
        for i, path in enumerate(sorted(glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*")))):
            c = path.split("/")[-1]
            self.class_dict[i] = c
            self.class_dict[c] = i

        # Paths
        self.paths = glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*", mode, "*[0-9].mp4"))
        for i, path in enumerate(self.paths):
                self.paths[i] = path[:-4]

        # Video Transforms
        self.video_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            layers.Permute(dims=(1, 0, 2, 3)),
            torchvision.transforms.Grayscale(),
            layers.Permute(dims=(1, 0, 2, 3)),
            transforms.NormalizeVideo(mean=img_mean, std=img_std),
            video_transform if video_transform != None else nn.Identity()
        ])

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, n):

        # Load Video
        if self.load_video:
            if self.crop_mouth:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + "_mouth.mp4")
            else:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + ".mp4")
        else:
            video = None

        # Load Audio
        if self.load_audio:
            audio = torchaudio.load(self.paths[n] + ".flac")[0]
        else:
            audio = None

        # Label
        c = self.paths[n].split("/")[-1].split("_")[0]
        label = self.class_dict[c]

        # Preprocessing
        video = self.video_preprocessing(video.permute(3, 0, 1, 2))
        audio = audio.squeeze(dim=0)
        label = torch.tensor(label)

        return video, audio, label

    class PrepareDataset:

        def __init__(self, paths, mean_face_path):
            self.paths = paths
            self.lip_crop = transforms.LipDetectCrop(mean_face_landmarks_path=mean_face_path)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):

            file_path = self.paths[idx]

            # Load Video
            video, audio , info = torchvision.io.read_video(file_path.replace(".txt", ".mp4"))

            # Save Audio
            torchaudio.save(file_path.replace(".txt", ".flac"), audio, sample_rate=16000)

            # Extract Landmarks
            landmarks_pathname = file_path.replace(".txt", ".npz").replace("lipread_mp4", "LRW_landmarks")
            person_id = 0
            multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
            landmarks = [None] * len(multi_sub_landmarks)
            for frame_idx in range(len(landmarks)):
                try:
                    landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]['facial_landmarks']
                except IndexError:
                    continue

            # Interpolate Landmarks
            preprocessed_landmarks = self.lip_crop.landmarks_interpolate(landmarks)

            # Crop
            if not preprocessed_landmarks:
                video = torchvision.transforms.functional.resize(video.permute(3, 0, 1, 2), size=(self.lip_crop.crop_height, self.lip_crop.crop_width)).permute(1, 2, 3, 0)
            else:
                video = self.lip_crop.crop_patch(video.numpy(), preprocessed_landmarks)
                assert video is not None
                video = torch.tensor(video)
         
            # Save Video
            torchvision.io.write_video(filename=file_path.replace(".txt", "_mouth.mp4"), video_array=video, fps=info["video_fps"], video_codec="libx264")
            
            return file_path

    def prepare(self):

        # Prepare
        print("Prepare Dataset")
        dataloader = torch.utils.data.DataLoader(
            self.PrepareDataset(
                paths=glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*", "*", "*.txt")),
                mean_face_path=self.mean_face_path
            ),
            batch_size=16,
            num_workers=self.workers_prepare,
            collate_fn=collate_fn.Collate(),
        )
        for batch in tqdm(dataloader):
            pass

    def download(self):

        # Print
        print("Download dataset")
        os.makedirs(os.path.join(self.root, "LRW"), exist_ok=True)

        # Download Pretrain
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaa",
            path=os.path.join(self.root, "LRW", "lrw-v1-partaa")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partab",
            path=os.path.join(self.root, "LRW", "lrw-v1-partab")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partac",
            path=os.path.join(self.root, "LRW", "lrw-v1-partac")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partad",
            path=os.path.join(self.root, "LRW", "lrw-v1-partad")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partae",
            path=os.path.join(self.root, "LRW", "lrw-v1-partae")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaf",
            path=os.path.join(self.root, "LRW", "lrw-v1-partaf")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partag",
            path=os.path.join(self.root, "LRW", "lrw-v1-partag")
        )
        os.system("cat " + os.path.join(self.root, "LRW", "lrw-v1*") + " > " +  os.path.join(self.root, "LRW", "lrw-v1.tar"))
        extract_archive(
            from_path=os.path.join(self.root, "LRW", "lrw-v1.tar"),
            to_path=os.path.join(self.root, "LRW")
        )   

        # Download Landmarks from https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
        gdown.download("https://drive.google.com/uc?id=12mHlNQKCE2AXkFHzvRyqSbsmOMEs259i", os.path.join(self.root, "LRW", "LRW_landmarks.zip"), quiet=False)
        extract_archive(
            from_path=os.path.join(self.root, "LRW", "LRW_landmarks.zip"),
            to_path=os.path.join(self.root, "LRW")
        )   

    def download_file(self, url, path):

        # Download, Open and Write
        with requests.get(url, auth=(os.getenv("LRW_USERNAME"), os.getenv("LRW_PASSWORD")), stream=True) as r:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB") 
    
