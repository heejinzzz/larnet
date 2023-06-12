import os
import torch
import torch.utils.data.dataset
import torchaudio
import json
import csv
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from config import Config


def do_encode_texts(speakers, raw_texts, tokenizer, encode_text_len):
    encode_texts = []
    encode_sentences = []
    i = 0
    while i < len(speakers):
        sentence = f"{speakers[i]}: {raw_texts[i]}"
        encode_sentence = tokenizer(sentence, return_tensors="pt").input_ids[0].numpy().tolist()
        encode_sentences.append(encode_sentence)
        prompt = f"When saying \"{raw_texts[i]}\", {speakers[i]} feels {tokenizer.mask_token}"
        encode_text = tokenizer(prompt, return_tensors="pt").input_ids[0].numpy().tolist()
        j = len(encode_sentences) - 1
        while j >= 0:
            pre_encode_sentence = encode_sentences[j]
            if len(pre_encode_sentence) + 1 + len(encode_text) > encode_text_len:
                break
            encode_text = pre_encode_sentence + [tokenizer.sep_token_id] + encode_text
            j -= 1
        encode_text_padding = [tokenizer.pad_token_id] * (encode_text_len - len(encode_text))
        encode_text = encode_text_padding + encode_text
        encode_texts.append(encode_text)
        i += 1
    encode_texts = torch.LongTensor(encode_texts)
    return encode_texts


def do_encode_audio(audio_file, feature_extractor, max_len):
    audio, rate = torchaudio.load(audio_file)
    audio = torch.squeeze(audio, 0).numpy().tolist()
    if len(audio) > max_len:
        audio = audio[:max_len]
    encode_audio = feature_extractor(audio, sampling_rate=rate, return_tensors="pt").input_values
    encode_audio = torch.squeeze(encode_audio, 0).numpy().tolist()
    if len(encode_audio) <= max_len:
        encode_audio_padding = [feature_extractor.padding_value] * (max_len - len(encode_audio))
        encode_audio += encode_audio_padding
    return encode_audio


class MELD_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split, config: Config):
        self.emotion2id = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        self.sentiment2id = {'neutral': 0, 'positive': 1, 'negative': 2}
        self.modal = config.modal
        data_root_path = config.dataset_path + "/MELD"
        roberta_model_path = "princeton-nlp/sup-simcse-roberta-large"
        wav2vec_model_path = "Zahra99/wav2vec2-base-finetuned-iemocap6"
        self.text_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path, local_files_only=False)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=False)
        self.encode_text_len = config.encode_text_len
        self.encode_audio_len = config.encode_audio_len

        if split == "train":
            self.csv_file = data_root_path + "/train/train_sent_emo.csv"
            self.audios_path = data_root_path + "/train/train_audios"
        elif split == "dev":
            self.csv_file = data_root_path + "/dev/dev_sent_emo.csv"
            self.audios_path = data_root_path + "/dev/dev_audios"
        else:
            self.csv_file = data_root_path + "/test/test_sent_emo.csv"
            self.audios_path = data_root_path + "/test/test_audios"
        csv_reader = csv.DictReader(open(self.csv_file, encoding="utf-8"))
        self.dialogs = {}
        for uttr in csv_reader:
            audio_file = f"{self.audios_path}/dia{uttr['Dialogue_ID']}_utt{uttr['Utterance_ID']}.wav"
            if not os.path.exists(audio_file):
                print(f"Warning: File '{audio_file}' Missing")
                continue
            dia_id = int(uttr["Dialogue_ID"])
            if dia_id not in self.dialogs:
                self.dialogs[dia_id] = []
            speaker = uttr["Speaker"]
            raw_text = uttr['Utterance']
            emotion_id = self.emotion2id[uttr["Emotion"]]
            sentiment_id = self.sentiment2id[uttr["Sentiment"]]
            uttr_info = {
                "uttr_id": int(uttr["Utterance_ID"]),
                "speaker": speaker,
                "raw_text": raw_text,
                "audio_file": audio_file,
                "emotion_id": emotion_id,
                "sentiment_id": sentiment_id,
            }
            self.dialogs[dia_id].append(uttr_info)

        def uttr_sort_id(item):
            return item["uttr_id"]

        for dia_id in self.dialogs:
            self.dialogs[dia_id].sort(key=uttr_sort_id)
        self.dialog_list = list(self.dialogs.keys())

        self.encode_texts = []
        self.audio_files = []
        self.emotion_ids = []
        self.sentiment_ids = []
        for dia_id in self.dialog_list:
            dialog = self.dialogs[dia_id]
            while len(dialog) > 0:
                dialog_cut = dialog[:config.max_uttrs_num]
                speakers, raw_texts, audio_files, emotion_ids, sentiment_ids = [], [], [], [], []
                for info in dialog_cut:
                    speakers.append(info["speaker"])
                    raw_texts.append(info["raw_text"])
                    audio_files.append(info["audio_file"])
                    emotion_ids.append(info["emotion_id"])
                    sentiment_ids.append(info["sentiment_id"])
                encode_texts = do_encode_texts(speakers, raw_texts, self.text_tokenizer, config.encode_text_len)
                emotion_ids = torch.LongTensor(emotion_ids)
                sentiment_ids = torch.LongTensor(sentiment_ids)
                self.encode_texts.append(encode_texts)
                self.audio_files.append(audio_files)
                self.emotion_ids.append(emotion_ids)
                self.sentiment_ids.append(sentiment_ids)
                dialog = dialog[config.max_uttrs_num:]

    def __len__(self):
        return len(self.encode_texts)

    def __getitem__(self, index):
        encode_texts, audio_files, emotion_ids, sentiment_ids = self.encode_texts[index], self.audio_files[index], self.emotion_ids[index], self.sentiment_ids[index]
        if self.modal == "text":
            return encode_texts, torch.empty([]), emotion_ids, sentiment_ids
        encode_audios = []
        for audio_file in audio_files:
            encode_audio = do_encode_audio(audio_file, self.audio_processor, self.encode_audio_len)
            encode_audios.append(encode_audio)
        encode_audios = torch.Tensor(encode_audios)
        if self.modal == "audio":
            return torch.empty([]), encode_audios, emotion_ids, sentiment_ids
        return encode_texts, encode_audios, emotion_ids, sentiment_ids


class IEMOCAP_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split, config: Config):
        self.emotion2id = {'neutral': 0, 'frustration': 1, 'anger': 2, 'sadness': 3, 'happiness': 4, 'excited': 5}
        self.emotion2sentiment_id = {'neutral': 0, 'happiness': 1, 'excited': 1, 'frustration': 2, 'anger': 2, 'sadness': 2}
        self.modal = config.modal
        data_root_path = config.dataset_path + "/IEMOCAP"
        roberta_model_path = "princeton-nlp/sup-simcse-roberta-large"
        wav2vec_model_path = "Zahra99/wav2vec2-base-finetuned-iemocap6"
        self.text_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path, local_files_only=False)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=False)
        self.encode_audio_len = config.encode_audio_len
        order_data = json.load(open(data_root_path + "/utterances.json"))
        audios_path = data_root_path + "/IEMOCAP_wav"
        texts_path = data_root_path + "/IEMOCAP_text"
        if split == "train":
            order = order_data["train"]
        elif split == "dev":
            order = order_data["val"]
        else:
            order = order_data["test"]
        self.encode_texts = []
        self.audio_files = []
        self.emotion_ids = []
        self.sentiment_ids = []
        for dialog_id in order:
            dialog = order[dialog_id]
            while len(dialog) > 0:
                dialog_cut = dialog[:config.max_uttrs_num]
                speakers, raw_texts, audio_files, emotion_ids, sentiment_ids = [], [], [], [], []
                for uttr_id in dialog_cut:
                    uttr_json_file = f"{texts_path}/{uttr_id}.json"
                    uttr = json.load(open(uttr_json_file))
                    speakers.append(uttr["Speaker"])
                    raw_texts.append(uttr["Utterance"])
                    audio_file = f"{audios_path}/{uttr_id}.wav"
                    audio_files.append(audio_file)
                    emotion = uttr["Emotion"]
                    if emotion in self.emotion2id:
                        emotion_ids.append(self.emotion2id[emotion])
                        sentiment_ids.append(self.emotion2sentiment_id[emotion])
                    else:
                        emotion_ids.append(-1)
                        sentiment_ids.append(-1)
                encode_texts = do_encode_texts(speakers, raw_texts, self.text_tokenizer, config.encode_text_len)
                emotion_ids = torch.LongTensor(emotion_ids)
                sentiment_ids = torch.LongTensor(sentiment_ids)
                self.encode_texts.append(encode_texts)
                self.audio_files.append(audio_files)
                self.emotion_ids.append(emotion_ids)
                self.sentiment_ids.append(sentiment_ids)
                dialog = dialog[config.max_uttrs_num:]

    def __len__(self):
        return len(self.encode_texts)

    def __getitem__(self, index):
        encode_texts, audio_files, emotion_ids, sentiment_ids = self.encode_texts[index], self.audio_files[index], self.emotion_ids[index], self.sentiment_ids[index]
        if self.modal == "text":
            return encode_texts, torch.empty([]), emotion_ids, sentiment_ids
        encode_audios = []
        for audio_file in audio_files:
            encode_audio = do_encode_audio(audio_file, self.audio_processor, self.encode_audio_len)
            encode_audios.append(encode_audio)
        encode_audios = torch.Tensor(encode_audios)
        if self.modal == "audio":
            return torch.empty([]), encode_audios, emotion_ids, sentiment_ids
        return encode_texts, encode_audios, emotion_ids, sentiment_ids
