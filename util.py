from transformers import ViltProcessor, ViltForImagesAndTextClassification, AutoImageProcessor, ViTModel
import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import json
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import torch
import numpy as np



class NLVR2Dataset(Dataset):
    def __init__(self, path, model_type, image_path = None):
        self.examples = []
        # self.image_path = "/data/mrigankr/mml/dev/"
        self.image_path = image_path
        if model_type == "vilt":
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
        elif model_type == "roberta":
            self.processor = AutoTokenizer.from_pretrained("roberta-large")
        elif model_type == "vit":
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.model_type = model_type
        with open(path) as infile:
            for line in infile:
            # cnt+=1
                example = json.loads(line)
                self.examples.append(example)

        

    def get_image(self, name):
        image = Image.open(self.image_path+name)
        image = image.convert("RGB")
        return image

    # def processor()

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):
        if self.model_type == "vilt":
            image_1 =self.get_image(self.examples[index]['identifier'][:-2]+"-img0.png")
            image_2 =self.get_image(self.examples[index]['identifier'][:-2]+"-img1.png")
            text = self.examples[index]['sentence']
            label = self.examples[index]['label']
            encoding = self.processor([image_1, image_2], text, return_tensors="pt", padding="max_length", max_length=40, truncation=True)
            input_ids = encoding.input_ids.squeeze(0)
            pixel_values = encoding.pixel_values
            if(label=="True"):
                label=1
            else:
                label=0
            return input_ids, pixel_values, label
        elif self.model_type == "roberta":
            text = self.examples[index]['sentence']
            label = self.examples[index]['label']
            inputs_dict = self.processor(text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict["attention_mask"]
            # import ipdb
            # ipdb.set_trace()
            if(label=="True"):
                label=1
            else:
                label=0
            return {"input_ids": torch.tensor(input_ids.squeeze(0)), "attention_mask":  torch.tensor(attention_mask.squeeze(0)), "labels": torch.tensor(label)}
        elif self.model_type == "vit":
            image_1 =self.get_image(self.examples[index]['identifier'][:-2]+"-img0.png")
            image_2 =self.get_image(self.examples[index]['identifier'][:-2]+"-img1.png")
            pixel_values_1 = self.processor(image_1, return_tensors="pt")
            pixel_values_2 = self.processor(image_2, return_tensors="pt")
            label = self.examples[index]['label']
            if(label=="True"):
                label=1
            else:
                label=0
            return pixel_values_1["pixel_values"].squeeze(0), pixel_values_2["pixel_values"].squeeze(0), label
        else:
            image_1 =self.get_image(self.examples[index]['identifier'][:-2]+"-img0.png")
            image_2 =self.get_image(self.examples[index]['identifier'][:-2]+"-img1.png")

            pixel_values_1 = self.processor(image_1, return_tensors="pt")
            pixel_values_2 = self.processor(image_2, return_tensors="pt")
            label = self.examples[index]['label']
            if(label=="True"):
                label=1
            else:
                label=0
            text = self.examples[index]['sentence']
            inputs_dict = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict["attention_mask"]
            return pixel_values_1["pixel_values"].squeeze(0), pixel_values_2["pixel_values"].squeeze(0), label


        


# eval_data = NLVR2Dataset('/home/mrigankr/PGM/nlvr2/data/dev.json', "roberta")
# train_data = NLVR2Dataset('/home/mrigankr/PGM/nlvr2/data/train.json', "roberta")
eval_data = NLVR2Dataset('/home/mrigankr/PGM/nlvr2/data/dev.json', "vit", "/data/mrigankr/mml/dev/")
train_data = NLVR2Dataset('/home/mrigankr/PGM/nlvr2/data/train.json', "vit", "/data/mrigankr/mml/train/")
# model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
class ViTforNLVR2(nn.Module):
    def __init__(self):
        super(ViTforNLVR2, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Linear(768*2, 2)
    def forward(self, pixel_values_1, pixel_values_2):
        # import ipdb
        # ipdb.set_trace()
        x1 = self.vit(pixel_values = pixel_values_1).pooler_output
        x2 = self.vit(pixel_values = pixel_values_2).pooler_output
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x
    
class ViTRobforNLVR2(nn.Module):
    def __init__(self):
        super(ViTforNLVR2, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.rob = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
        self.classifier = nn.Linear(768*3, 2)
    def forward(self, pixel_values_1, pixel_values_2, input_ids, attention_mask):
        # import ipdb
        # ipdb.set_trace()
        x1 = self.vit(pixel_values = pixel_values_1).pooler_output
        x2 = self.vit(pixel_values = pixel_values_2).pooler_output
        x3 = self.rob(input_ids = input_ids, attention_mask = attention_mask).hidden_states[-1][:,0,:]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)
        return x

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

model = ViTforNLVR2()
model.to("cuda")
epochs = 20
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_data, batch_size=32, shuffle=True)
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader):
        pixel_values_1, pixel_values_2, labels = batch
        pixel_values_1, pixel_values_2, labels = pixel_values_1.to("cuda"), pixel_values_2.to("cuda"), labels.to("cuda")
        outputs = model(pixel_values_1, pixel_values_2)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    model.eval()
    correct = 0
    for batch in eval_loader:
        pixel_values_1, pixel_values_2, labels = batch
        pixel_values_1, pixel_values_2, labels = pixel_values_1.to("cuda"), pixel_values_2.to("cuda"), labels.to("cuda")
        outputs = model(pixel_values_1, pixel_values_2)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
    print("Epoch: {}, Accuracy: {}".format(epoch, correct/len(eval_data)))
        
    


# train_args = TrainingArguments(
#     output_dir="/home/mrigankr/PGM/",
#     num_train_epochs=10,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_steps=1000,
#     save_steps=1000,
#     evaluation_strategy="steps",
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     greater_is_better=True,
#     learning_rate=1e-5,
#     do_train=True,
#     do_eval=True,
#     do_predict=True,
#     save_total_limit=1,
#     seed=42,
# )
# trainer = Trainer(
#     model=model,
#     args=train_args,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     compute_metrics=compute_metrics,
# )
# trainer.train()




# for i in range(len(data)):
#     print(data[i][0].shape)


# loader = DataLoader(data, batch_size=1, shuffle=True)
# model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
# model.load_state_dict(torch.load("/home/mrigankr/PGM/vilt_nlvr2.ckpt"))
# model = model.cuda()
# correct = 0
# size = len(data)
# i=0
# for i, batch in (enumerate(tqdm(loader))):
#     input_ids, pixel_values, label = batch
#     input_ids, pixel_values, label = input_ids.cuda(), pixel_values.cuda(), label.cuda()
#     outputs = model(input_ids=input_ids, pixel_values=pixel_values)
#     logits = outputs.logits
#     idx = logits.argmax(-1)
#     correct = correct+ (idx==label).sum()

# print("Accuracy:" , (correct/size)*100)

    # input_ids = batch[0][:,50].reshape([batch_size,50])
    # pixel_values = batch[0][50:].reshape([batch_size, 2, 3, 384, 384])
    # if(i%100==0):
    #     print(i, batch[1])
    # if(i>10):
    #     break
    # print(batch[0], batch[1])


# input_ids, pixel_values, label =data[1]        
# print(input_ids.shape, pixel_values.shape, label)


# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
# model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

# # # prepare inputs
# encoding = processor([image1, image2], text, return_tensors="pt")
# print(encoding)
# print(processor.tokenizer(text))
# import torch
# forward pass
# outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
# pixel_values = torch.zeros([3,2,3,384,384]).type(torch.LongTensor)
# outputs = model(input_ids=torch.zeros([3,21]).type(torch.LongTensor), pixel_values = torch.zeros([3,2,3,384,384]))
# logits = outputs.logits
# idx = logits.argmax(-1).item()
# print("Predicted answer:", model.config.id2label[idx])



# class NLVR2Dataset(BaseDataset):
#     def __init__(self, path):
        

#     def __getitem__(self, index):
#         result = None
#         while result is None:
#             try:
#                 image_tensor_0 = self.get_image(index, image_key="image_0")["image"]
#                 image_tensor_1 = self.get_image(index, image_key="image_1")["image"]
#                 text = self.get_text(index)["text"]
#                 result = True
#             except:
#                 print(
#                     f"error while read file idx {index} in {self.names[0]}",
#                     file=sys.stderr,
#                 )
#                 index = random.randint(0, len(self.index_mapper) - 1)

#         index, question_index = self.index_mapper[index]
#         answers = self.table["answers"][index][question_index].as_py()
#         answers = answers == "True"

#         return {
#             "image_0": image_tensor_0,
#             "image_1": image_tensor_1,
#             "text": text,
#             "answers": answers,
#             "table_name": self.table_names[index],
#         }


