import argparse
import string
import json
from types import SimpleNamespace
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

class TextReader(object):

    opts = SimpleNamespace()
    trmodel = None
    device = None
    convertor = None

    def __init__(self, args):
        
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        
        self.opts.workers = args.get("workers",5) # number of data loading workers
        ## data processing args
        self.opts.batch_size = args.get('batch_size',50) #maximum-label-length
        self.opts.batch_max_length = args.get('batch_max_length',25) #maximum-label-length
        self.opts.imgH = args.get('imgH',32)  # the height of the input image
        self.opts.imgW = args.get('imgW',100) # #the width of the input image
        self.opts.rgb = args.get('rgb',False) # use rgb input
        self.opts.character = args.get('character', '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        #self.opts.character = args.get('character','0123456789abcdefghijklmnopqrstuvwxyz') #character label
        self.opts.sensitive = args.get('sensitive',True) #for sensitive character mode
        self.opts.PAD = args.get('PAD',True) #whether to keep ratio then pad for image resize
        ## Model architecture
        self.opts.Transformation = args.get('Transformation','TPS') #Transformation stage. None|TPS
        self.opts.FeatureExtraction = args.get('FeatureExtraction','ResNet') #FeatureExtraction stage. VGG|RCNN|ResNet
        self.opts.SequenceModeling = args.get('SequenceModeling','BiLSTM') #SequenceModeling stage. None|BiLSTM
        self.opts.Prediction = args.get('Prediction','Attn') #Prediction stage. CTC|Attn
        self.opts.num_fiducial = args.get('num_fiducial',20) #number of fiducial points of TPS-STN
        self.opts.input_channel = args.get('input_channel',1)  #the number of input channel of Feature extractor
        self.opts.output_channel = args.get('output_channel',512) #the number of output channel of Feature extractor
        self.opts.hidden_size = args.get('hidden_size',256) #the size of the LSTM hidden state
        self.opts.num_gpu = 0 if torch.cuda.is_available() else torch.cuda.device_count()
        self.opts.saved_model = args.get('saved_model',dir_path+"/models/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth")

        if self.opts.sensitive:
            self.opts.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        if 'CTC' in self.opts.Prediction:
            self.converter = CTCLabelConverter(self.opts.character)
        else:
            self.converter = AttnLabelConverter(self.opts.character)
        self.opts.num_class = len(self.converter.character)

        if self.opts.rgb:
            self.opts.input_channel = 3
        
        self.pre_load_model()
                

    #
    # private function to preload the torch model
    #
    def pre_load_model(self):

        opt = self.opts        
        print("preloading the model with opts "+str(opt))
        
        self.trmodel = Model(opt)        

        self.trmodel = torch.nn.DataParallel(self.trmodel)    
        if torch.cuda.is_available():
            self.trmodel = self.trmodel.cuda()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')       

        # load model
        print('loading pretrained model from %s' % self.opts.saved_model)
        if torch.cuda.is_available():
            self.trmodel.load_state_dict(torch.load(self.opts.saved_model))
        else:
            self.trmodel.load_state_dict(torch.load(opt.saved_model, map_location='cpu'))

        self.trmodel.eval()

    #
    #
    #
    def predictAllImagesInFolder(self, src_path):

        opt = self.opts
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = RawDataset(root=src_path, opt=opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=torch.cuda.is_available())

        results = []               
        for image_tensors, image_path_list in demo_loader:
                    
            preds_str = self.predict(image_tensors)
            
            for img_name, pred in zip(image_path_list, preds_str):
                if 'Attn' in opt.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                results.append(f'{os.path.basename(img_name)},{pred}')
        
        return results
                

    ##
    ##
    ##    
    def predict(self,image_tensors):

        print("############# About to predict for next batch ****")
        batch_size = image_tensors.size(0)
        with torch.no_grad():
            image = image_tensors.to(self.device)
            length_for_pred = torch.IntTensor([self.opts.batch_max_length] * batch_size)
            text_for_pred = torch.LongTensor(batch_size, self.opts.batch_max_length + 1).fill_(0)

        if 'CTC' in self.opts.Prediction:
            preds = self.trmodel(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self.trmodel(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        # print('-' * 80)
        # print('image_path\tpredicted_labels')
        # print('-' * 80)
        # pred = ""
        # for pred in preds_str:
            # if 'Attn' in self.opts.Prediction:
                # pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                # print("predictions = "+pred)
        
        return preds_str