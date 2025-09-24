__author__ = 'zengxh'
__version__ = '4.0'
# Interface for evaluating the PASCAL in Detail dataset. 
# Please visit https://sites.google.com/view/pasd/home for more
# information about the PASCAL in Detail challenge. For example usage of the
# detail API, see detailEvalDemo.ipynb.

# Throughout the API "cat"=category, and "img"=image.

# To import:
# from detail import DetailEvalCls

# PASCAL in Detail Toolbox     version 4.0
# TODO: Check the super_catergory of the classification
# TODO: Add param support, like support for evaluation on part of the imgs and categories
import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy
import json

class DetailEvalCls:
    def __init__(self, detailGt=None, iouType='cls'):
        '''
        Initialize DetailEvalCls using Detail APIs for gt 
        :param detail_gt: Detail object with ground truth annotations
        :param detail_res: Detail result file in json
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.detail_gt   = detailGt              # ground truth Detail API
        self.is_res_load = False
        #self.loadRes()
        #self.params   = {}                  # evaluation parameters
        #self.params.iouType = iouType
        
        self.cats = self.detail_gt.getCats() 
        self.num_cats = len(self.cats)
        self.gt_imgs = self.detail_gt.getImgs()
        self.num_img = len(self.gt_imgs)
        print('Total category #%d, total img #%d '%(self.num_cats, self.num_img))
        self.createIndex()
 
    def createIndex(self):
        '''
        createIndex to convert imgIds to gt_idx
        '''
        self.gt_idxs = {} #np.zeros([self.num_img])
        for i in range(self.num_img):
            self.gt_idxs[self.gt_imgs[i]['image_id']] = i
        #self.cat_idxs = {} #np.zeros([self.num_img])
        #for i in range(self.num_cats):
        #    self.cat_idxs[self.cats[i]['category_id']] = i

    def loadRes(self, detailRes):  
        # TODO: check if input is string or dict
        #print('loading result file: %s'%(self.detail_res))
        #import os.path
        #if not os.path.isfile(self.detail_res):
        #    print('File not exist!')
        #    return
        self.results = detailRes #json.load(open(self.DetailRes, 'r'))
        self.num_results = len(self.results)
        self.is_res_load = True

    def evaluate(self, plot=False):
        '''
        Evaluation of image classification task
        Run per image evaluation on all images in the set 
        :return: map 
        '''
        assert(self.is_res_load)
        #print('Running per image evaluation...')
        print('Evaluate annotation type cls') #*{}*'.format(p.iouType))
        voc_ap = np.zeros(self.num_cats)
        record = {}
        record['cat_ap'] = [] 
        for eval_order in range(self.num_cats):
            category_id = self.detail_gt.eval_orders[eval_order]
            cat = self.detail_gt.getCats([category_id])[0] #self.cats[i]
            #print('Evaluating class: %s(%d)...'%(cat['name'], cat['category_id']))
            voc_eval = self.VOCevalcls(eval_order, category_id)
            voc_ap[eval_order] = voc_eval['ap']
            if plot: 
                import matplotlib.pyplot as plt
                print(voc_eval['rec'], voc_eval['prec'])
                plt.plot(voc_eval['rec'], voc_eval['prec'])
                plt.show()
            record['cat_ap'].append({'category_id':cat['category_id'], 'name': cat['name'], 'ap':voc_ap[eval_order]})
        # average ap
        record['map'] = voc_ap.sum()/float(len(record['cat_ap']))
        #TODO: print stats as COCO?
        [print(record_single) for record_single in record['cat_ap']]
        print('======= \n Mean ap over #%d categories %f'%(self.num_cats, record['map']))
        return record['map']
 
    def VOCap(self, rec, prec):
        '''
        average percision calculations
        '''
        #TODO: add 2007 metric(11 recall point) if needed
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
   
    def VOCevalcls(self, eval_order, cat_id):
        '''
        Evaluate per category AP and average over number of category to get meanAP
        Follow the evaluation in VOC2010 Toolbox
        '''
        # imgs = self.detail_gt.getImgs(cats=cat['name']);
        # cat_id = self.cats[cat_idx]['category_id']
        gt = np.zeros([self.num_img])
        for i in range(self.num_img):
            if cat_id in self.gt_imgs[i]['categories']:
                gt[i] = 1
                #print(self.gt_imgs[i]['image_id'])
        #print('number of gt images: %d for cat_id: %d'%(gt.sum(), cat_id))
        out = np.zeros([self.num_img]) #* np.inf    
        # map results to gt img
        for i in range(self.num_results):
            img_id = self.results[i]['image_id']
            gt_idx = self.gt_idxs[img_id]
            out[gt_idx] = self.results[i]['category_score'][eval_order]
  
        so = np.sort(-out)
        si = np.argsort(-out)

                        
        tp = gt[si]==1
        fp = gt[si]==0
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        
        rec = tp/float((gt>0).sum()) if (gt>0).sum() else tp*0 
        prec = tp/np.maximum(fp+tp, np.finfo(np.float64).eps) # if (fp+tp) else 0 # aviod devide by 0
        ap = self.VOCap(rec, prec)
        
        return {'ap': ap, 'rec': rec, 'prec': prec}
