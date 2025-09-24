__author__ = 'mricha56'
__version__ = '4.0'
# Interface for accessing the PASCAL in Detail dataset. detail is a Python API
# that assists in loading, parsing, and visualizing the annotations of PASCAL
# in Detail. Please visit https://sites.google.com/view/pasd/home for more
# information about the PASCAL in Detail challenge. For example usage of the
# detail API, see detailDemo.ipynb.

# Throughout the API "ann"=annotation, "cat"=category, "img"=image,
# "bbox"= bounding box, "kpts"=keypoints, "occl"=occlusion,
# "bounds"=boundaries.

# To import:
# from detail import Detail

# For help:
# help(Detail)

# PASCAL in Detail Toolbox     version 4.0
# Modifications of COCO toolbox made by Matt Richard and Zhuotun Zhu
# Forked from:
#     Microsoft COCO Toolbox.      version 2.0
#     Data, paper, and tutorials available at:  http://mscoco.org/
#     Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
#     Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon,Rectangle,Circle,Arrow,FancyArrow
import matplotlib.colors
import numpy as np
import skimage.io as io
import copy
import itertools
from scipy.ndimage.morphology import binary_dilation
from . import mask as maskUtils
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

# When displaying boundaries, dilate the mask before displaying it, to
# improve visibility
NUM_BOUNDARY_DILATION_ITERATIONS = 1

# When displaying occlusion, draw an arrow every OCCLUSION_ARROW_DISTANCE
# pixels
OCCLUSION_ARROW_DISTANCE = 7

class Detail:
    def __init__(self, annotation_file='json/trainval_withkeypoints.json',
                 image_folder='VOCdevkit/VOC2010/JPEGImages',
                 phase='trainval'):
        """
        Constructor of Detail helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that has pascal JPEG images.
        :param phase (str): image set to look at: train, val, test, or any combination
                            of the three (trainval, trainvaltest)
        :return:
        """
        # load dataset
        self.phase = phase
        self.img_folder = image_folder

        print('loading annotations into memory...')
        tic = time.time()

        self.data = json.load(open(annotation_file, 'r'))
        assert type(self.data)==dict, 'annotation file format {} not supported'.format(type(self.data))
        print('JSON root keys:' + str(self.data.keys()))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

        self.waiting = False
        self.__createIndex()

    def __createIndex(self):
        # create index
        tic = time.time()
        print('creating index...')

        # create class members
        self.cats,self.imgs,self.segmentations,self.occlusion,self.parts,\
            self.kpts, self.bounds= {},{},{},{},{},{},{}

        # Organize data into instance variables
        for img in self.data['images']:
            self.imgs[img['image_id']] = img
        for segm in self.data['annos_segmentation']:        # many per image
            self.segmentations[segm['id']] = segm
        for occl in self.data['annos_occlusion']:           # one per image
            self.occlusion[occl['image_id']] = occl
        for bound in self.data['annos_boundary']:           # one per image
            self.bounds[bound['image_id']] = bound
        #for skeleton in self.data['annos_joints']:          # many per image
        #    # skeletons are 1-indexed in JSON file and
        #    # 0-indexed in self.kpts
        #    self.kpts[skeleton['person_id'] - 1] = skeleton


        # Follow references
        for img in self.data['images']:
            img['annotations'] = []
            img['categories'] = []
            img['parts'] = []
            img['keypoints'] = []

        for part in self.data['parts']:
            part['categories'] = []
            part['annotations'] = []
            part['images'] = []
            self.parts[part['part_id']] = part

        # fixed eval_orders here for classification task
        self.eval_orders = {}
        eval_orders = [2, 23, 25, 31, 34, 45, 59, 65, 72, 98, 397, 113, 207, 258, 284, 308, 347, 368, 416, 427, 9, 18, 22, 33, 44, 46, 68, 80, 85, 104, 115, 144, 158, 159, 162, 187, 189, 220, 232, 259, 260, 105, 296, 355, 295, 324, 326, 349, 354, 360, 366, 19, 415, 420, 424, 440, 445, 454, 458]
        for i in range(len(eval_orders)):
            self.eval_orders[i] = eval_orders[i]
        for order, cat in enumerate(self.data['categories']):
            cat['images'] = []
            cat['annotations'] = []
            #print('adding cat id: %d'%(cat['category_id']))
            self.cats[cat['category_id']] = cat
            # self.eval_orders[order] = cat['category_id']
            if cat.get('parts'):
                for partId in cat['parts']:
                    part = self.parts[partId]
                    if cat['category_id'] not in part['categories']:
                        part['categories'].append(cat['category_id'])


        self.keypoints_str = ['head', 'neck', 'lsho', 'lelb', 'lhip', 'lwri', 'lknee', 'lank', 'rsho', 'relb', 'rwri', 'rhip', 'rknee', 'rank']
        for skeleton_id, skeleton in self.kpts.items():
             self.imgs[skeleton['image_id']]['keypoints'].append(skeleton_id)

        for segm_id, segm in self.segmentations.items():
            img = self.imgs[segm['image_id']]
            cat = self.cats[segm['category_id']]
            img['annotations'].append(segm_id)
            cat['annotations'].append(segm_id)

            if cat['category_id'] not in img['categories']:
                img['categories'].append(cat['category_id'])
            if img['image_id'] not in cat['images']:
                cat['images'].append(img['image_id'])

            if segm.get('parts'):
                for partsegm in segm['parts']:
                    if partsegm['part_id'] == 255: continue

                    part = self.parts[partsegm['part_id']]
                    part['annotations'].append(segm_id)
                    if img['image_id'] not in part['images']:
                        part['images'].append(img['image_id'])
                    if part['part_id'] not in img['parts']:
                        img['parts'].append(part['part_id'])

        print('index created! (t={:0.2f}s)'.format(time.time() - tic))

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.data['info'].items():
            print('{}: {}'.format(key, value))

    def __getSegmentationAnns(self, anns=[], imgs=[], cats=[], areaRng=[], supercat=None, crowd=None):
        """
        Get segmentation annotations that satisfy given filter conditions. default is no filter
        :param anns (int array) : get anns with the given IDs
               imgs  (image array)     : get anns in the given imgs
               cats  (category array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               supercat (str) : filter anns by supercategory
               crowd (True/False) : filter anns by 'iscrowd' key
        :return: anns  (annotation array)       : array of annotations
        """
        if len(imgs) > 0: imgs = self.getImgs(imgs)
        if len(cats) > 0: cats = self.getCats(cats)

        anns = self.__toList(anns)

        # Get starting list of anns
        if len(anns) == 0:
            anns = list(self.segmentations.values())
        else:
            for i in range(len(anns)):
                try:
                    if type(anns[i]) is int: anns[i] = self.segmentations[anns[i]]
                    elif type(anns[i]) is dict: anns[i] = self.segmentations[anns[i]['id']]
                except IndexError: assert False, 'Annotation with id %s not found' % anns[i]['id']

        # Filter anns according to params
        imgAnns = np.unique(np.array([img['annotations'] for img in imgs]).flatten())
        catAnns = np.unique(np.array([cat['annotations'] for cat in cats]).flatten())
        if len(imgs) > 0:
            anns = [ann for ann in anns if ann['id'] in imgAnns]
        if len(cats) > 0:
            anns = [ann for ann in anns if ann['id'] in catAnns]
        if len(areaRng) == 2:
            anns = [ann for ann in anns if ann['area'] >= areaRng[0] and ann['area'] <= areaRng[1]]
        if supercat is not None:
            subcats = [cat['category_id'] for cat in self.getCats(supercat=supercat)]
            anns = [ann for ann in anns if ann['category_id'] in subcats]
        if crowd is not None:
            if (crowd):
                anns = [ann for ann in anns if ann['iscrowd']]
            else:
                anns = [ann for ann in anns if not ann['iscrowd']]

        return anns

    # getX() functions #

    def getOccl(self, img, show=False):
        img = self.getImgs(img)[0]

        occl = self.occlusion[img['image_id']]

        if show:
            self.showOccl(occl, img)

        return occl

    def getBounds(self, img, show=False):
        """
        Get boundary mask for given image.
        """
        img = self.getImgs(img)[0]

        bound = self.bounds[img['image_id']]
        mask = self.decodeMask(bound['boundary_mask'])

        if show:
            if np.count_nonzero(mask) > 0:
                self.showBounds(mask, img)
            else:
                print('Mask is empty')

        return mask

    def getBboxes(self, img, cat='object', show=False):
        """
        Get bounding box for each instance of given category in image.

        :param img : image containing bounding boxes
        :param cat : category or supercategory to filter by. Default returns
                     bboxes for all "object" (non-background) categories.
        :param show (boolean): whether to pass result to self.showBboxes() before
                               proceeding.
        :return: bboxes : list of bboxes, where each bbox is a dict:
                          {'bbox':[pos_x, pos_y, width, height],
                           'category': 'category_name'}
        """
        img = self.getImgs(img)[0]

        if cat in ['object', 'animal', 'background']: # supercategory
            anns = self.__getSegmentationAnns(imgs=img, supercat=cat,crowd=False)
        else:
            cat = self.getCats(cat)[0]
            assert not cat['onlysemantic'], 'No instance-level data for category %s' % cat['name']
            anns = self.__getSegmentationAnns(imgs=img, cats=cat, crowd=False)

        bboxes = []
        for ann in anns:
            bboxes.append({
                'bbox': ann['bbox'],
                'category': self.getCats(ann['category_id'])[0]['name']})

        if show:
            self.showBboxes(bboxes, img)
        return bboxes

    def getMask(self, img, cat=None, instance=None, superpart=None, part=None, show=False):
        """
        Get mask for a particular level of segmentation. You may "drill down" to
        the desired level of detail by specifying more parameters.

        If semantic segmentation of an image is requested (cat=instance=superpart=part=None), the result
        is an image whose pixel values are the class IDs for that image.

        If instance-level segmentation for one category of an image is requested (img and cat provided),
        the result is an image whose pixel values are the instance IDs for that class and 0 everywhere else.

        If part-level segmentation of an instance is requested (img, cat, and instance provided),
        the result is an image whose pixel values are the part IDs for that instance
        and 0 everywhere else.

        If a single-part binary mask for a part or superpart is requested (img,
        cat, instance, and part or superpart provided), the result is an image whose pixel values are
        0 everywhere except for the given part/superpart.

        :param img (string/int/dict) : image that mask describes
               cat (string/int/dict) : category or supercategory that mask describes
               instance (string/int/dict) : instance that the mask describes. If integer, interpreted
                                            as id of an "annotation" object in JSON. If
                                            string starting with #, e.g. '#0', interpreted as 0-based index
                                            of instance within the image (cat is None)
                                            or of instance within the given class (cat not None).
               part (string or int) :  part that mask describes (None means all parts)
               superpart (string): superpart that mask describes
               show (boolean) :  whether to pass the mask to self.showMask() before returning.
        :return: mask (numpy 2D array) : a mask describing the requested annotation.
        """

        # Validate params and convert them to dicts
        img = self.getImgs(img)[0]
        supercat = None
        if cat is not None:
            if cat in ['object', 'animal', 'background']:
                supercat = cat
                cat = None
            else:
                cat = self.getCats(cat)[0]
        if part is not None:
            part = self.getParts(part)[0]

        # When part or superpart is requested, instance is assumed to be first instance
        # of the given category
        if (cat or supercat) and (part or superpart) and not instance:
            instance = '#0'

        if instance is not None:
            try:
                if type(instance) is str:
                    if instance.startswith('#'):
                        # If instance is set to '#N' where N is an integer,
                        # get the Nth (0-indexed) instance of the given category.
                        if cat is not None:
                            instance = self.__getSegmentationAnns(imgs=img, cats=cat)[int(instance[1:])]
                        else:
                            instance = self.__getSegmentationAnns(imgs=img, supercat='object')[int(instance[1:])]
                    else:
                        instance = self.__getSegmentationAnns(int(instance))[0]
                elif type(instance) is int:
                    instance = self.__getSegmentationAnns(instance)[0]
            except IndexError:
                assert False, 'Couldn\'t find the requested instance'


        anns = self.__getSegmentationAnns(imgs=img, cats=[] if cat is None else cat,
                                          supercat=supercat, crowd=None if instance is None else False)
        mask = np.zeros((img['height'], img['width']))

        # Generate mask based on params
        if not (cat or supercat or instance or part):
            # Generate class mask
            for ann in anns:
                m = self.decodeMask(ann['segmentation'])
                mask[np.nonzero(m)] = ann['category_id']
        elif (cat or supercat) and not (instance or part):
            # Generate instance (or single-class semantic) mask
            i = 1
            for ann in anns:
                m = self.decodeMask(ann['segmentation'])
                if cat and cat['onlysemantic']:
                    mask[np.nonzero(m)] = 1
                else:
                    mask[np.nonzero(m)] = i
                    i = i + 1
        elif instance and not part:
            assert not instance['iscrowd'], 'Instance-level segmentation not available'
            # Generate part mask
            for p in instance['parts']:
                m = self.decodeMask(p['segmentation'])
                mask[np.nonzero(m)] = p['part_id']

            if superpart is not None:
                parts = [p['part_id'] for p in self.getParts(superpart=superpart)]
                newmask = np.zeros(mask.shape)
                for p in parts:
                    newmask += p * (mask == p)
                mask = newmask
        elif instance and part:
            # Generate single-part mask
            partMask = [p['segmentation'] for p in instance['parts'] \
                        if p['part_id'] == part['part_id']]
            assert len(partMask) > 0, 'Coudn\'t find a part mask for the given part and instance'
            partMask = partMask[0]
            m = self.decodeMask(partMask)
            mask[np.nonzero(m)] = part['part_id']
        else:
            assert False, 'Invalid parameters'

        if show:
            if np.count_nonzero(mask) > 0:
                self.showMask(mask, img)
            else:
                print('Mask is empty')

        return mask

    def getKptAnno(self, skeleton_id=0):
        """
        Get keypoints annotations by skeleton_id
        :param skeleton_id (int): get the #skeleton_id of kpts annotations 
        :return: kpt_annotation (dict)   : kpts dicts
        """
        assert(type(skeleton_id) is int) # skeleton_id must be int
        assert(skeleton_id < len(self.kpts) and skeleton_id >= 0) # skeleton_id can not get out of bound
        return self.kpts[skeleton_id]

    def getKpts(self, img, show=False):
        """
        Get human keypoints for the image.
        :param imgs (int/string/dict array)  : get cats present in at least one of the given image names/ids
        :return: kpts (dict array)  : array of kpts dict in the img
        """
        img = self.getImgs(img)[0]
        kpts = []
        for skeleton_id in img['keypoints']:
            kpts.append(self.kpts[skeleton_id])

        if show:
            self.showKpts(kpts, img)

        return kpts

    def getCats(self, cats=[], imgs=[], supercat=None, with_instances=None):
        """
        Get categories abiding by the given filters. default is no filter.
        :param cats (int/string/dict array)  : get cats for given cat names/ids/dicts
        :param imgs (int/string/dict array)  : get cats present in at least one of the given image names/ids
        :param supercat : get cats that belong to the specified supercategory
        :param with_instances (boolean): filter cats based on whether they have
                                        instance-level annotations
        :return: cats (dict array)   : array of category dicts
        """
        cats = self.__toList(cats)
        if len(cats) == 0:
            cats = list(self.cats.values())
        else:
            for i in range(len(cats)):
                if type(cats[i]) == int: cats[i] = self.cats[cats[i]]
                elif type(cats[i]) == dict: cats[i] = self.cats[cats[i]['category_id']]
                elif type(cats[i]) == str:
                    try:
                        cats[i] = [c for c in self.cats.values() if c['name'] == cats[i]][0]
                    except IndexError:
                        assert False, 'Category "%s" not found' % cats[i]


        if type(imgs) is not list or len(imgs) > 0:
            imgs = self.getImgs(imgs)
            catIds = np.unique(np.array([img['categories'] for img in imgs]).flatten())
            cats = [cat for cat in cats if cat['category_id'] in catIds]

        if supercat is not None:
            scs = []
            if supercat is 'object': scs = ['object', 'animal']
            else: scs = [supercat]
            cats = [cat for cat in self.cats.values() if cat['supercategory'] in scs]

        if with_instances is not None:
            cats = [cat for cat in cats if not cat['onlysemantic'] == with_instances]

        return cats

    def getSuperparts(self, cat=None):
        """
        Get list of all defined superparts.
        :return: superparts (string array): list of superpart names
        """
        superparts = set()
        parts = self.getParts(cat=cat)
        for part in parts:
            if part['superpart'] != 'none':
                superparts.add(part['superpart'])

        return list(superparts)

    def getParts(self, parts=[], cat=None, superpart=None):
        """
        Get parts of a particular category.
        :param parts (int/string/dict array) : list of parts to get
        :param cat (int, string, or dict) : category to get parts for (default: any)
        :param superpart (string) : superpart to get parts for - one of ["object",
                                    "background", "animal"]
        :return: parts (dict array) : array of part dicts, e.g.
        [{"name": "mouth", "superpart": "head", "part_id": 110},...]
        """
        parts = self.__toList(parts)
        if len(parts) == 0:
            parts = list(self.parts.values())
        else:
            for i in range(len(parts)):
                if type(parts[i]) == int: parts[i] = self.parts[parts[i]]
                elif type(parts[i]) == dict: parts[i] = self.parts[parts[i]['part_id']]
                elif type(parts[i] == str):
                    try: parts[i] = [p for p in self.parts.values() if p['name'] == parts[i]][0]
                    except IndexError: assert False, 'No part named \"%s\"' % parts[i]

        if cat is not None:
            cat = self.getCats(cat)[0]

        if cat is not None:
            oldparts = copy.copy(parts)
            for part in oldparts:
                if part['part_id'] not in cat['parts']:
                    parts.remove(part)

        if superpart is not None:
            oldparts = copy.copy(parts)
            for part in oldparts:
                if part['superpart'] != superpart:
                    parts.remove(part)

        return parts

    def getImgs(self, imgs=[], cats=[], supercat=None, phase=None):
        '''
        Get images that satisfy given filter conditions.
        :param imgs (int/string/dict array) : get imgs with given ids
        :param cats (int/string/dict array) : get imgs with all given cats
        :param supercat (string) : get imgs with the given supercategory
        :param phase (string) : filter images by phase. If None, the phase
                                provided to the Detail() constructor is used.
        :return: images (dict array)  :  array of image dicts
        '''
        if phase is None:
            phase = self.phase
        phases = []
        if "train" in phase: phases.append("train")
        if "val" in phase: phases.append("val")
        if "test" in phase: phases.append("test")
        assert len(phases) > 0, 'Invalid phase, {}'.format(phase)

        imgs = self.__toList(imgs)
        if len(imgs) == 0:
            imgs = list(self.imgs.values())
        else:
            for i in range(len(imgs)):
                if type(imgs[i]) == int: imgs[i] = self.imgs[imgs[i]]
                elif type(imgs[i]) == dict: imgs[i] = self.imgs[imgs[i]['image_id']]
                elif type(imgs[i]) == str:
                    imstr = imgs[i]
                    imgs[i] = self.imgs[int(imstr[:4] + imstr[5:])]

        if type(cats) is not list or len(cats) > 0:
            cats = self.getCats(cats)
            oldimgs = copy.copy(imgs)
            for img in oldimgs:
                for cat in cats:
                    if cat['category_id'] not in img['categories']:
                        imgs.remove(img)
                        break

        if supercat is not None:
            catIds = set([c['category_id'] for c in self.getCats(supercat=supercat)])
            oldimgs = copy.copy(imgs)
            for img in oldimgs:
                if len(catIds & set(img['categories'])) == 0:
                    imgs.remove(img)

        oldimgs = copy.copy(imgs)
        for img in oldimgs:
            if img['phase'] not in phases:
                imgs.remove(img)

        return imgs

    # showX() functions #

    def showImg(self, img, wait=False, ax=None):
        """
        Display the given image
        """
        img = self.getImgs(img)[0]
        jpeg = io.imread(os.path.join(self.img_folder, img['file_name']))

        # print image details
        print('showing image %s: ' % img['file_name'])
        keys = ['image_id', 'width', 'height', 'phase', 'date_captured']
        for k in keys:
            print('\t%s: %s,' % (k, img[k] if img.get(k) else 'N/A'))

        if ax is not None:
            ax.imshow(jpeg)
        else:
            plt.imshow(jpeg)

        plt.axis('off')
        if wait:
            self.waiting = True
        else:
            plt.show()

    def showMask(self, mask, img=None):
        """
        Display given mask (numpy 2D array) as a colormapped image.
        """
        if img is not None and not self.waiting:
            self.showImg(img, wait=True)

        # Draw mask, random colormap, 0s transparent
        self.waiting = False
        mycmap = self.__genRandColormap()
        mycmap.set_under(alpha=0.0)
        nonzero = np.unique(mask[np.nonzero(mask)])
        plt.imshow(mask, cmap=mycmap, vmin=np.min(nonzero), vmax=np.max(nonzero)+1)
        plt.axis('off')
        plt.show()

    def showBboxes(self, bboxes, img=None):
        """
        Display given bounding boxes.
        """
        fig,ax = plt.subplots(1)
        if img is not None and not self.waiting:
            self.showImg(img, wait=True, ax=ax)

        for bbox in bboxes:
            ax.add_patch(Rectangle((bbox['bbox'][0],bbox['bbox'][1]), bbox['bbox'][2], bbox['bbox'][3], linewidth=2,
                                   edgecolor=np.random.rand(3), facecolor='none',
                                   label=bbox['category']))
        print('categories: %s' % [bbox['category'] for bbox in bboxes])

        self.waiting = False
        plt.legend()
        plt.axis('off')
        plt.show()

    def showKpts(self, kpts, img=None):
        """
        Display given kpts.
        """
        fig,ax = plt.subplots(1)
        if img is not None:
            self.showImg(img, wait=True, ax=ax)

        pv = np.zeros(14)
        px = np.zeros(14)
        py = np.zeros(14)
        for kpt in kpts:
            skeleton_color = np.random.rand(3)
            num_kpt = len(kpt['keypoints'])/3 # always 14
            assert num_kpt == 14, 'Expected 14 keypoints but found {}'.format(num_kpt)

            for i in range(int(num_kpt)):
                px[i] = kpt['keypoints'][3*i]
                py[i] = kpt['keypoints'][3*i+1]
                pv[i] = kpt['keypoints'][3*i+2]

            kpt_pair = [[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13]]
            for p in kpt_pair:
                p0 = p[0]
                p1 = p[1]
                if pv[p0] == 0 or pv[p1] == 0:
                    continue
                if pv[p0] == 2 or pv[p1] == 2:
                    pcolor = 'blue'
                else:
                    pcolor = 'red'
                ax.add_patch(Arrow(px[p0], py[p0], px[p1]-px[p0], py[p1]-py[p0],
                                   width=2.0, facecolor=skeleton_color,
                                   edgecolor=skeleton_color))

            for i in range(int(num_kpt)):
                if pv[i] == 0:
                    continue
                pcolor = 'none'
                if pv[i] == 1:
                    pcolor = 'red'
                else:
                    pcolor = 'blue'
                ax.add_patch(Circle((px[i], py[i]), radius=3, facecolor=pcolor,
                                    edgecolor=skeleton_color, linewidth=2.0))

        self.waiting = False
        plt.axis('off')
        plt.show()

    def showBounds(self, mask, img):
        """
        Dilate mask before passing it to showMask()
        """
        img = self.getImgs(img)[0]

        # dilate mask (creates new ndarray of bools)
        mask = binary_dilation(mask, iterations=NUM_BOUNDARY_DILATION_ITERATIONS)

        # show mask
        self.showMask(mask, img)

    def showOccl(self, occl, img):
        """
        Show occlusion data
        """
        img = self.getImgs(img)[0]

        fig,ax = plt.subplots(1)
        if img is not None and not self.waiting:
            self.showImg(img, wait=True, ax=ax)

        bounds = np.zeros(occl['imsize'])
        for i in range(occl['ne']): # ne = "number of edges"

            pixel_indices = occl['edges']['indices'][i]
            num_pixels = len(pixel_indices)
            pixel_coords = np.unravel_index(pixel_indices, occl['imsize'], order='F')
            edgelabel = occl['edges']['edgelabel'][i]
            bounds[pixel_coords] = 1

            for j in range(num_pixels - OCCLUSION_ARROW_DISTANCE):
                if j % OCCLUSION_ARROW_DISTANCE == 0:
                    # draw arrow
                    a = [pixel_coords[0][j]  , pixel_coords[1][j]]
                    b = [pixel_coords[0][j+OCCLUSION_ARROW_DISTANCE],
                         pixel_coords[1][j+OCCLUSION_ARROW_DISTANCE] ]

                    # midpoint
                    mid = (np.array(a) + np.array(b)) / 2

                    d0 = b[0] - a[0]
                    d1 = b[1] - a[1]

                    # point towards b if edgelabel is 1,
                    # else (edgelabel is 2), point away
                    if edgelabel != 1:
                        d0 *= -1
                        d1 *= -1

                    # normalize direction vector
                    norm = np.sqrt(d0 * d0 + d1 * d1)
                    d0 /= norm
                    d1 /= norm

                    ax.add_patch(FancyArrow(mid[1], mid[0], d0, -d1, width=0.0,
                                            head_width=2.5, head_length = 5.0,
                                            facecolor='white', edgecolor='none'))


        # show mask
        self.showMask(bounds, img)

    # Helper functions #

    def __toList(self, param):
        return param if type(param) == list else [param]

    def __genRandColormap(self):
        return matplotlib.colors.ListedColormap(np.random.rand (256,3))

    def decodeMask(self, json):
        """
        Convert json mask to binary mask.
        :return: binary mask (numpy 2D array)
        """
        return maskUtils.decode(json)

    def find_cat(self,index): # Zhishuai
        try:
            assert len(self.cats_mapping)>0
        except:
            self.cats_mapping=[(_['category_id'],_['name']) for _ in self.getCats()]
        if type(index)==int:
            return self.cats_mapping[[_[0] for _ in self.cats_mapping].index(index)][1]
        elif type(index)==str:
            return self.cats_mapping[[_[1] for _ in self.cats_mapping].index(index)][0]


    # From COCO API...not implemented for PASCAL in Detail
#     def showAnns(self, anns):
#         """
#         Display the specified annotations.
#         :param anns (array of object): annotations to display
#         :return: None
#         """
#         if len(anns) == 0:
#             return 0
#         if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
#             datasetType = 'instances'
#         elif 'caption' in anns[0]:
#             datasetType = 'captions'
#         else:
#             raise Exception('datasetType not supported')
#         if datasetType == 'instances':
#             ax = plt.gca()
#             ax.set_autoscale_on(False)
#             polygons = []
#             color = []
#             for ann in anns:
#                 c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
#                 if 'segmentation' in ann:
#                     if type(ann['segmentation']) == list:
#                         # polygon
#                         for seg in ann['segmentation']:
#                             poly = np.array(seg).reshape((int(len(seg)/2), 2))
#                             polygons.append(Polygon(poly))
#                             color.append(c)
#                     else:
#                         # mask
#                         t = self.imgs[ann['image_id']]
#                         if type(ann['segmentation']['counts']) == list:
#                             rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
#                         else:
#                             rle = [ann['segmentation']]
#                         m = maskUtils.decode(rle)
#                         img = np.ones( (m.shape[0], m.shape[1], 3) )
#                         if ann['iscrowd'] == 1:
#                             color_mask = np.array([2.0,166.0,101.0])/255
#                         if ann['iscrowd'] == 0:
#                             color_mask = np.random.random((1, 3)).tolist()[0]
#                         for i in range(3):
#                             img[:,:,i] = color_mask[i]
#                         ax.imshow(np.dstack( (img, m*0.5) ))
#                 if 'keypoints' in ann and type(ann['keypoints']) == list:
#                     # turn skeleton into zero-based index
#                     sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
#                     kp = np.array(ann['keypoints'])
#                     x = kp[0::3]
#                     y = kp[1::3]
#                     v = kp[2::3]
#                     for sk in sks:
#                         if np.all(v[sk]>0):
#                             plt.plot(x[sk],y[sk], linewidth=3, color=c)
#                     plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
#                     plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
#             p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
#             ax.add_collection(p)
#             p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
#             ax.add_collection(p)
#         elif datasetType == 'captions':
#             for ann in anns:
#                 print(ann['caption'])


    # From COCO API...not implemented for PASCAL in Detail
#     def loadRes(self, resFile):
#         """
#         Load result file and return a result api object.
#         :param   resFile (str)     : file name of result file
#         :return: res (obj)         : result api object
#         """
#         res = Detail()
#         res.dataset['images'] = [img for img in self.dataset['images']]
#
#         print('Loading and preparing results...')
#         tic = time.time()
#         if type(resFile) == str or type(resFile) == unicode:
#             anns = json.load(open(resFile))
#         elif type(resFile) == np.ndarray:
#             anns = self.loadNumpyAnnotations(resFile)
#         else:
#             anns = resFile
#         assert type(anns) == list, 'results in not an array of objects'
#         annsImgIds = [ann['image_id'] for ann in anns]
#         assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
#                'Results do not correspond to current Detail set'
#         if 'caption' in anns[0]:
#             imgIds = set([img['image_id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
#             res.dataset['images'] = [img for img in res.dataset['images'] if img['image_id'] in imgIds]
#             for id, ann in enumerate(anns):
#                 ann['id'] = id+1
#         elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 bb = ann['bbox']
#                 x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
#                 if not 'segmentation' in ann:
#                     ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
#                 ann['area'] = bb[2]*bb[3]
#                 ann['id'] = id+1
#                 ann['iscrowd'] = 0
#         elif 'segmentation' in anns[0]:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 # now only support compressed RLE format as segmentation results
#                 ann['area'] = maskUtils.area(ann['segmentation'])
#                 if not 'bbox' in ann:
#                     ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
#                 ann['id'] = id+1
#                 ann['iscrowd'] = 0
#         elif 'keypoints' in anns[0]:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 s = ann['keypoints']
#                 x = s[0::3]
#                 y = s[1::3]
#                 x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
#                 ann['area'] = (x1-x0)*(y1-y0)
#                 ann['id'] = id + 1
#                 ann['bbox'] = [x0,y0,x1-x0,y1-y0]
#         print('DONE (t={:0.2f}s)'.format(time.time()- tic))
#
#         res.dataset['annotations'] = anns
#         res.createIndex()
#         return res


    # From COCO API...not implemented for PASCAL in Detail
#     def loadNumpyAnnotations(self, data):
#         """
#         Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
#         :param  data (numpy.ndarray)
#         :return: annotations (python nested list)
#         """
#         print('Converting ndarray to lists...')
#         assert(type(data) == np.ndarray)
#         print(data.shape)
#         assert(data.shape[1] == 7)
#         N = data.shape[0]
#         ann = []
#         for i in range(N):
#             if i % 1000000 == 0:
#                 print('{}/{}'.format(i,N))
#             ann += [{
#                 'image_id'  : int(data[i, 0]),
#                 'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
#                 'score' : data[i, 5],
#                 'category_id': int(data[i, 6]),
#                 }]
#         return ann
