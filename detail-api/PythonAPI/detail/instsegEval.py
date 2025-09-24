import numpy as np
import json
from detail import param
from detail import mask as maskUtils
class instsegEval:
    def __init__(self,details):
        self.params = param.Params(iouType='segm')
        self.details = details
        self.params.useCat = True
        self.params.imgIds = list(details.imgs.keys())
        self.params.catIds = list(details.cats.keys())
        self.params.maxDets = [1, 10, 100, np.inf]
        self.params.iouThrs = \
            np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

    def loadSegRes(self, resFile='/Users/zhishuaizhang/github_repo/detail-api/res/inst_seg.json'): # Zhishuai
        '''
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        '''
        assert type(resFile) == str or type(resFile) == unicode
        anns = json.load(open(resFile))

        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.params.imgIds)), \
               'Results do not correspond to current Detail set'
        self.dts = [(_['image_id'],_['category_id'],_['segmentation'],_['score']) for _ in anns]

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''

        self.gts = [(i,j,maskUtils.encode(np.asfortranarray((self.details.getMask(i,cat=j).astype(np.uint8)==k).astype(np.uint8)))) \
            for i in self.params.imgIds \
                for j in [_['category_id'] for _ in self.details.getCats(imgs=i)] \
                    for k in (set(np.unique(self.details.getMask(i,cat=j).astype(np.uint8)))-{0})]

        imgIds_ = self.params.imgIds
        #self._prepare()
        catIds_ = self.params.catIds if self.params.useCats else [-1]

        maxDet_ = self.params.maxDets[-1]

        self.ious = {(imgId, catId): self._computeIoU(imgId, catId, self.params.useCats) \
                        for imgId in imgIds_
                        for catId in catIds_}

        self.evalImgs = [self._evaluateImg(imgId, catId, maxDet_)
                 for catId in catIds_
                 for imgId in imgIds_
             ]

    def _evaluateImg(self, imgId, catId, maxDet):
            '''
            perform evaluation for single category and image
            :return: dict (single image results)
            '''
            if self.params.useCats:
                gt = [_ for _ in self.gts if _[0] == imgId and _[1] == catId]
                dt = [_ for _ in self.dts if _[0 ]== imgId and _[1] == catId]
            else:
                gt = [_ for _ in self.gts if _[0] == imgId]
                dt = [_ for _ in self.dts if _[0] == imgId]
            if len(gt) == 0 and len(dt) ==0:
                return None
            iouThrs_ = self.params.iouThrs
            dtScores = [d[-1] for d in dt]
            dtind = np.argsort([-d[-1] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind[0:min(maxDet,len(dtind))]]
            # load computed ious
            ious_ = self.ious[imgId, catId]

            T = len(iouThrs_)
            G = len(gt)
            D = len(dt)
            gtm  = -np.ones((T,G))
            dtm  = -np.ones((T,D))
            # gtIg = np.array([g['_ignore'] for g in gt])
            # dtIg = np.zeros((T,D))
            if not len(ious_)==0:
                for tind, t in enumerate(iouThrs_):
                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min([t,1-1e-10])
                        m   = -1
                        for gind, g in enumerate(gt):
                            # if this gt already matched, and not a crowd, continue
                            if gtm[tind,gind]>-1:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            #if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            #    break
                            # continue to next gt unless better match made
                            if ious_[gind,dtind[dind]] < iou:
                                continue
                            # if match successful and best so far, store appropriately
                            iou=ious_[gind,dtind[dind]]
                            m=gind
                        # if match made store id of match for both dt and gt
                        if m ==-1:
                            continue
                        #dtIg[tind,dind] = gtIg[m]
                        dtm[tind,dtind[dind]]  = m
                        gtm[tind,m]     = dtind[dind]
            # set unmatched detections outside of area range to ignore
            # a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
            # dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
            # store results for given image and category
            return {
                    'image_id':     imgId,
                    'category_id':  catId,
                    'maxDet':       maxDet,
                    'dtMatches':    dtm,
                    'gtMatches':    gtm,
                    'dtScores':     dtScores,
                    'numGt': G,
                    'dtind': dtind,
                }

    def _computeIoU(self, imgId, catId, useCats = True):
        maxDet_ = self.params.maxDets[-1]
        if useCats:
            gt = [_[2] for _ in self.gts if _[0] == imgId and _[1] == catId]
            dt = [_[2] for _ in self.dts if _[0] == imgId and _[1] == catId]
        else:
            gt = [_[2] for _ in self.gts if _[0] == imgId]
            dt = [_[2] for _ in self.dts if _[0] == imgId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        '''
        inds = np.argsort([-d[-1] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        '''
        if len(dt) > maxDet_:
            dt=dt[0:maxDet_]

        iscrowd = [0 for o in gt]
        ious = maskUtils.iou(dt,gt,iscrowd)

        return ious

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,M))

        # create dictionary for future indexing
        setK = p.catIds
        setM = p.maxDets
        setI = p.imgIds
        # get inds to evaluate
        # k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        # m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        # a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        # i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(setI)
        # A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(setK):
            Nk = k*I0
            for m, maxDet in enumerate(setM):
                E = [self.evalImgs[Nk + i] for i in range(len(setI))]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([np.asarray(e['dtScores'])[e['dtind']][0:maxDet] if maxDet < np.inf else np.asarray(e['dtScores'])[e['dtind']] for e in E])
                num_gt = np.sum([e['numGt'] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')

                dtm  = np.concatenate([e['dtMatches'][:,e['dtind']][:,0:maxDet] if maxDet < np.inf else e['dtMatches'][:,e['dtind']] for e in E], axis=1)[:,inds]

                tps = dtm>-1
                fps = np.logical_not(dtm>-1)

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / num_gt
                    pr = tp / (fp+tp+np.spacing(1))
                    q  = np.zeros((R,))

                    if nd:
                        recall[t,k,m] = rc[-1]
                    else:
                        recall[t,k,m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                    except:
                        pass
                    precision[t,:,k,m] = np.array(q)
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'precision': precision,
            'recall':   recall,
        }
        self.ap = np.mean(self.eval['precision'],axis=1)
