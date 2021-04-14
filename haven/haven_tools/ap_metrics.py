import numpy as np
import copy
import pycocotools.mask as mask_util
from itertools import product
import torch


class APMonitor:
    def __init__(self):
        self.pred_ann_list = []
        self.gt_ann_list = []
        self.n_batches = 0.0
        self.iou_thr = 0.5
        self.iou_thr_list = [0.5, 0.75]

    def add(self, gt_ann_list, pred_ann_list):
        self.pred_ann_list += pred_ann_list
        self.gt_ann_list += gt_ann_list

        self.n_batches += 1.0

    def get_avg_score(self):
        results = compute_precision(
            self.gt_ann_list, self.pred_ann_list, iouType="segm", iouThr=0.5, iouThrList=np.array([0.25, 0.5, 0.75])
        )
        results = {k: v for k, v in results.items() if "mAP" in k}
        results["val_score"] = results["mAP50.0"]

        return results


def compute_precision(gt_annList, pred_annList, iouType, iouThr, iouThrList):
    gt_annList = copy.deepcopy(gt_annList)
    # pred_annList = copy.deepcopy(pred_annList)
    if len(gt_annList) == 0:
        return {"mAP50.0": 0}
    if len(pred_annList) == 0:
        return {"mAP50.0": 0}

    result_dict = evaluate_annList(
        pred_annList=pred_annList, gt_annList=gt_annList, iouType=iouType, iouThr=iouThr, ap=1, iouThrList=iouThrList
    )

    return result_dict


def evaluate_annList(
    pred_annList, gt_annList, ap=1, iouType="bbox", iouThr=0.5, maxDets=100, aRngLabel="all", iouThrList=None
):
    if iouThrList is None:
        iouThrList = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # print("iouThrList", iouThrList)
    recThrList = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)

    areaRngList = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    aRngLabelList = ["all", "small", "medium", "large"]
    maxDetList = [1, 10, 100]

    # Put them in dict
    gt_dict = {}  # gt for evaluation
    dt_dict = {}  # dt for evaluation
    imgList = set()
    catList = set()

    for i, gt in enumerate(copy.deepcopy(gt_annList)):
        key = (gt["image_id"], gt["category_id"])
        if key not in gt_dict:
            gt_dict[key] = []

        gt["id"] = i + 1
        gt_dict[key] += [gt]

        imgList.add(gt["image_id"])
        catList.add(gt["category_id"])

    for i, dt in enumerate(copy.deepcopy(pred_annList)):
        key = (dt["image_id"], dt["category_id"])
        if key not in dt_dict:
            dt_dict[key] = []

        bb = dt["bbox"]
        dt["area"] = bb[2] * bb[3]

        dt["id"] = i + 1
        dt["iscrowd"] = 0

        dt_dict[key] += [dt]

        # imgList.add(dt['image_id'])
        # catList.add(dt['category_id'])

    imgList = list(imgList)
    catList = list(catList)

    # compute ious
    iou_dict = {}
    for imgId, catId in product(imgList, catList):
        key = (imgId, catId)

        if key in gt_dict:
            gt = gt_dict[key]
        else:
            gt = []

        if key in dt_dict:
            dt = dt_dict[key]
        else:
            dt = []

        iou_dict[key] = computeIoU(gt, dt, iouType=iouType, maxDets=maxDets)
    # evaluate detections
    evalImgs = []
    for catId, areaRng, imgId in product(catList, areaRngList, imgList):
        evalImgs += [evaluateImg(gt_dict, dt_dict, iou_dict, imgId, catId, areaRng, maxDets, iouThrs=iouThrList)]

    eval_dict = accumulate(evalImgs, imgList, catList, iouThrList, recThrList, areaRngList, maxDetList)

    result_dict = {"iouThrList": iouThrList, "iouType": iouType, "n_classes": len(catList), "categories": catList}

    # Get thresholds for individual APs
    aind = [i for i, aRng in enumerate(aRngLabelList) if aRng == aRngLabel]
    mind = [i for i, mDet in enumerate(maxDetList) if mDet == maxDets]

    for thr in iouThrList:
        ap_dict = compute_AP_dict(ap, aind, mind, eval_dict, thr, iouThrList)
        result_dict.update(ap_dict)

    if ap == 1:
        result_dict["mAP"] = result_dict["mAP%s" % (iouThr * 100)]
    else:
        result_dict["mRC"] = result_dict["mRC%s" % (iouThr * 100)]
    return result_dict


def computeIoU(gt, dt, iouType="bbox", maxDets=100):
    if len(gt) == 0 and len(dt) == 0:
        return []

    inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
    dt = [dt[i] for i in inds]
    if len(dt) > maxDets:
        dt = dt[0:maxDets]

    if iouType == "segm":
        g = [g["segmentation"] for g in gt]
        d = [d["segmentation"] for d in dt]
    elif iouType == "bbox":
        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]
    else:
        raise Exception("unknown iouType for iou computation")

    # compute iou between each dt and gt region
    iscrowd = [int(o["iscrowd"]) for o in gt]

    ious = mask_util.iou(d, g, iscrowd)

    # ####
    # print("id:%s - cat:%d" % (imgId, catId))
    # print("d:", d)
    # print("g:", g)
    # print("ious", ious)
    # ####
    return ious


def compute_AP_dict(ap, aind, mind, eval_dict, iouThr, iouThrList):
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = eval_dict["precision"]
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == iouThrList)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = eval_dict["recall"]
        if iouThr is not None:
            t = np.where(iouThr == iouThrList)[0]
            s = s[t]
        s = s[:, :, aind, mind]

    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])

    per_class = []
    for i in range(s.shape[2]):
        si = s[:, :, i]
        per_class += [np.mean(si[si > -1])]

    if ap == 1:
        result_dict = {"AP%s" % (iouThr * 100): np.array(per_class), "mAP%s" % (iouThr * 100): mean_s}
    else:
        result_dict = {"RC%s" % (iouThr * 100): np.array(per_class), "mRC%s" % (iouThr * 100): mean_s}

    return result_dict


def evaluateImg(gt_dict, dt_dict, ious, imgId, catId, aRng, maxDet, iouThrs):
    """
    perform evaluation for single category and image
    :return: dict (single image results)
    """
    # p = self.params
    key = (imgId, catId)
    if key in gt_dict:
        gt = gt_dict[key]
    else:
        gt = []

    if key in dt_dict:
        dt = dt_dict[key]
    else:
        dt = []

    if len(gt) == 0 and len(dt) == 0:
        return None

    for g in gt:
        if g["area"] < aRng[0] or g["area"] > aRng[1]:
            g["_ignore"] = 1
        else:
            g["_ignore"] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = [int(o["iscrowd"]) for o in gt]
    # load computed ious
    ious = ious[(imgId, catId)][:, gtind] if len(ious[(imgId, catId)]) > 0 else ious[(imgId, catId)]

    T = len(iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g["_ignore"] for g in gt])
    dtIg = np.zeros((T, D))
    if not len(ious) == 0:
        for tind, t in enumerate(iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[(dind, gind)] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[(dind, gind)]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]["id"]
                gtm[tind, m] = d["id"]
    # set unmatched detections outside of area range to ignore
    a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
    # store results for given image and category
    return {
        "image_id": imgId,
        "category_id": catId,
        "aRng": aRng,
        "maxDet": maxDet,
        "dtIds": [d["id"] for d in dt],
        "gtIds": [g["id"] for g in gt],
        "dtMatches": dtm,
        "gtMatches": gtm,
        "dtScores": [d["score"] for d in dt],
        "gtIgnore": gtIg,
        "dtIgnore": dtIg,
    }


def accumulate(evalImgs, imgIds, catIds, iouThrs, recThrs, areaRng, maxDets):
    """
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    """
    # print('Accumulating evaluation results...')

    # allows input customized parameters

    T = len(iouThrs)
    R = len(recThrs)
    K = len(catIds)
    A = len(areaRng)
    M = len(maxDets)

    precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    recall = -np.ones((T, K, A, M))
    scores = -np.ones((T, R, K, A, M))

    # create dictionary for future indexing
    catIds = catIds
    setK = set(catIds)
    setA = set(map(tuple, areaRng))
    setM = set(maxDets)
    setI = set(imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(catIds) if k in setK]
    m_list = [m for n, m in enumerate(maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(imgIds) if i in setI]
    I0 = len(imgIds)
    A0 = len(areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        Nk = k0 * A0 * I0
        for a, a0 in enumerate(a_list):
            Na = a0 * I0
            for m, maxDet in enumerate(m_list):
                E = [evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind="mergesort")
                dtScoresSorted = dtScores[inds]

                dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e["gtIgnore"] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    q = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t, k, a, m] = rc[-1]
                    else:
                        recall[t, k, a, m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    inds = np.searchsorted(rc, recThrs, side="left")
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except Exception:
                        pass
                    precision[t, :, k, a, m] = np.array(q)
                    scores[t, :, k, a, m] = np.array(ss)
    eval_dict = {
        "counts": [T, R, K, A, M],
        # 'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "precision": precision,
        "recall": recall,
        "scores": scores,
    }

    return eval_dict


def targets_to_ann_list(images, targets):
    ann_list = []
    ann_id = 0

    img, targets = images[0], targets[0]
    image_id = targets["image_id"].item()
    img_dict = {}
    img_dict["id"] = image_id
    img_dict["height"] = img.shape[-2]
    img_dict["width"] = img.shape[-1]
    # dataset['images'].append(img_dict)
    bboxes = targets["boxes"]
    bboxes[:, 2:] -= bboxes[:, :2]
    bboxes = bboxes.tolist()
    labels = targets["labels"].tolist()
    areas = targets["area"].tolist()
    iscrowd = targets["iscrowd"].tolist()
    if "masks" in targets:
        masks = targets["masks"]
        # make masks Fortran contiguous for coco_mask
        masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if "keypoints" in targets:
        keypoints = targets["keypoints"]
        keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
    num_objs = len(bboxes)
    for i in range(num_objs):
        ann = {}
        ann["image_id"] = image_id
        ann["bbox"] = bboxes[i]
        ann["category_id"] = labels[i]
        # categories.add(labels[i])
        ann["score"] = 1
        ann["area"] = areas[i]
        ann["iscrowd"] = iscrowd[i]
        ann["id"] = ann_id
        if "masks" in targets:
            ann["segmentation"] = mask_util.encode(masks[i].numpy())
        if "keypoints" in targets:
            ann["keypoints"] = keypoints[i]
            ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])

        ann_list += [ann]
        ann_id += 1

    return ann_list


def preds_to_ann_list(preds_dict, mask_void=None):
    ann_list = []
    for original_id, prediction in preds_dict.items():
        if len(prediction) == 0:
            continue

        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks"]

        masks = masks > 0.5

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        rles = []
        for mask in masks:
            binmask = np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
            if mask_void is not None:
                ind = mask_void[0][:, :, None].numpy() == 1
                binmask[ind] = 0
            rle = mask_util.encode(binmask)[0]
            rles += [rle]

        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        boxes = prediction["boxes"]
        boxes = xyxy_to_xywh(boxes).tolist()

        ann_list.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": boxes[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return ann_list


def targets_to_ann_list(images, targets):
    ann_list = []
    ann_id = 0

    img, targets = images[0], targets[0]
    image_id = targets["image_id"].item()
    img_dict = {}
    img_dict["id"] = image_id
    img_dict["height"] = img.shape[-2]
    img_dict["width"] = img.shape[-1]
    # dataset['images'].append(img_dict)
    bboxes = targets["boxes"]
    bboxes[:, 2:] -= bboxes[:, :2]
    bboxes = bboxes.tolist()
    labels = targets["labels"].tolist()
    areas = targets["area"].tolist()
    iscrowd = targets["iscrowd"].tolist()
    if "masks" in targets:
        masks = targets["masks"]
        # make masks Fortran contiguous for coco_mask
        masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if "keypoints" in targets:
        keypoints = targets["keypoints"]
        keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
    num_objs = len(bboxes)
    for i in range(num_objs):
        ann = {}
        ann["image_id"] = image_id
        ann["bbox"] = bboxes[i]
        ann["category_id"] = labels[i]
        # categories.add(labels[i])
        ann["score"] = 1
        ann["area"] = areas[i]
        ann["iscrowd"] = iscrowd[i]
        ann["id"] = ann_id
        if "masks" in targets:
            ann["segmentation"] = mask_util.encode(masks[i].numpy())
        if "keypoints" in targets:
            ann["keypoints"] = keypoints[i]
            ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])

        ann_list += [ann]
        ann_id += 1

    return ann_list


def bbox_yxyx_to_ann_list(bbox_yxyx, image_id=None, category_list=None, score_list=None):
    bbox_xywh = yxyx_to_xywh(bbox_yxyx)

    return bbox_to_ann_list(bbox_xywh, image_id=image_id, category_list=category_list, score_list=score_list)


def bbox_xyxy_to_ann_list(bbox_xyxy, image_id=None, category_list=None, score_list=None):
    bbox_xywh = xyxy_to_xywh(bbox_xyxy)

    return bbox_to_ann_list(bbox_xywh, image_id=image_id, category_list=category_list, score_list=score_list)


def bbox_to_ann_list(bbox_xywh, image_id=None, category_list=None, score_list=None):
    ann_list = []
    _, d = bbox_xywh.shape
    for i in range(bbox_xywh.shape[0]):
        bbox = bbox_xywh[i]
        x, y, w, h = bbox

        ann_list += [
            {
                "iscrowd": 0,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(h * w),
                "image_id": image_id or 0,
                "category_id": category_list or 1,
                "height": int(h),
                "width": int(w),
                "score": score_list or 1,
            }
        ]

    return ann_list


def yxyx_to_xywh(boxes):
    ymin, xmin, ymax, xmax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def xyxy_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
