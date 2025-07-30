import numpy as np

class AntiSpoofing_CM_Metrics:
    def __init__(self, predicted_logits, prediction, label):
        self.prediction_logits = predicted_logits[:, 1] # bonafide logits
        self.prediction = prediction[:, 1] # bonafide score
        self.label = label
        
        self.thresholdLimit = 500
        self.precision = 1/self.thresholdLimit
        
        TN, FP, FN, TP = self.calculate_confusion_matrices()
        
        self.FAR = [fp / (fp + tn) if (fp + tn) > 0 else 0 for fp, tn in zip(FP, TN)]
        self.FRR = [fn / (fn + tp) if (fn + tp) > 0 else 0 for fn, tp in zip(FN, TP)]

    def calculate_confusion_matrices(self):
        TN_list = []
        FP_list = []
        FN_list = []
        TP_list = []
        
        for threshold in range(1,self.thresholdLimit + 1):
            predicted_binary = (self.prediction >= threshold*self.precision).astype(int)
            
            TN_list.append(sum((predicted_binary == 0) & (self.label == 0)))
            FP_list.append(sum((predicted_binary == 1) & (self.label == 0)))
            FN_list.append(sum((predicted_binary == 0) & (self.label == 1)))
            TP_list.append(sum((predicted_binary == 1) & (self.label == 1)))
            
        return TN_list, FP_list, FN_list, TP_list

    def compute_EER(self):
        abs_diffs = np.abs(np.array(self.FRR) - np.array(self.FAR))
        min_index = np.argmin(abs_diffs)
        eer = np.mean((self.FRR[min_index], self.FAR[min_index]))
        return eer

    def compute_mindcf(self, Pspoof, Cmiss, Cfa):
        p_target = 1- Pspoof
        c_det = [
            Cmiss * frr * p_target + Cfa * far * (1 - p_target)
            for frr, far in zip(self.FRR, self.FAR)
        ]
        
        return min(c_det)

    def compute_actDCF(self, Pspoof, Cmiss, Cfa):
        bonafide_scores = self.prediction_logits[self.label == 1]
        spoof_scores = self.prediction_logits[self.label == 0]

        beta = Cmiss * (1 - Pspoof) / (Cfa * Pspoof)
        threshold = - np.log(beta)

        rate_miss = np.sum((bonafide_scores < threshold).astype(int)) / bonafide_scores.shape[0]
        rate_fa = np.sum((spoof_scores >= threshold).astype(int)) / spoof_scores.shape[0]

        act_dcf = Cmiss * (1 - Pspoof) * rate_miss + Cfa * Pspoof * rate_fa
        act_dcf = act_dcf / np.min([Cfa * Pspoof, Cmiss * (1 - Pspoof)])
        
        return act_dcf


class AntiSpoofing_SASV_Metrics:
    def __init__(self, cm_pred, cm_labels, as_pred, as_labels, sasv_pred, sasv_labels):
        self.thresholdLimit = 1000
        self.precision = 1/self.thresholdLimit
        
        cmtn, cmfp, cmfn, cmtp = self.calculate_confusion_matrices(cm_pred[:,1], cm_labels)
        astn, asfp, asfn, astp = self.calculate_confusion_matrices(as_pred, as_labels)
        sasvtn, sasvfp, sasvfn, sasvtp = self.calculate_confusion_matrices(sasv_pred, sasv_labels)
        
        self.cm_far = [fp / (fp + tn) if (fp + tn) > 0 else 0 for fp, tn in zip(cmfp, cmtn)]
        self.cm_frr = [fn / (fn + tp) if (fn + tp) > 0 else 0 for fn, tp in zip(cmfn, cmtp)]
        
        self.as_far = [fp / (fp + tn) if (fp + tn) > 0 else 0 for fp, tn in zip(asfp, astn)]
        self.as_frr = [fn / (fn + tp) if (fn + tp) > 0 else 0 for fn, tp in zip(asfn, astp)]
        
        self.sasv_far = [fp / (fp + tn) if (fp + tn) > 0 else 0 for fp, tn in zip(sasvfp, sasvtn)]
        self.sasv_frr = [fn / (fn + tp) if (fn + tp) > 0 else 0 for fn, tp in zip(sasvfn, sasvtp)]

    def calculate_confusion_matrices(self, prediction, label):
        TN_list = []
        FP_list = []
        FN_list = []
        TP_list = []
        
        for threshold in range(1,self.thresholdLimit + 1):
            predicted_binary = (prediction >= threshold*self.precision).astype(int)
            
            TN_list.append(sum((predicted_binary == 0) & (label == 0)))
            FP_list.append(sum((predicted_binary == 1) & (label == 0)))
            FN_list.append(sum((predicted_binary == 0) & (label == 1)))
            TP_list.append(sum((predicted_binary == 1) & (label == 1)))
            
        return TN_list, FP_list, FN_list, TP_list
    
    def compute_aDCF(self, cmiss, cfanon, cfaspf, ptar, pnon, pspf):
        aDCF_list = [cmiss * ptar * self.as_frr[i] + cfanon * pnon * self.as_far[i] + cfaspf * pspf * self.cm_far[i] for i in range(self.thresholdLimit)]
        return min(aDCF_list)
    
    def compute_tDCF(self, cmiss, cfanon, cfaspf, ptar, pnon, pspf):
        cspoof = [cfaspf*pspf*self.cm_far[i] for i in range(self.thresholdLimit)]
        cverif = [(cmiss*ptar*self.as_frr[i]) + (cfanon*pnon*self.as_far[i]) for i in range(self.thresholdLimit)]
        
        tdcf = [cverif[i] * (1 - self.cm_far[i]) + cspoof[i] for i in range(self.thresholdLimit)]
        return min(tdcf)
    
    def compute_tEER(self):
        abs_diffs = np.abs(np.array(self.sasv_frr) - np.array(self.sasv_far))
        min_index = np.argmin(abs_diffs)
        return np.mean((self.sasv_frr[min_index], self.sasv_far[min_index]))
