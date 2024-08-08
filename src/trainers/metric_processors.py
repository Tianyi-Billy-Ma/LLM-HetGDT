import sys
import logging
from easydict import EasyDict
from sklearn import metrics
from imblearn import metrics as im_metrics

sys.dont_write_bytecode = True
logger = logging.getLogger(__name__)


class MetricsProcessor:
    def __init__(self) -> None:
        pass

    def compute_metrics(self, data_dict):
        """
        Compute metrics
        """
        log_dict = EasyDict(
            {
                "metrics": {},
                "artifacts": {},
            }
        )
        for metrics in self.config.metrics:
            compute_func = getattr(self, metrics.name)
            logger.info(f"Running metrics {str(metrics)}...")
            log_dict = compute_func(metrics, data_dict, log_dict)
            # print(f"Metrics columns {log_dict.metrics.keys()} ")

        return log_dict

    def compute_classification(self, module, data_dict, log_dict):
        y_true = data_dict.y_true
        y_pred = data_dict.y_pred

        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_weighted = metrics.f1_score(y_true, y_pred, average="weighted")
        f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
        precision = metrics.precision_score(y_true, y_pred, average="weighted")
        recall = metrics.recall_score(y_true, y_pred, average="weighted")

        gmean = im_metrics.geometric_mean_score(y_true, y_pred, average="macro")

        log_dict.metrics["accuracy"] = accuracy
        log_dict.metrics["f1_weighted"] = f1_weighted
        log_dict.metrics["precision"] = precision
        log_dict.metrics["recall"] = recall
        log_dict.metrics["f1_macro"] = f1_macro
        log_dict.metrics["gmean"] = gmean

        return log_dict
