
import segmentation_models_pytorch.utils as segu


metrics = [segu.metrics.IoU(threshold=0.5), segu.metrics.Fscore()]