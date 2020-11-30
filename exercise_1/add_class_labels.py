import logging

import numpy as np

from gunpowder.array import Array
from gunpowder.batch_request import BatchRequest
from gunpowder import BatchFilter

logger = logging.getLogger(__name__)


class AddClassLabels(BatchFilter):
    def __init__(self, gt, target):
        self.gt = gt
        self.target = target

    def setup(self):
        self.provides(self.target, self.spec[self.gt])
        self.enable_autoskip()

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.gt] = request[self.target].copy()
        return deps

    def process(self, batch, request):
        spec = batch[self.gt].spec.copy()
        spec.dtype = np.int64
        batch[self.target] = Array(batch[self.gt].data.astype(np.int64), spec)
