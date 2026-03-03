"""ADARRI: Detecting spurious R-peaks in ECG for HRV analysis in the ICU.

Reference:
    Rebergen DJ, Nagaraj SB, Rosenthal ES, Bianchi MT, van Putten MJAM,
    Westover MB. ADARRI: a novel method to detect spurious R-peaks in the
    electrocardiogram for heart rate variability analysis in the intensive
    care unit. J Clin Monit Comput. 2018;32:53-61.
"""

__version__ = "1.0.0"

from .detector import classify_epoch_adrri, classify_epoch_rri, flag_identification
from .rri import compute_rri, compute_adrri, interpolate_rri
from .peak_detection import detect_r_peaks
