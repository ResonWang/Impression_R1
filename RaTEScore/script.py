# from RaTEScore import scorer
# from RaTEScore.scorer import RaTEScore
from score import *


pred_report = ['There are no intracranial hemorrhages.',
              'The musculature and soft tissues are intact.']

gt_report = ['There is no finding to suggest intracranial hemorrhage.',
            'The muscle compartments are intact.']
#
# assert len(pred_report) == len(gt_report)
#
# ratescore = RaTEScore()
# # Add visualization_path here if you want to save the visualization result
# # ratescore = RaTEScore(visualization_path = '', , affinity_matrix='short') # affinity_matrix='long' for long paragraph evaluation
#
# scores = ratescore.compute_score(pred_report, gt_report)