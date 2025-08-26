#ifndef OC_SORT_CPP_OCSORT_HPP
#define OC_SORT_CPP_OCSORT_HPP
#include <functional>
#include <unordered_map>

#include "Association.hpp"
#include "KalmanBoxTracker.hpp"
#include "lapjv.hpp"
namespace ocsort
{

// This is a C++ implementation of the OC-SORT multi-object tracking algorithm.
// It also supports ReId embeddings for association (like DeepOCSort) if provided.
class OCSort
{
  public:
  OCSort(float det_thresh_, int max_age_ = 30, int min_hits_ = 3,
      float iou_threshold_ = 0.3, int delta_t_ = 3, std::string asso_func_ = "iou",
      float inertia_ = 0.2, float w_assoc_emb = 0.75, float alpha_fixed_emb = 0.95,
      int max_id = 0, bool aw_off = false, float aw_param = 0.5, bool cmc_off = true);

  /* Takes a numpy array of detections and embeddings, and returns a numpy array of tracking results.
  Params:
  - dets - a (N,6) numpy array of detections in the format [[x1,y1,x2,y2,score,class_id],[x1,y1,x2,y2,score,class_id],...],
  where N is the number of detections.

  - embs - a (N,emb_len) numpy array of corresponding ReId embeddings for each detections in the format [[emb1],[emb2],...],
  where N is the number of detections and emb_len is the length of the embedding.
  - Order of the embeddings must match the order of the detections!
  - The embeddings are optional, and can be empty if not used.
  - If embs is not empty, then the tracker will use the embeddings for association.

  Return:
  - The function returns a numpy array of tracking results in the format
  [[x1,y1,x2,y2,object_id,class_id,score,latest_detection_id],...]
  where object_id is the ID assigned to the object by the tracker,
  and latest_detection_id is the index (in dets) of the latest detection that was associated with this object
  (to get corresponding detection use dets.row(latest_detection_id)).

  Requires:
  - This method must be called once for each frame even with empty detections (use np.empty((0, 6)) for frames without detections).
  - NOTE: The number of objects returned may differ from the number of detections provided.
  */
  std::vector<Eigen::RowVectorXf> update(Eigen::MatrixXf dets,
      Eigen::MatrixXf embs = Eigen::MatrixXf(),
      Eigen::Matrix<float, 2, 3> cmc_transform = Eigen::Matrix<float, 2, 3>::Identity());

  public:
  float det_thresh;
  int max_age;
  int min_hits;
  float iou_threshold;
  int delta_t;
  std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)> asso_func;
  float inertia;
  float w_assoc_emb;
  float alpha_fixed_emb;
  bool aw_off;
  float aw_param;
  // Used to store KalmanBoxTracker
  std::vector<KalmanBoxTracker> trackers;
  int frame_count;
  bool cmc_off;
};

} // namespace ocsort
#endif // OC_SORT_CPP_OCSORT_HPP
