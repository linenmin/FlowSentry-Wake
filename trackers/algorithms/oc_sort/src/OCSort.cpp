#include "../include/OCSort.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include "iomanip"

namespace ocsort
{
template <typename Matrix>
std::ostream &
operator<<(std::ostream &os, const std::vector<Matrix> &v)
{
  os << "{";
  for (auto it = v.begin(); it != v.end(); ++it) {
    os << "(" << *it << ")\n";
    if (it != v.end() - 1)
      os << ",";
  }
  os << "}\n";
  return os;
}

OCSort::OCSort(float det_thresh_, int max_age_, int min_hits_, float iou_threshold_,
    int delta_t_, std::string asso_func_, float inertia_, float w_assoc_emb_,
    float alpha_fixed_emb_, int max_id_, bool aw_off_, float aw_param_, bool cmc_off_)
{
  cmc_off = cmc_off_;
  max_age = max_age_;
  min_hits = min_hits_;
  iou_threshold = iou_threshold_;
  trackers.clear();
  frame_count = 0;
  det_thresh = det_thresh_;
  delta_t = delta_t_;
  aw_off = aw_off_;
  aw_param = aw_param_;
  std::unordered_map<std::string, std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)>> ASSO_FUNCS{
    { "iou", iou_batch }, { "giou", giou_batch }
  };
  std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)> asso_func
      = ASSO_FUNCS[asso_func_];
  inertia = inertia_;
  w_assoc_emb = w_assoc_emb_;
  alpha_fixed_emb = alpha_fixed_emb_;

  assert(max_id_ >= 0);
  KalmanBoxTracker::max_id = max_id_;
}
std::ostream &
precision(std::ostream &os)
{
  os << std::fixed << std::setprecision(2);
  return os;
}
std::vector<Eigen::RowVectorXf>
OCSort::update(Eigen::MatrixXf dets, Eigen::MatrixXf embs, Eigen::Matrix<float, 2, 3> cmc_transform)
{
  bool use_reid = false;
  if (embs.size() > 0) {
    use_reid = true;

    if (embs.rows() != dets.rows()) {
      std::string error_msg("DeepOCSort tracker: Error: The number of detections and embeddings do not match.");
      throw std::runtime_error(error_msg);
    }

    // Normalize each row of embs
    for (int i = 0; i < embs.rows(); ++i) {
      float magnitude = std::sqrt(embs.row(i).squaredNorm());
      if (magnitude > 0) {
        embs.row(i) /= magnitude;
      }
    }
  }

  frame_count += 1;
  Eigen::Matrix<float, Eigen::Dynamic, 4> xyxys = dets.leftCols(4);
  Eigen::Matrix<float, 1, Eigen::Dynamic> confs = dets.col(4);
  Eigen::Matrix<float, 1, Eigen::Dynamic> clss = dets.col(5);
  Eigen::MatrixXf output_results = dets;
  auto inds_low = confs.array() > 0.1;
  auto inds_high = confs.array() < det_thresh;
  auto inds_second = inds_low && inds_high;
  Eigen::Matrix<float, Eigen::Dynamic, 6> dets_second;
  Eigen::Matrix<bool, 1, Eigen::Dynamic> remain_inds = (confs.array() > det_thresh);
  Eigen::Matrix<float, Eigen::Dynamic, 6> dets_first;
  std::vector<int> map_dets_first_to_dets;
  Eigen::MatrixXf dets_embs = Eigen::MatrixXf::Zero(0, embs.cols());
  for (int i = 0; i < output_results.rows(); i++) {
    if (true == inds_second(i)) {
      dets_second.conservativeResize(dets_second.rows() + 1, Eigen::NoChange);
      dets_second.row(dets_second.rows() - 1) = output_results.row(i);
    }
    if (true == remain_inds(i)) {
      dets_first.conservativeResize(dets_first.rows() + 1, Eigen::NoChange);
      dets_first.row(dets_first.rows() - 1) = output_results.row(i);
      map_dets_first_to_dets.push_back(i);

      if (use_reid) {
        // Fill dets_embs with corresponding embeddings
        dets_embs.conservativeResize(dets_embs.rows() + 1, Eigen::NoChange);
        dets_embs.row(dets_embs.rows() - 1) = embs.row(i);
      }
    }
  }

  // Apply CMC affine correction if cmc is enabled
  if (!cmc_off) {
    for (auto &tracker : trackers) {
      tracker.apply_affine_correction(cmc_transform);
    }
  }

  // Dynamic Appearance, 3.3 from https://arxiv.org/pdf/2302.11813
  Eigen::VectorXf trust = (dets_first.col(4).array() - det_thresh) / (1 - det_thresh);
  float af = alpha_fixed_emb;
  Eigen::VectorXf dets_alpha = af + (1 - af) * (1 - trust.array());

  Eigen::MatrixXf trks = Eigen::MatrixXf::Zero(trackers.size(), 5);
  std::vector<int> to_del;
  std::vector<Eigen::RowVectorXf> ret;

  int emb_len = 0;
  if (use_reid && trackers.size() > 0)
    emb_len = trackers[0].get_emb().size();

  Eigen::MatrixXf trk_embs = Eigen::MatrixXf::Zero(trackers.size(), emb_len);
  for (int i = 0; i < trks.rows(); i++) {
    Eigen::RowVectorXf pos = trackers[i].predict();
    trks.row(i) << pos(0), pos(1), pos(2), pos(3), 0;

    if (use_reid)
      trk_embs.row(i) = trackers[i].get_emb().transpose();
  }
  Eigen::MatrixXf velocities = Eigen::MatrixXf::Zero(trackers.size(), 2);
  Eigen::MatrixXf last_boxes = Eigen::MatrixXf::Zero(trackers.size(), 5);
  Eigen::MatrixXf k_observations = Eigen::MatrixXf::Zero(trackers.size(), 5);
  for (int i = 0; i < trackers.size(); i++) {
    velocities.row(i) = trackers[i].velocity;
    last_boxes.row(i) = trackers[i].last_observation;
    k_observations.row(i)
        = k_previous_obs(trackers[i].observations, trackers[i].age, delta_t);
  }
  /////////////////////////
  ///  Step1 First round of association
  ////////////////////////
  // Perform IOU association associate()
  std::vector<Eigen::Matrix<int, 1, 2>> matched;
  std::vector<int> unmatched_dets;
  std::vector<int> unmatched_trks;
  auto result = associate(dets_first, trks, dets_embs, trk_embs, iou_threshold,
      velocities, k_observations, inertia, w_assoc_emb, aw_off, aw_param);
  matched = std::get<0>(result);
  unmatched_dets = std::get<1>(result);
  unmatched_trks = std::get<2>(result);
  // Update matched tracks
  for (auto m : matched) {
    Eigen::Matrix<float, 5, 1> tmp_bbox;
    tmp_bbox = dets_first.block<1, 5>(m(0), 0);
    trackers[m(1)].update(&(tmp_bbox), dets_first(m(0), 5), map_dets_first_to_dets[m(0)]);
    if (use_reid)
      trackers[m(1)].update_emb(dets_embs.row(m(0)), dets_alpha(m(0)));
  }

  ///////////////////////
  /// Step2 Second round of association by OCR to find lost tracks back
  ///////////////////////
  if (unmatched_dets.size() > 0 && unmatched_trks.size() > 0) {
    Eigen::MatrixXf left_dets(unmatched_dets.size(), 6);
    int inx_for_dets = 0;
    for (auto i : unmatched_dets) {
      left_dets.row(inx_for_dets++) = dets_first.row(i);
    }
    Eigen::MatrixXf left_trks(unmatched_trks.size(), last_boxes.cols());
    int indx_for_trk = 0;
    for (auto i : unmatched_trks) {
      left_trks.row(indx_for_trk++) = last_boxes.row(i);
    }
    Eigen::MatrixXf iou_left = giou_batch(left_dets, left_trks);
    if (iou_left.maxCoeff() > iou_threshold) {
      std::vector<std::vector<float>> iou_matrix(
          iou_left.rows(), std::vector<float>(iou_left.cols()));
      for (int i = 0; i < iou_left.rows(); i++) {
        for (int j = 0; j < iou_left.cols(); j++) {
          iou_matrix[i][j] = -iou_left(i, j);
        }
      }
      std::vector<int> rowsol, colsol;
      float MIN_cost = execLapjv(iou_matrix, rowsol, colsol, true, 0.01, true);
      std::vector<std::vector<int>> rematched_indices;
      for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol.at(i) >= 0) {
          rematched_indices.push_back({ colsol.at(rowsol.at(i)), rowsol.at(i) });
        }
      }
      // If still unmatched after reassignment, these need to be deleted
      std::vector<int> to_remove_det_indices;
      std::vector<int> to_remove_trk_indices;
      for (auto i : rematched_indices) {
        int det_ind = unmatched_dets[i.at(0)];
        int trk_ind = unmatched_trks[i.at(1)];
        if (iou_left(i.at(0), i.at(1)) < iou_threshold) {
          continue;
        }
        ////////////////////////////////
        ///  Step3  update status of second matched tracks
        ///////////////////////////////
        Eigen::Matrix<float, 5, 1> tmp_bbox;
        tmp_bbox = dets_first.block<1, 5>(det_ind, 0);
        trackers.at(trk_ind).update(
            &tmp_bbox, dets_first(det_ind, 5), map_dets_first_to_dets[det_ind]);
        if (use_reid)
          trackers.at(trk_ind).update_emb(dets_embs.row(det_ind), dets_alpha(det_ind));
        to_remove_det_indices.push_back(det_ind);
        to_remove_trk_indices.push_back(trk_ind);
      }
      std::vector<int> tmp_res(unmatched_dets.size());
      sort(unmatched_dets.begin(), unmatched_dets.end());
      sort(to_remove_det_indices.begin(), to_remove_det_indices.end());
      auto end = set_difference(unmatched_dets.begin(), unmatched_dets.end(),
          to_remove_det_indices.begin(), to_remove_det_indices.end(), tmp_res.begin());
      tmp_res.resize(end - tmp_res.begin());
      unmatched_dets = tmp_res;
      std::vector<int> tmp_res1(unmatched_trks.size());
      sort(unmatched_trks.begin(), unmatched_trks.end());
      sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());
      auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
          to_remove_trk_indices.begin(), to_remove_trk_indices.end(), tmp_res1.begin());
      tmp_res1.resize(end1 - tmp_res1.begin());
      unmatched_trks = tmp_res1;
    }
  }

  for (auto m : unmatched_trks) {
    trackers.at(m).update(nullptr, 0, -1);
  }
  ///////////////////////////////
  /// Step4 Initialize new tracks and remove expired tracks
  ///////////////////////////////
  /* Create and initialize new trackers for unmatched detections */
  for (int i : unmatched_dets) {
    Eigen::RowVectorXf tmp_bbox = dets_first.block(i, 0, 1, 5);
    int cls_ = int(dets_first(i, 5));
    Eigen::VectorXf det_emb_;
    if (use_reid)
      det_emb_ = dets_embs.row(i);

    KalmanBoxTracker trk = KalmanBoxTracker(
        tmp_bbox, det_emb_, cls_, map_dets_first_to_dets[i], delta_t);
    // Append newly created tracker to the end of trackers
    trackers.push_back(trk);
  }
  int tmp_i = trackers.size();
  for (int i = trackers.size() - 1; i >= 0; i--) {
    Eigen::Matrix<float, 1, 4> d;
    int last_observation_sum = trackers.at(i).last_observation.sum();
    if (last_observation_sum < 0) {
      d = trackers.at(i).get_state();
    } else {
      d = trackers.at(i).last_observation.block(0, 0, 1, 4);
    }
    if (trackers.at(i).time_since_update < 1
        && ((trackers.at(i).hit_streak >= min_hits) | (frame_count <= min_hits))) {
      Eigen::RowVectorXf tracking_res(8);
      tracking_res << d(0), d(1), d(2), d(3), trackers.at(i).id + 1,
          trackers.at(i).cls, trackers.at(i).conf, trackers.at(i).latest_detection_id;
      ret.push_back(tracking_res);
    }
    if (trackers.at(i).time_since_update > max_age) {
      trackers.at(i).release_id();
      trackers.erase(trackers.begin() + i);
    }
  }
  return ret;
}
} // namespace ocsort
