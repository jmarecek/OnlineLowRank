// Copyright 2018 IBM.

// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root 
// directory of this source tree or at 
// http://www.apache.org/licenses/LICENSE-2.0.
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice 
// indicating that they have been altered from the originals.

// If you use this code, please cite our paper:
// @inproceedings{akhriev2018pursuit,
//  title={Pursuit of low-rank models of time-varying matrices robust to sparse and measurement noise},
//  author={Akhriev, Albert and Marecek, Jakub and Simonetto, Andrea},
//  note={arXiv preprint arXiv:1809.03550},
//  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
//  volume={34},
//  year={2020}
// }

// IBM-Review-Requirement: Art30.3
// Please note that the following code was developed for the project VaVeL at
// IBM Research -- Ireland, funded by the European Union under the
// Horizon 2020 Program.
// The project started on December 1st, 2015 and was completed by December 1st,
// 2018. Thus, in accordance with Article 30.3 of the Multi-Beneficiary General
// Model Grant Agreement of the Program, certain limitations are in force up
// to December 1st, 2022. For further details please contact Jakub Marecek
// (jakub.marecek@ie.ibm.com) or Gal Weiss (wgal@ie.ibm.com).

#include "stdafx.h"
#include "utils.h"
#include "config_file.h"
#include "options.h"
#include "i_metrics.h"
#include "i_background.h"
#include "i_low_rank_solver.h"
#include "i_lp_solver.h"
#include "i_context.h"
#include "i_motion_mask.h"

namespace {

//=============================================================================
// Implementation of background model estimator.
//=============================================================================
class Background : public IBackground
{
public:
//-----------------------------------------------------------------------------
// Constructor.
//-----------------------------------------------------------------------------
explicit Background(const IContext & ctx)
    : m_ctx(ctx) , m_lr_solv() , m_lp_solv() , m_motion_mask() , m_moving_avg()
    , m_frame_count(0) , m_background() , m_history() , m_history_length(0) , m_rank(0) , m_size(0,0)
    , m_elem_size(0) , m_verbose(ctx.getOptions().verbose) {
    m_history_length = ctx.getConfig().asUInt("background.history_length");
    m_rank = ctx.getConfig().asInt("background.rank");

    std::string method = ctx.getConfig().asStr("lp-solver.method");
    if (method == "L1") { m_lp_solv = CreateLpSolverL1(ctx);
//    } else if (method == "Linf") {
//        m_lp_solv = CreateSolverLinCombRobustEx(ctx);
//    } else if (method == "Robust") {
//        m_lp_solv = CreateSolverLinCombRobust(ctx);
    } else { MY_RUNTIME_ERR("unknown method: " + method); }

//    m_lr_solv = CreateLowRankSolver5(ctx);
    m_lr_solv = CreateLowRankSolverTuned(ctx);

    m_motion_mask = CreateMotionMaskGenerator(ctx);
}

//-----------------------------------------------------------------------------
// Destructor.
//-----------------------------------------------------------------------------
virtual ~Background() { }

//-----------------------------------------------------------------------------
// Function updates the current background estimation given a new image
// and motion mask (0 - foreground pixel, > 0 - background pixel). Note,
// the new image and mask can be ignored if computation had not been
// finished since the last call to this function. This is a way to split
// up a long job into incremental phases over the time line.
//-----------------------------------------------------------------------------
virtual cv::Mat Update(const cv::Mat & img, const cv::Mat & roi) {
    cv::Mat mask;

    // Check the input images.
    MY_VERIFY(IsContinuousColorImage(img) && IsSameLayout(img, roi), "24-bit continuous color images are expected on input");

    // Before any further processing the full history must be accumulated.
    if (m_frame_count < m_history_length) { CopyImageToHistory(img, roi, mask); return mask; }

    // Compute the motion mask if background estimation is already available.
    if (!m_background.empty()) { mask = m_motion_mask->ComputeMask(m_background, img, roi); }

    // Update low-rank approximation of history matrix. If the updating
    // procedure had been finished (m_lr_solv->Update() returned "true"),
    // then we proceed inserting a new frame in the list of recent frames;
    // otherwise we return from here and finalize low-rank background
    // approximation next time this function is invoked.
    if (!m_lr_solv->Update( m_history.data(), m_history.data() + m_history.size(),
            Begin(roi), End(roi), // <-- actually these pointers are not used
            m_moving_avg.data(), m_moving_avg.data() + m_moving_avg.size()))
        return mask;

    // If we want to collect the optimization statistics, this must be
    // run right AFTER the regular low-rank solver.
    if (m_ctx.isMetricsEnabled()) { m_ctx.getMetrics().RunOptimStatistics(m_history.data(), m_history.data() + m_history.size()); }

    // Replace the oldest image in the history.
    CopyImageToHistory(img, roi, mask);

    // Low-rank decomposition requires several epochs to converge (recall,
    // m_lr_solv->Update() returns after just one epoch). As such, we postpone
    // background estimation until frame counter is larger than history length.
//#if 1
#warning "Skip frames = 5"
    if (m_frame_count < m_history_length + 5)
//#else
//    if (m_frame_count < m_history_length + 10)
//#endif
        return mask;

    EstimateBackground(img);
    return mask;
}

//-----------------------------------------------------------------------------
// Function replaces the oldest image in the history and increments frame
// counter. It also combines the input image with ROI mask. Important: function
// imposes the lower bound on color pixel value - (1,1,1). The special pixel
// color (0,0,0) is used to mark foreground (moving) points. If pixel is "not
// interesting" (roi(x,y) == 0), it will be set to "black" color (1,1,1).
//-----------------------------------------------------------------------------
virtual void CopyImageToHistory(const cv::Mat & img, const cv::Mat & roi, const cv::Mat & mask) {
//#if 1

    const size_t nbytes = SizeInBytes(img);
    const size_t offset = (m_frame_count % m_history_length) * nbytes;

    // Resize history and create low-rank solver the first time.
    // Initially we fill up the background by the first image.
    {   const size_t hist_size = m_history_length * nbytes;
        if (m_history.empty()) { m_size = img.size();
            m_elem_size = static_cast<int>(img.elemSize());
            m_history.resize(hist_size);
        }
        MY_ASSERT(m_history.size() == hist_size); }

    // Maintain pixel-wise moving average of image sequence.
    MaintainMovingAverage(Begin(img), End(img), roi);

    const ubyte * pImg = Begin(img);
    const ubyte * pRoi = Begin(roi);
          ubyte * pHis = m_history.data() + (long)offset;

    if (mask.empty()) {
        for (size_t k = 0; k < nbytes; ++k) {
            if (pRoi[k] == 0) { pHis[k] = ubyte(BLACK_POINT); }
            else { pHis[k] = std::max(pImg[k], ubyte(BLACK_POINT)); }
        }
    } else {
        MY_ASSERT(IsSameLayout(img, mask));
        const ubyte * pMsk = Begin(mask);

        for (size_t k = 0; k < nbytes; ++k) {
            if (pRoi[k] == 0) { pHis[k] = ubyte(BLACK_POINT); }
            else if (pMsk[k] > 0) { pHis[k] = ubyte(MOVING_POINT); }
            else { pHis[k] = std::max(pImg[k], ubyte(BLACK_POINT)); }
        }
    }
    ++m_frame_count;

//#else
//#warning "E X P E R I M E N T A L: masking by means of optic flow"
//
//    // Create optic flow object if not ready yet.
//    if (m_optflow.empty()) {
//#if (CV_VERSION_MAJOR >= 4)
//        m_optflow = cv::DISOpticalFlow::create(
//                                cv::DISOpticalFlow::PRESET_MEDIUM);
//#else
//        m_optflow = cv::optflow::createOptFlow_DIS(
//                                cv::optflow::DISOpticalFlow::PRESET_MEDIUM);
//#endif
//        cv::cvtColor(img, m_curr_gray_img, cv::COLOR_BGR2GRAY);
//        m_velocity_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC2);
//        m_flow_squared.resize(img.total());
//    }
//
//    const size_t npixels = img.total();
//    const size_t nbytes = SizeInBytes(img);
//    const size_t offset = (m_frame_count % m_history_length) * nbytes;
//    float        sigma2 = 1.0f;
//
//    // Resize history and create low-rank solver the first time.
//    // Initially we fill up the background by the first image.
//    {
//        const size_t hist_size = m_history_length * nbytes;
//        if (m_history.empty()) {
//            m_frame_count = 0;
//            m_size = img.size();
//            m_elem_size = static_cast<int>(img.elemSize());
//            m_history.resize(hist_size);
//        }
//        MY_ASSERT(m_history.size() == hist_size);
//    }
//
//    // Maintain pixel-wise moving average of image sequence.
//    MaintainMovingAverage(Begin(img), End(img), roi);
//
//    // Compute optic flow and its squared magnitude at every point.
//    if (!m_prev_gray_img.empty()) {
//        m_optflow->calc(m_prev_gray_img, m_curr_gray_img, m_velocity_img);
//        size_t k = 0;
//        for (int r = 0; r < m_velocity_img.rows; ++r) {
//        for (int c = 0; c < m_velocity_img.cols; ++c) {
//            const cv::Point2f & v = m_velocity_img.at<cv::Point2f>(r,c);
//            m_flow_squared[k++] = v.x * v.x + v.y * v.y;
//        }}
//        MY_ASSERT(k == m_flow_squared.size());
//        std::nth_element(m_flow_squared.begin(),
//                         m_flow_squared.begin() + (npixels/2),
//                         m_flow_squared.end());
//        float median = m_flow_squared[npixels/2];
//        sigma2 = std::max(median, 0.5f/25.0f);
//    }
//
//    const ubyte * pImg = Begin(img);
//    const ubyte * pRoi = Begin(roi);
//          ubyte * pHis = m_history.data() + (long)offset;
//
//    if (m_prev_gray_img.empty() || mask.empty()) {
//        for (size_t k = 0; k < nbytes; ++k) {
//            if (pRoi[k] == 0) {
//                pHis[k] = ubyte(BLACK_POINT);
//            } else {
//                pHis[k] = std::max(pImg[k], ubyte(BLACK_POINT));
//            }
//        }
//    } else {
//        const unsigned bpp = (unsigned)m_elem_size;
//        MY_ASSERT(bpp * npixels == nbytes);
//        MY_ASSERT(IsSameLayout(img, mask));
//        const ubyte * pMsk = Begin(mask);
//
//        for (size_t k = 0; k < nbytes; ++k) {
//            if (pRoi[k] == 0) {
//                pHis[k] = ubyte(BLACK_POINT);
//            } else if (pMsk[k] > 0) {
//                pHis[k] = ubyte(MOVING_POINT);
//            } else {
//                float w = std::exp(-m_flow_squared[k/bpp] / (2.0f * sigma2));
//                pHis[k] = cv::saturate_cast<ubyte>(
//                        RoundInt(static_cast<float>(pHis[k]) * (1.0f - w) +
//                                 static_cast<float>(pImg[k]) * w));
//                pHis[k] = std::max(pHis[k], ubyte(BLACK_POINT));
//            }
//        }
//    }
//    m_curr_gray_img.copyTo(m_prev_gray_img);
//    ++m_frame_count;
//
//#endif
}

//-----------------------------------------------------------------------------
// Function computes the best background estimation from the low-rank
// approximation.
//-----------------------------------------------------------------------------
virtual void EstimateBackground(const cv::Mat & img) {
    const bool timeit = (m_verbose > 0);
    time_point_t start_time = timeit ? Tic() : time_point_t();

    const size_t      rank = static_cast<size_t>(m_rank);
    const size_t      nbytes = SizeInBytes(img);
    const ubyte   * pImg = img.ptr<ubyte>();
    const flt_arr_t & R = m_lr_solv->getR();

    const flt_arr_t * y = m_lp_solv->Solve(R, pImg, pImg + (long)nbytes);
    if (y == nullptr) return;
    MY_ASSERT(y->size() == rank);

    if (m_background.empty()) img.copyTo(m_background);
    MY_ASSERT(IsSameLayout(m_background, img));

    ubyte * backgr = Begin(m_background);
    for (size_t c = 0; c < nbytes; ++c) { double yR = 0.0;
        for (size_t r = 0; r < rank; ++r) { yR += (*y)[r] * R[r + c * rank]; }
        backgr[c] = cv::saturate_cast<ubyte>(yR);
    }

    // Metrics: apply different sub-sampling rates for L1 and Linf
    // projectors of the current image onto low-rank background model.
    if (m_ctx.isMetricsEnabled()) { m_ctx.getMetrics().RunSubSampling(R, pImg, pImg + (long)nbytes); }

    if (timeit) LOG(INFO) << "Background formation time: " << Toc(start_time);
}

//-----------------------------------------------------------------------------
// Function maintains point-wise moving average of image sequence.
//-----------------------------------------------------------------------------
virtual void MaintainMovingAverage(const ubyte * X, const ubyte * X_end, const cv::Mat & roi) {
    const size_t n = static_cast<size_t>(std::distance(X, X_end));
    const float alpha = 1.0f / static_cast<float>(m_history_length);

    MY_ASSERT(SizeInBytes(roi) == n);
    if (m_moving_avg.empty()) {
        m_moving_avg.resize(n);
        std::transform(X, X_end, Begin(roi), m_moving_avg.begin(),
            [](ubyte x, ubyte r) -> float { return ((r == 0) ? static_cast<float>(BLACK_POINT) : static_cast<float>(x)); }
        );
    }
    MY_ASSERT(m_moving_avg.size() == n);

    const ubyte * r = Begin(roi);
    for (size_t i = 0; i < n; ++i) {
        float & ma_i = m_moving_avg[i];
        if (r[i] == 0) { ma_i = static_cast<float>(BLACK_POINT); }
        else { ma_i += alpha * (static_cast<float>(X[i]) - ma_i); }
    }
}

//-----------------------------------------------------------------------------
// Function returns so far estimated background as an image. Note, the
// background can remain the same for quite a long time, if the function
// Update() is slowly progressing.
//-----------------------------------------------------------------------------
virtual cv::Mat asImage() const { return m_background; }

//-----------------------------------------------------------------------------
// Function returns one of the background model, by its index, from
// the low-rank approximation. The background is converted to image.
//-----------------------------------------------------------------------------
virtual cv::Mat & asImage(int index, cv::Mat & img) const {
    MY_ASSERT((unsigned)index < m_history_length);
    MY_ASSERT(!m_size.empty());
    if ((img.size() != m_size) || (img.type() != CV_8UC3)) { img = cv::Mat::zeros(m_size, CV_8UC3); }

    const size_t nbytes = SizeInBytes(img);
    const size_t offset = (size_t)index * nbytes;
    flt_arr_t    A;

    m_lr_solv->getApproximation(A);
    if (A.empty()) { return img; }      // not ready yet
    MY_ASSERT(A.size() == nbytes * m_history_length);

    ubyte * pImg = Begin(img);
    for (size_t k = 0; k < nbytes; ++k) { pImg[k] = cv::saturate_cast<ubyte>(A[offset + k]); }
    MY_ASSERT(pImg == End(img));
    return img;
}

//-----------------------------------------------------------------------------
// Function returns one of the image stored in the recent history.
//-----------------------------------------------------------------------------
virtual const cv::Mat getHistory(int index) const
{
    MY_ASSERT((unsigned)index < m_history_length);
    MY_ASSERT(!m_size.empty() && !m_history.empty());
    const size_t nbytes = m_history.size() / m_history_length;
    const ubyte * p = m_history.data() + index * (long)nbytes;
    return cv::Mat(m_size, CV_8UC3, const_cast<void*>(static_cast<const void*>(p)));
}

//-----------------------------------------------------------------------------
// Function returns a point-wise moving-average image of type CV_32FC3;
// if moving average is not available, an empty image is returned.
//-----------------------------------------------------------------------------
virtual const cv::Mat getMovingAverage() const {
    if (m_moving_avg.size() == (size_t)(m_elem_size * m_size.width * m_size.height)) {
        return cv::Mat(m_size, CV_32FC3, const_cast<void*>(static_cast<const void*>(m_moving_avg.data())));
    } else { return cv::Mat(); }
}

private:
    typedef std::unique_ptr<ILowRankSolver> lr_solver_t;
    typedef std::unique_ptr<ILpSolver>      lp_solver_t;
    typedef std::unique_ptr<IMotionMask>    motion_mask_t;

    const IContext & m_ctx;            // context of application or thread
    lr_solver_t      m_lr_solv;        // low-rank solver
    lp_solver_t      m_lp_solv;        // linear-programming solver
    motion_mask_t    m_motion_mask;    // object estimating the motion mask
    flt_arr_t        m_moving_avg;     // moving average background
    uint64_t         m_frame_count;    // counter of frames inserted in history
    cv::Mat          m_background;     // current background estimation
    ubyte_arr_t      m_history;        // recent frames stacked as vectors
    unsigned         m_history_length; // number of recent images in history
    int              m_rank;           // rank of background approximation
    cv::Size         m_size;           // image size
    int              m_elem_size;      // element size, e.g. bytes per point
    int              m_verbose;        // level of verbosity

    // Experimental: masking by means of optic flow.
    cv::Ptr<cv::DenseOpticalFlow> m_optflow;
    cv::Mat                       m_prev_gray_img, m_curr_gray_img, m_velocity_img;
    std::vector<float>            m_flow_squared;

};  // class Background

}   // anonymous namespace

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
std::unique_ptr<IBackground> CreateBackground(const IContext & ctx) {
    return std::unique_ptr<IBackground>(new Background(ctx));
}


