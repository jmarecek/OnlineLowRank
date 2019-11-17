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
#include "changedetection_net.h"
#include "i_metrics.h"
#include "i_context.h"
#include "i_motion_mask.h"

namespace {

const unsigned MAX_UBYTE = std::numeric_limits<ubyte>::max();

//-----------------------------------------------------------------------------
// Function returns residual at the point in question given the image of
// component-wise, absolute differences between two color images.
//-----------------------------------------------------------------------------
inline int Residual(const cv::Mat & diff_image, int r, int c)
{
    const cv::Vec3b & p = diff_image.at<cv::Vec3b>(r,c);
    return static_cast<int>((unsigned)p[0] + (unsigned)p[1] + (unsigned)p[2]);
}

//=============================================================================
// Implementation of IMotionMask interface.
//=============================================================================
class MotionMask : public IMotionMask
{
public:
//-----------------------------------------------------------------------------
// Constructor.
//-----------------------------------------------------------------------------
explicit MotionMask(const IContext & ctx)
    : m_ctx(ctx)
    , m_histogram()
    , m_motion_mask()
    , m_default_threshold(0)
    , m_noisy_points_fraction(0.0f)
    , m_element()
    , m_tmp_mask()
    , m_verbose(ctx.getOptions().verbose)
{
    static_assert((CDNET14_BLACK == 0) && (CDNET14_WHITE == MAX_UBYTE), "");
    m_default_threshold =
        3 * m_ctx.getConfig().asUInt("background.noise_delta");
    m_default_threshold = std::max(m_default_threshold, unsigned(3));
    m_noisy_points_fraction =
        m_ctx.getConfig().asFloat("background.noisy_points_fraction");
    MY_ASSERT(0.0 < m_noisy_points_fraction);
    MY_ASSERT(m_noisy_points_fraction < 0.5);

    const int morph_size = 1;
    m_element = cv::getStructuringElement(cv::MORPH_RECT,
                            cv::Size(2*morph_size + 1, 2*morph_size + 1));
}

//-----------------------------------------------------------------------------
// Destructor.
//-----------------------------------------------------------------------------
virtual ~MotionMask()
{
}

//-----------------------------------------------------------------------------
// Function computes motion mask based on background estimation, current
// image and, possibly, some accumulated history.
//-----------------------------------------------------------------------------
virtual cv::Mat ComputeMask(const cv::Mat & background,
                            const cv::Mat & img,
                            const cv::Mat & roi)
{
    MY_ASSERT(IsContinuousColorImage(background));
    MY_ASSERT(IsSameLayout(img, background) && IsSameLayout(img, roi));

    // Allocate the histogram, if needed.
    if (m_histogram.empty()) { m_histogram.resize(MAX_UBYTE + 1); }

    // Compute the image of residuals.
    cv::absdiff(background, img, m_motion_mask);
    std::transform(
            Begin(m_motion_mask), End(m_motion_mask), Begin(roi),
            Begin(m_motion_mask),
            [] (ubyte m, ubyte r) -> ubyte {
                return ((r == 0) ? ubyte(0) : m);
            }
    );

#if 0
    // Accumulate the histogram of residuals. Mind the 3-channel color image.
    std::fill(m_histogram.begin(), m_histogram.end(), int(0));
    for (int r = 0; r < m_motion_mask.rows; ++r) {
    for (int c = 0; c < m_motion_mask.cols; ++c) {
        cv::Vec3b & p = m_motion_mask.at<cv::Vec3b>(r,c);
        unsigned val = (unsigned)p[0] + (unsigned)p[1] + (unsigned)p[2];
        m_histogram[std::min(val, MAX_UBYTE)] += 1;
    }}

    // Compute approximate threshold.
    float threshold = 1.0f;
    {
        const int64_t NPIXELS = (int64_t)m_motion_mask.total();

        int64_t sum = 0;
        for (size_t k = 0; k <= MAX_UBYTE; ++k) {
            sum += m_histogram[k];
            if (2 * sum >= NPIXELS) {
                threshold = 5.0f * static_cast<float>(k);
                break;
            }
        }
    }
#warning "Temporary code"
threshold = gConfig.asFloat("background", "noise_delta");

    // Thresholding. Mind the 3-channel color image.
    {
        const cv::Vec3b BLACK = cv::Vec3b(0, 0, 0);
        const cv::Vec3b WHITE = cv::Vec3b(MAX_UBYTE, MAX_UBYTE, MAX_UBYTE);

        for (int r = 0; r < m_motion_mask.rows; ++r) {
        for (int c = 0; c < m_motion_mask.cols; ++c) {
            cv::Vec3b & p = m_motion_mask.at<cv::Vec3b>(r,c);
            unsigned val = (unsigned)p[0] + (unsigned)p[1] + (unsigned)p[2];
            m_motion_mask.at<cv::Vec3b>(r,c) =
                        (static_cast<float>(val) >= threshold) ? WHITE : BLACK;
        }}
    }
#else
    const unsigned threshold = ComputeThreshold(m_motion_mask);

    // Thresholding. Mind the 3-channel color image.
    {
        const cv::Vec3b BLACK = cv::Vec3b(CDNET14_BLACK,
                                          CDNET14_BLACK,
                                          CDNET14_BLACK);
        const cv::Vec3b WHITE = cv::Vec3b(CDNET14_WHITE,
                                          CDNET14_WHITE,
                                          CDNET14_WHITE);
        for (int r = 0; r < m_motion_mask.rows; ++r) {
        for (int c = 0; c < m_motion_mask.cols; ++c) {
            unsigned res = (unsigned)Residual(m_motion_mask, r, c);
            m_motion_mask.at<cv::Vec3b>(r, c) =
                                (res >= threshold) ? WHITE : BLACK;
        }}
    }
#endif

#if 1
    // Apply simple morphological operations to improve the motion mask.
    if (m_tmp_mask.empty()) m_motion_mask.copyTo(m_tmp_mask);
    MY_ASSERT(IsSameLayout(m_motion_mask, m_tmp_mask));
    cv::morphologyEx(m_motion_mask, m_tmp_mask, cv::MORPH_OPEN, m_element);
    cv::morphologyEx(m_tmp_mask, m_motion_mask, cv::MORPH_CLOSE, m_element);
#endif

    // Demo and statistics.
    if (m_ctx.isMetricsEnabled()) {
        m_ctx.getMetrics().SaveHistogramOfResiduals(m_histogram);
        m_ctx.getMetrics().CreateScaledResiduals(background, img, roi);
    }

    return m_motion_mask;
}

//-----------------------------------------------------------------------------
// Function computes the motion mask using several thresholding techniques.
//-----------------------------------------------------------------------------
virtual unsigned ComputeThreshold(const cv::Mat & diff_image)
{
    MY_ASSERT(diff_image.type() == CV_8UC3);

    const unsigned HISTO_LEN = 3*MAX_UBYTE + 1;

    // Histograms and cumulative function(s).
    uint64_t thr_histo[HISTO_LEN];      // histogram of thresholds
    unsigned res_histo[HISTO_LEN];      // histogram of residuals

    std::memset(thr_histo, 0, sizeof(thr_histo));
    std::memset(res_histo, 0, sizeof(res_histo));

    // 8 neighbours of a pixel.
    const int ri[8] = {1, 1, 0,-1,-1,-1, 0, 1};
    const int ci[8] = {0, 1, 1, 1, 0,-1,-1,-1};

    // .
    for (int r = 1; r < diff_image.rows - 1; ++r) {
    for (int c = 1; c < diff_image.cols - 1; ++c) {
        int p = Residual(diff_image, r, c);
        int count = 0, first = -1, second = -1, third = -1;
        for (int k = 0; k < 8; ++k) {
            int n = Residual(diff_image, r + ri[k], c + ci[k]);
            if (n >= p)
                ++count;
            if (count >= 2)
                break;
            if (n >= first) {
                third  = second;
                second = first;
                first  = n;
            } else if (n >= second) {
                third  = second;
                second = n;
            } else if (n >= third) {
                third = n;
            }
        }

        // Accumulate histogram of thresholds.
        if (count < 2) {
            int upto = (count == 0) ? first : second;
            for (int k = third + 1; k <= upto; ++k) {
                if ((unsigned)k < HISTO_LEN)
                    thr_histo[(unsigned)k] += 1;
            }
        }

        // Accumulate histogram of residuals.
        if ((unsigned)p < HISTO_LEN)
            res_histo[(unsigned)p] += 1;
    }}

    // Estimate threshold from the distribution mode margins.
    unsigned threshold = m_default_threshold;
    std::pair<unsigned,unsigned> margins =
            EstimateDistribMode(thr_histo, HISTO_LEN, 0.5);
    if (margins.first < margins.second) {
        threshold = std::max(threshold,
                             NoisyPointsCutOff(diff_image, margins.second,
                                               thr_histo, HISTO_LEN));
    }

    // Estimate threshold from the maximum entropy partition.
//    threshold = std::max(threshold, MaxEntropyThreshold(thr_histo, HISTO_LEN));
//    threshold = std::max(threshold, MutualInfoThreshold(thr_histo, HISTO_LEN));

    // Simple thresholding from residual distribution: residuals at a certain
    // fraction of stationary (background) points should be below threshold.
    threshold = std::max(threshold,
                         SimpleThreshold(res_histo, HISTO_LEN, 0.67));

    if (m_verbose > 0) LOG(INFO) << "%%%% threshold = " << threshold;

    // Demo and statistics.
    if (m_ctx.isMetricsEnabled() &&
        m_ctx.getConfig().asBool("metrics.threshold_histograms")) {
        std::vector<uint64_t> save_histo;
        save_histo.assign(thr_histo, thr_histo + HISTO_LEN);
        m_ctx.getMetrics().SaveHistogramOfThresholds(save_histo,
                margins.first, margins.second, threshold);
    }

    return threshold;
}

//-----------------------------------------------------------------------------
// Function computes the most narrow interval that contains the specified
// fraction of histogram volume. If "fraction" is not very big, the interval
// tends to embrace the mode of distribution. Function returns empty interval
// if there no enough samples in the histogram.
//-----------------------------------------------------------------------------
virtual std::pair<unsigned,unsigned> EstimateDistribMode(
                        const uint64_t * histo, unsigned len, double fraction)
{
    const unsigned HISTO_LEN = 3*MAX_UBYTE + 1;
    MY_ASSERT(len == HISTO_LEN);
    MY_ASSERT((0.1 < fraction) && (fraction < 0.9));

    std::pair<unsigned,unsigned> margins(0,0);      // the best margins
    uint64_t                     cumul[HISTO_LEN];  // cumulative function

    // Compute cumulative function of the histogram.
    cumul[0] = histo[0];
    for (size_t k = 1; k < HISTO_LEN; ++k) {
        cumul[k] = cumul[k - 1] + histo[k];
    }

    // Compute a fraction of the total sum of histogram.
    const uint64_t fraction_int = static_cast<uint64_t>(
                        std::ceil(fraction * (double)cumul[HISTO_LEN - 1]));
    if (fraction_int < 10) {
        LOG(WARNING) << "too small number of samples";
        return margins;         // too small number of samples
    }

    // Compute the most narrow interval containing the specified fraction
    // of histogram volume.
    unsigned narrowest = 2 * HISTO_LEN;
    for (unsigned b = 0; b < HISTO_LEN; ++b) {
        unsigned t = b;
        uint64_t cb = (b > 0) ? cumul[b - 1] : 0;
        while ((t + 1 < HISTO_LEN) && ((cumul[t] - cb) < fraction_int)) {
            ++t;
        }
        if ((cumul[t] - cb) < fraction_int) {
            break;
        }
        if (narrowest > (t - b)) {
            narrowest = (t - b);
            margins.first = b;      // bottom margin
            margins.second = t;     // top margin
        }
    }
    return margins;
}

//-----------------------------------------------------------------------------
// Function finds first (!) position "t" in the histogram of thresholds such
// that the maximum histogram value inside interval [t-half,t+half] does not
// exceed a certain small number - the number of (almost) standalone noisy
// background pixels. The search starts from the initial position ("initial
// threshold") up to the end of the histogram. The first found position is
// accepted as a binarization threshold for motion mask construction.
//-----------------------------------------------------------------------------
virtual unsigned NoisyPointsCutOff(const cv::Mat  & diff_image,
                                   const unsigned   initial_threshold,
                                   const uint64_t * thr_histo,
                                   const unsigned   len)
{
    const unsigned HISTO_LEN = 3*MAX_UBYTE + 1;
    const unsigned HALF = 5;
    MY_ASSERT((len == HISTO_LEN) && (HISTO_LEN > HALF + 1));

    size_t   npixels = diff_image.total();
    unsigned threshold = std::max(initial_threshold, m_default_threshold);
    unsigned fraction = RoundUInt(m_noisy_points_fraction * (double)npixels);
    fraction = std::max(fraction, unsigned(1));

    for (unsigned k = std::max(HALF, threshold);
                  k < HISTO_LEN - HALF - 1; ++k) {
        uint64_t n = *(std::max_element(thr_histo + k - HALF,
                                        thr_histo + k + HALF + 1));
        if ((0 < n) && (n <= fraction)) {
            threshold = k;
            break;
        }
    }
    return threshold;
}

//-----------------------------------------------------------------------------
// Function implements a simple thresholding from residual distribution:
// residuals at a certain fraction of stationary (background) points should
// be below threshold. Function returns zero if there no enough samples
// in the histogram.
//-----------------------------------------------------------------------------
virtual unsigned SimpleThreshold(
            const unsigned * histo, unsigned len, double fraction)
{
    const unsigned HISTO_LEN = 3*MAX_UBYTE + 1;
    MY_ASSERT(len == HISTO_LEN);
    MY_ASSERT((0.1 < fraction) && (fraction < 0.9));

    uint64_t sum = std::accumulate(histo, histo + HISTO_LEN, uint64_t(0));
    uint64_t partial_sum = 0;
    unsigned threshold = 0;

    // Compute a fraction of the total sum of histogram.
    const uint64_t fraction_int = static_cast<uint64_t>(
                                        std::ceil(fraction * (double)sum));
    if (fraction_int < 10) {
        LOG(WARNING) << "too small number of samples";
        return unsigned(0);         // too small number of samples
    }

    for (unsigned k = 0; k < HISTO_LEN; ++k) {
        partial_sum += histo[k];
        if (partial_sum > fraction_int) {
            threshold = k;
            break;
        }
    }
    return threshold;
}

////-----------------------------------------------------------------------------
//// Function implements the maximum entropy thresholding method:
//// Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for
//// Gray-Level Picture Thresholding Using the Entropy of the Histogram"
//// Graphical Models and Image Processing, 29(3): 273-285.
////-----------------------------------------------------------------------------
//virtual unsigned MaxEntropyThreshold(const uint64_t * histo, unsigned len)
//{
//    const unsigned HISTO_LEN = 3*MAX_UBYTE + 1;
//    MY_ASSERT(len == HISTO_LEN);
//
//    unsigned threshold = 0;
//    unsigned first_nz = 2 * HISTO_LEN, last_nz = 0;
//    uint64_t cumsum[HISTO_LEN + 1];
//    double   max_entropy = 0.0;
//
//    cumsum[0] = 0;
//    for (unsigned i = 0; i < HISTO_LEN; i++) {
//        uint64_t h = histo[i];
//        if (h > 0) {
//            last_nz = i;
//            if (first_nz > HISTO_LEN) first_nz = i;
//        }
//        cumsum[i + 1] = h + cumsum[i];
//    }
//    if ((cumsum[HISTO_LEN] == 0) || (last_nz <= first_nz)) {
//        return threshold;
//    }
//
//    auto Entropy = [](uint64_t h, uint64_t cumsum) -> double {
//        if (h == 0) return 0.0;
//        double p = static_cast<double>(h) / static_cast<double>(cumsum);
//        return -p * std::log(p);
//    };
//
//    for (unsigned split = first_nz + 1; split < last_nz; ++split) {
//        MY_ASSERT(cumsum[split] > 0);
//        MY_ASSERT(cumsum[HISTO_LEN] > cumsum[split]);
//        double E1 = 0.0;
//        for (unsigned i = first_nz; i < split; ++i) {
//            E1 += Entropy(histo[i], cumsum[split]);
//        }
//        double E2 = 0.0;
//        for (unsigned i = split; i <= last_nz; ++i) {
//            E2 += Entropy(histo[i], cumsum[HISTO_LEN] - cumsum[split]);
//        }
//        if (max_entropy < E1 + E2) {
//            max_entropy = E1 + E2;
//            threshold = split;
//        }
//    }
//    return threshold;
//}
//
////-----------------------------------------------------------------------------
//// Function implements .
////-----------------------------------------------------------------------------
//virtual unsigned MutualInfoThreshold(const uint64_t * histo, unsigned len)
//{
//    const unsigned HISTO_LEN = 3*MAX_UBYTE + 1;
//    MY_ASSERT(len == HISTO_LEN);
//
//    unsigned threshold = 0;
//    unsigned first_nz = 2 * HISTO_LEN, last_nz = 0; // first & last non-zero
//    uint64_t cumsum[HISTO_LEN + 1];                 // cumulative function
//    double   entropy = std::numeric_limits<double>::max();
//
//    cumsum[0] = 0;
//    for (unsigned i = 0; i < HISTO_LEN; i++) {
//        uint64_t h = histo[i];
//        if (h > 0) {
//            last_nz = i;
//            if (first_nz > HISTO_LEN) first_nz = i;
//        }
//        cumsum[i + 1] = h + cumsum[i];
//    }
//    if ((cumsum[HISTO_LEN] == 0) || (last_nz <= first_nz)) {
//        return threshold;
//    }
//
//    auto Entropy = [](uint64_t h, uint64_t cumsum) -> double {
//        if (h == 0) return 0.0;
//        double p = static_cast<double>(h) / static_cast<double>(cumsum);
//        return -p * std::log(p);
//    };
//
//    for (unsigned split = first_nz + 1; split < last_nz; ++split) {
//        MY_ASSERT(cumsum[split] > 0);
//        MY_ASSERT(cumsum[HISTO_LEN] > cumsum[split]);
//        const double P1 = static_cast<double>(cumsum[split]) /
//                          static_cast<double>(cumsum[HISTO_LEN]);
//        const double P2 = 1.0 - P1;
//        double E1 = 0.0;
//        for (unsigned i = first_nz; i < split; ++i) {
//            E1 += Entropy(histo[i], cumsum[split]);
//        }
//        double E2 = 0.0;
//        for (unsigned i = split; i <= last_nz; ++i) {
//            E2 += Entropy(histo[i], cumsum[HISTO_LEN] - cumsum[split]);
//        }
//        if (entropy > P1*E1 + P2*E2) {
//            entropy = P1*E1 + P2*E2;
//            threshold = split;
//        }
//    }
//    // LOG(INFO) << "$$$$ mutual info thr: " << threshold << "   ";
//    return threshold;
//}

private:
    const IContext & m_ctx;         // context of application or thread

    int_arr_t m_histogram;
    cv::Mat   m_motion_mask;            // estimation of motion mask
    unsigned  m_default_threshold;      // default threshold = several deltas
    float     m_noisy_points_fraction;  // fraction of noisy background points
    cv::Mat   m_element;                // morphological element
    cv::Mat   m_tmp_mask;
    int       m_verbose;                // level of verbosity

};  // class MotionMask

}   // anonymous namespace

//-----------------------------------------------------------------------------
// Function creates an instance of MotionMask class.
//-----------------------------------------------------------------------------
std::unique_ptr<IMotionMask> CreateMotionMaskGenerator(const IContext & ctx)
{
    return std::unique_ptr<IMotionMask>(new MotionMask(ctx));
}

