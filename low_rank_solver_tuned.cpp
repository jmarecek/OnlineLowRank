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
#include "i_context.h"
#include "i_low_rank_solver.h"
#include <smmintrin.h>

namespace {

const float TINY = std::sqrt(std::numeric_limits<float>::min());
const float EPS = std::sqrt(std::numeric_limits<float>::epsilon());
const int   RANK = 4;
const float LO_THR = 1;
const float HI_THR = 255;

//=============================================================================
// Class generator is responsible for drawing random indices while low-rank
// approximation solver iterates over the elements of matrices L and R.
//  Strictly speaking, we draw element indices not fully randomly (that
// would require to keep a large array of permuted indices) but quite
// close to perfect randomness.
//=============================================================================
class RandPerm
{
public:
//-----------------------------------------------------------------------------
// Constructor.
//-----------------------------------------------------------------------------
explicit RandPerm(const IContext & ctx)
    : m_gen() , m_last(Tic()) , m_Mperm(), m_Nperm(), m_Rperm() , m_M(0), m_N(0), m_rank(0) , m_count(0), m_max_count(0), m_prev_i(-1), m_prev_j(-1)
    , m_update_period(ctx.getConfig().asFloat("background.perm_update_period")) , m_update_request(false), m_ready(false) {
    m_gen.seed(static_cast<unsigned long long>( std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    m_gen.discard(m_gen.state_size);
}

//-----------------------------------------------------------------------------
// Destructor.
//-----------------------------------------------------------------------------
virtual ~RandPerm() {}

//-----------------------------------------------------------------------------
// Function returns "true" if this object had been initialized.
//-----------------------------------------------------------------------------
bool IsReady() const { return m_ready; }

//-----------------------------------------------------------------------------
// Function initializes this object.
//-----------------------------------------------------------------------------
virtual void Init(int M, int N, int rank) {
    m_M = M;
    m_N = N;
    m_rank = rank;
    MY_ASSERT((0 < m_rank) && (m_rank < std::min(m_M, m_N)));
    InitArray(m_Mperm, m_M);
    InitArray(m_Nperm, m_N);
    InitArray(m_Rperm, m_rank);
    m_count = 0;
    m_max_count = std::max(m_M, m_N) * m_rank;  // max(|L|,|R|), |.| = size
    Reset();
    m_ready = true;
}

//-----------------------------------------------------------------------------
// Function randomly chooses elements to be updated: L[i,r], R[r,j].
// We say an epoch had passed if every element of matrices L and R
// was touched exactly once. If, for example, the number of elements in
// matrix R is (much) larger than the number of elements in matrix L, the
// latter is not updated on every iteration (but both matrices get updated
// by the end of the epoch) and as such we sometimes return negative
// indices (L_i, L_r) implying the element L[i,r] skips the update.
//  Once an epoch had finished, the function checks for permutation update
// request. If the flag is risen, the arrays of permuted indices are
// re-shuffled to improve randomness. By the end of the epoch the function
// returns "true" and all negative element indices.
//-----------------------------------------------------------------------------
bool GetRandIndices(int & L_i, int & L_r, int & R_r, int & R_j) {
    // It is said "an epoch has passed" when every element of matrices L and R
    // was touched. Then we start a new epoch. If update request had been
    // risen by the end of the epoch, the new index permutations are computed.
    MY_ASSERT(m_max_count > 1);
    if (m_count >= m_max_count) { m_count = 0;
        m_prev_i = m_prev_j = -1;
        if (m_update_request) { m_update_request = false;
            Permute(m_Mperm);
            Permute(m_Nperm);
            Permute(m_Rperm);
        }
        L_i = L_r = R_r = R_j = -1;
        return true;
    }

    // Choose random indices in "rank" dimensions of L and R.
    int r = m_count % m_rank;
    L_r = m_Rperm[(unsigned)r];
    R_r = m_Rperm[(unsigned)((r + 1) % m_rank)];

    // Returns: (a*b)/c.
    auto MulDiv = [](int a, int b, int c) -> int { return static_cast<int>(
            (static_cast<int64_t>(a) * static_cast<int64_t>(b)) / static_cast<int64_t>(c));
    };

    // Choose random indices L_i, R_j depending which size is larger.
    if (m_M > m_N) { int i = m_count / m_rank; int j = MulDiv(m_N - 1, i, m_M - 1); L_i = m_Mperm[(unsigned)i];
        R_j = (m_prev_j != j) ? m_Nperm[(unsigned)j] : -1; m_prev_j = j;
    } else {
        int j = m_count / m_rank; int i = MulDiv(m_M - 1, j, m_N - 1); R_j = m_Nperm[(unsigned)j];
        L_i = (m_prev_i != i) ? m_Mperm[(unsigned)i] : -1; m_prev_i = i;
    }
    ++m_count;
    return false;
}

//-----------------------------------------------------------------------------
// Function checks if a certain period of time had passed and rises
// a request to re-shuffle index permutations.
//-----------------------------------------------------------------------------
void Update(time_point_t t) {
    double elapsed = std::chrono::duration<double>(t - m_last).count();
    if (elapsed >= m_update_period) { m_update_request = true; m_last = t; }
}

//-----------------------------------------------------------------------------
// Function re-sets the internal state to default one.
//-----------------------------------------------------------------------------
void Reset() {
    m_last = Tic();
    m_count = 0;
    m_prev_i = m_prev_j = -1;
    m_update_request = false;
}

private:
//-----------------------------------------------------------------------------
// Initialization function.
//-----------------------------------------------------------------------------
virtual void InitArray(int_arr_t & a, int n) {
    a.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) { a[(unsigned)i] = i; }
    Permute(a);
}

//-----------------------------------------------------------------------------
// Function randomly permutes an array of indices.
//-----------------------------------------------------------------------------
virtual void Permute(int_arr_t & a) { std::shuffle(a.begin(), a.end(), m_gen); }

private:
    std::mt19937_64 m_gen;              // random generator
    time_point_t    m_last;             // last time update was requested
    int_arr_t       m_Mperm;            // index permutation [0..M)
    int_arr_t       m_Nperm;            // index permutation [0..N)
    int_arr_t       m_Rperm;            // index permutation [0..rank)
    int             m_M;                // number of rows of a data matrix
    int             m_N;                // number of columns of data matrix
    int             m_rank;             // desired rank of approximation
    int             m_count;            // counter of elements processed so far
    int             m_max_count;        // upper limit for m_count
    int             m_prev_i;           // previous index i for L_i
    int             m_prev_j;           // previous index j for R_j
    float           m_update_period;    // permutation update period
    bool            m_update_request;   // 0/1: rise update request
    bool            m_ready;            // 0/1: request initialization

};  // class RandPerm

#undef _L_
#undef _R_
#undef _X_
#undef _MASK_
#undef _Xraw_

// L: M-x-Rank, row-major
#define _L_(r,c) m_L[static_cast<size_t>((r) * RANK + (c))]
// R: Rank-x-N, column-major
#define _R_(r,c) m_R[static_cast<size_t>((r) + (c) * RANK)]
// X, A: M-x-N, row-major. If X[.] == 0, then the value is treated as missed.
#define _MASK_(r,c)                    X[static_cast<size_t>((r) * m_N + (c))]
#define _Xraw_(r,c)                    X[static_cast<size_t>((r) * m_N + (c))]
#define    _X_(r,c) static_cast<float>(X[static_cast<size_t>((r) * m_N + (c))])

//=============================================================================
// Implementation of solver for low-rank approximation.
//=============================================================================
class LowRankSolver : public ILowRankSolver
{
public:
//-----------------------------------------------------------------------------
// Constructor.
//-----------------------------------------------------------------------------
explicit LowRankSolver(const IContext & ctx)
    : m_ctx(ctx) , m_verbose(ctx.getOptions().verbose) , m_M(0) , m_N(0) , m_num_bulk_iters(1000) , m_mu(0) , m_exec_time(0)
    , m_noise_delta(0) , m_L() , m_R() , m_perm(ctx) , m_epoch_time(0) , m_optim_stats(false) {
    const ConfigFile & config = ctx.getConfig();
    m_M = config.asInt("background.history_length");
    m_num_bulk_iters = config.asInt("background.num_bulk_iterations");
    m_mu = config.asFloat("background.mu");
    m_exec_time = config.asFloat("background.decomp_time_budget");
    m_noise_delta = config.asFloat("background.noise_delta");
    MY_ASSERT(RANK   == config.asInt("background.rank"));
    MY_ASSERT(LO_THR == config.asFloat("background.lower_bound"));
    MY_ASSERT(HI_THR == config.asFloat("background.upper_bound"));
    MY_ASSERT((0 <= m_noise_delta) && (m_noise_delta <= (HI_THR/5)));
    MY_ASSERT((0 < RANK) && (RANK < m_M));
    MY_ASSERT(m_mu > EPS);
    MY_ASSERT(m_exec_time > EPS);
    Reset();
}

//-----------------------------------------------------------------------------
// Destructor.
//-----------------------------------------------------------------------------
virtual ~LowRankSolver() { }

//-----------------------------------------------------------------------------
// Function updates low-rank approximation of the data matrix X,
// considering the latter does not change dramatically between iterations.
//-----------------------------------------------------------------------------
virtual bool Update(const ubyte * X, const ubyte * X_end, const ubyte *, const ubyte *, const float * R_bar, const float * R_bar_end) {
    static_assert(RANK == 4, "");

    // Create some meaningful starting point if launching from scratch.
    if (!m_perm.IsReady() || (m_N == 0) || m_L.empty() || m_R.empty()) { if (!Init(X, X_end)) return false; }
    MY_ASSERT(m_M * m_N == std::distance(X, X_end));
    MY_ASSERT((R_bar == nullptr) || (m_N == std::distance(R_bar, R_bar_end)));
    MY_ASSERT(::boost::alignment::is_aligned(m_L.data(), 16));
    MY_ASSERT(::boost::alignment::is_aligned(m_R.data(), 16));

    // Accumulate gradient and Lipschitz constant (loosely called Hessian here)
    // to make a step along chosen coordinate.
    auto Accumulate = [this](float a, float x, float Lir_or_Rrj, double & grad, double & hess) {
        float res = a - std::max(x - m_noise_delta, LO_THR);
        if (res < 0) { grad += Lir_or_Rrj * res; hess += Lir_or_Rrj * Lir_or_Rrj;
        } else { res = a - std::min(x + m_noise_delta, HI_THR);
            if (res > 0) { grad += Lir_or_Rrj * res; hess += Lir_or_Rrj * Lir_or_Rrj; } }
    };

    // We periodically regenerate random permutations.
    time_point_t start_time = Tic();
    m_perm.Update(start_time);

    const float L_bar = 1.0f / static_cast<float>(RANK);

    // Optimization within a time budget.
    int epochs = 0;
    double time_elapsed = 0.0;
    while (time_elapsed < m_exec_time) {
        // Make a large number of iterations and check against the time limit.
        for (int iter = 0; iter < m_num_bulk_iters; ++iter) {
            // Get randomly permuted indices for L and R cycles respectively.
            int left_i = 0, left_r = 0, right_r = 0, right_j = 0;
            if (m_perm.GetRandIndices(left_i, left_r, right_r, right_j)) {
                // Special mode for collecting statistics.
                if (m_optim_stats && (epochs < IMetrics::MAX_NUM_EPOCHS)) {
                    m_ctx.getMetrics().AccumulateOptimStatistics(epochs++,
                                        ComputeResidual(X), Toc(start_time));
                    continue;   // keep optimizing
                }

                // We get here when the epoch is over, i.e. each element
                // of matrices L and R was touched exactly once.
                if (m_verbose > 0) { m_epoch_time += Toc(start_time);
                    if (m_verbose > 1) { LOG(INFO) << "<|LR - X|> = " << ComputeResidual(X); }
                    LOG(INFO) << "LR-solver time: " << m_epoch_time << " secs";
                }
                m_epoch_time = 0.0;     // for the next epoch of computation
                return true;            // the epoch has been finished
            }

            // Update matrix L.
            if ((left_i >= 0) && (left_r >= 0)) {
                const int i = left_i;
                const int r = left_r;

                double W_ir = m_mu;
                double grad = m_mu * (_L_(i,r) - L_bar);
                for (int j = 0; j < m_N; ++j) {
                    if (_MASK_(i,j)) {
//#if 0
                        float A_ij = _L_(i,0) * _R_(0,j) + _L_(i,1) * _R_(1,j) + _L_(i,2) * _R_(2,j) + _L_(i,3) * _R_(3,j);
//#else
//                        __m128 tmp = _mm_mul_ps(
//                                        _mm_load_ps(m_L.data() + i * RANK),
//                                        _mm_load_ps(m_R.data() + j * RANK));
//                        tmp = _mm_hadd_ps(tmp, tmp);
//                        float A_ij = _mm_cvtss_f32(_mm_hadd_ps(tmp, tmp));
//#endif
                        Accumulate(A_ij, _X_(i,j), _R_(r,j), grad, W_ir);

//                        float R_rj = _R_(r,j);
//                        int xd = X_ij - m_noise_delta;
//                        if (xd < LO_THR) xd = LO_THR;
//                        float res = A_ij - xd;
//                        if (res < 0) {
//                            grad += R_rj * res;
//                            W_ir += R_rj * R_rj;
//                        } else {
//                            xd = X_ij + m_noise_delta;
//                            if (xd > HI_THR) xd = HI_THR;
//                            res = A_ij - xd;
//                            if (res > 0) {
//                                grad += R_rj * res;
//                                W_ir += R_rj * R_rj;
//                            }
//                        }
                    }
                }

                const float delta = static_cast<float>(-grad / W_ir);
                _L_(i,r) += delta;
            }

            // Update matrix R.
            if ((right_r >= 0) && (right_j >= 0)) {
                const int r = right_r;
                const int j = right_j;

                float  R_bar_j = (R_bar != nullptr) ? R_bar[j] : 0.0f;
                double V_rj = m_mu;
                double grad = m_mu * (_R_(r,j) - R_bar_j);
                for (int i = 0; i < m_M; ++i) {
                    if (_MASK_(i,j)) {
//#if 0
                        float A_ij = _L_(i,0) * _R_(0,j) + _L_(i,1) * _R_(1,j) + _L_(i,2) * _R_(2,j) + _L_(i,3) * _R_(3,j);
//#else
//                        __m128 tmp = _mm_mul_ps(
//                                        _mm_load_ps(m_L.data() + i * RANK),
//                                        _mm_load_ps(m_R.data() + j * RANK));
//                        tmp = _mm_hadd_ps(tmp, tmp);
//                        float A_ij = _mm_cvtss_f32(_mm_hadd_ps(tmp, tmp));
//#endif
                        Accumulate(A_ij, _X_(i,j), _L_(i,r), grad, V_rj);

//                        float L_ir = _L_(i,r);
//                        int xd = X_ij - m_noise_delta;
//                        if (xd < LO_THR) xd = LO_THR;
//                        float res = A_ij - xd;
//                        if (res < 0) {
//                            grad += L_ir * res;
//                            V_rj += L_ir * L_ir;
//                        } else {
//                            xd = X_ij + m_noise_delta;
//                            if (xd > HI_THR) xd = HI_THR;
//                            res = A_ij - xd;
//                            if (res > 0) {
//                                grad += L_ir * res;
//                                V_rj += L_ir * L_ir;
//                            }
//                        }
                    }
                }

                const float delta = static_cast<float>(-grad / V_rj);
                _R_(r,j) += delta;
            }
        }
        time_elapsed = Toc(start_time);
    }
    m_epoch_time += time_elapsed;       // just for statistics
    return false;                       // the epoch has NOT been finished yet
}

//-----------------------------------------------------------------------------
// Function resets the internal state, thus the updating process will
// start from scratch upon calling the function Update(..).
//-----------------------------------------------------------------------------
virtual void Reset()
{
    m_N = 0;
    m_L.clear();
    m_R.clear();
    m_perm.Reset();
}

//-----------------------------------------------------------------------------
// Function computes some meaningful initial matrices L and R,
// also computes the number of data matrix columns N.
//-----------------------------------------------------------------------------
bool Init(const ubyte * X, const ubyte * X_end)
{
    const int64_t Nelems = std::distance(X, X_end);
    MY_ASSERT((m_M <= Nelems) && ((Nelems % m_M) == 0));    // divisible?
    m_N = static_cast<int>(Nelems / m_M);

    m_perm.Init(m_M, m_N, RANK);

    double     total_sum = 0.0;      // sum over the regular values of matrix X
    int        total_count = 0;      // counter of regular values in matrix X
    uint_arr_t xcolumn((size_t)m_M); // column of matrix X

    m_L.resize(static_cast<size_t>(m_M * RANK));
    m_R.resize(static_cast<size_t>(RANK * m_N));

    // Compute the overall mean value of the data matrix X and initialize
    // the first row of R by the column-wise mean values. We say the entry
    // X(r,c) is totally missed if _MASK_(r,c) is zero for all r = [0..M).
    for (int c = 0; c < m_N; ++c) {
        double column_sum = 0.0;    // sum of regular values over c-th column
        int    column_count = 0;    // counter of regular values in c-th column

        // Compute the mean value over c-th column of X.
        for (int r = 0; r < m_M; ++r) { if (_MASK_(r,c)) { total_sum += _X_(r,c); ++total_count; xcolumn[(unsigned)column_count] = _Xraw_(r,c); column_sum += _X_(r,c); ++column_count; } }

        // Totally missed value is temporary set to negative unit.
        // In normal case: R(0,c) = median(xcolumn).
        if (column_count == 0) { _R_(0,c) = -1.0f;
        } else if (column_count == 1) { _R_(0,c) = static_cast<float>(xcolumn[0]);
        } else if (column_count == 2) { _R_(0,c) = static_cast<float>((xcolumn[0] + xcolumn[1]) / 2);
        } else { // initialize R(0,c) by the median value
            auto xcol = xcolumn.begin();
            unsigned sz = (unsigned)column_count;
            unsigned med = sz / 2;
            std::nth_element(xcol, xcol + med, xcol + sz);
            unsigned x_med = xcolumn[med];
            if (sz % 2 == 1) {  // odd size
                _R_(0,c) = static_cast<float>(x_med);
            } else {            // even size
                std::nth_element(xcol, xcol + med - 1, xcol + sz);
                _R_(0,c) = static_cast<float>((x_med + xcolumn[med - 1]) / 2);
            }
        }
//        _R_(0,c) = static_cast<float>(
//                    (column_count > 0) ? (column_sum / column_count) : -1.0);
    }
    if (2 * total_count < m_M * m_N) { LOG(WARNING) << "too many missed values in data matrix X"; return false; }
    const float mean = static_cast<float>(total_sum / total_count);

    // Copy the first row of R to rest ones. Totally missed entries will
    // be initialized by the mean value.
    for (int c = 0; c < m_N; ++c) { float & R_oc = _R_(0,c); bool    missed = (R_oc < -0.5);

        for (int r = 1; r < RANK; ++r) { _R_(r,c)= missed ? mean : R_oc; }
        if (missed) R_oc = mean;
    }

    // Set equal weights to all combinations of the rows of matrix R.
    std::fill(m_L.begin(), m_L.end(), static_cast<float>(1.0/(double)RANK));
    return true;
}

//-----------------------------------------------------------------------------
// Function returns M-x-rank row-major (!!!) matrix L
// of low-rank L*R approximation to the data matrix X.
//-----------------------------------------------------------------------------
virtual const flt_arr_t & getL() const {
    return m_L;
}

//-----------------------------------------------------------------------------
// Function returns rank-x-N column-major (!!!) matrix R
// of low-rank L*R approximation to the data matrix X.
//-----------------------------------------------------------------------------
virtual const flt_arr_t & getR() const {
    return m_R;
}

//-----------------------------------------------------------------------------
// Function returns A = L*R the row-major (!!!), M-x-N, low-rank
// approximation to the data matrix X.
//-----------------------------------------------------------------------------
virtual void getApproximation(flt_arr_t & A) const {
    static_assert(RANK == 4, "");
    A.resize((size_t)(m_M * m_N));
    for (int r = 0; r < m_M; ++r) {
    for (int c = 0; c < m_N; ++c) { A[static_cast<size_t>((r) * m_N + (c))] = _L_(r,0) * _R_(0,c) + _L_(r,1) * _R_(1,c) + _L_(r,2) * _R_(2,c) + _L_(r,3) * _R_(3,c); }}
}

//-----------------------------------------------------------------------------
// Function prints information about the solver.
//-----------------------------------------------------------------------------
virtual void About() const { LOG(INFO) << "Solver for low-rank matrix approximation, Version 5"; }

//-----------------------------------------------------------------------------
// Function prints the left-hand side matrix of low-rank approximation L*R.
//-----------------------------------------------------------------------------
virtual void PrintL() const
{
    std::cout << "Matrix L (" << m_M << "x" << RANK << "):" << std::endl;
    for (int r = 0; r < m_M; ++r) { for (int k = 0; k < RANK; ++k) { std::cout << std::setw(10) << _L_(r,k) << "  "; } std::cout << std::endl; }
    std::cout << std::endl;
}

//-----------------------------------------------------------------------------
// Function computes residual (L2 vector norm) between the data matrix X and
// its low-rank approximation (achieved so far).
//-----------------------------------------------------------------------------
virtual double ComputeResidual(const ubyte * X) const {
    long double diff = 0.0;
    int count = 0;
    for (int r = 0; r < m_M; ++r) {
    for (int c = 0; c < m_N; ++c) {
        if (_MASK_(r,c)) {
            float A_rc = _L_(r,0) * _R_(0,c) + _L_(r,1) * _R_(1,c) + _L_(r,2) * _R_(2,c) + _L_(r,3) * _R_(3,c); diff += std::pow(A_rc - _X_(r,c), 2);
            ++count;
        }
    }}
    return std::sqrt(std::fabs((double)diff / std::max(count,1)));
}

//-----------------------------------------------------------------------------
// Function enables 100 epochs of optimization (in one go) aiming
// to collect statistics for presentation, profiling or debugging.
//-----------------------------------------------------------------------------
virtual void EnableOptimStatistics() {
    MY_ASSERT(m_ctx.isMetricsEnabled());
    m_optim_stats = true;
    LOG(INFO) << "optimization statistics is enabled in low-rank solver";
    m_exec_time = std::numeric_limits<float>::max();    // do all in one go
}

private:
    const IContext & m_ctx;

    int   m_verbose;            // level of verbosity
    int   m_M;                  // number of rows of a data matrix
    int   m_N;                  // number of columns of data matrix
    int   m_num_bulk_iters;     // number of bulk iterations
    float m_mu;                 // regularization parameter
    float m_exec_time;          // execution time budget per update (seconds)
    float m_noise_delta;        // uniform noise interval [-delta,+delta]

    flt_arr_t m_L;              // matrix L of decomposition L*R
    flt_arr_t m_R;              // matrix R of decomposition L*R
    RandPerm  m_perm;           // generator of random indices

    double     m_epoch_time;    // time spent in the last epoch (statistics)
    bool       m_optim_stats;   // 0/1: collect optimization statistics

};  // class LowRankSolver

}   // anonymous namespace

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
lr_solver_t CreateLowRankSolverTuned(const IContext & ctx) { return lr_solver_t(new LowRankSolver(ctx)); }

