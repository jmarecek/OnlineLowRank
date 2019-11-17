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
#include "i_metrics.h"
#include "config_file.h"
#include "options.h"
#include "i_context.h"
#include "i_lp_solver.h"

namespace {

#ifdef MY_ENABLE_CPLEX
struct SetName { std::string str;
    SetName(const char * name, IloInt v) { std::stringstream ss; ss << name << v; str = ss.str(); }
    const char * operator()() const { return str.c_str(); }
};
#endif	// MY_ENABLE_CPLEX

//#undef _R_
//// R: Rank-x-N, column-major
//#define _R_(r,c) R[static_cast<size_t>((r) + (c) * rank)]

#define MY_STRICT_PIXEL_RANGE 0

//=============================================================================
// Interface to LP (linear-programming) problem solver.
//=============================================================================
class LpSolverL1 : public ILpSolver
{
public:
//-----------------------------------------------------------------------------
// Constructor.
//-----------------------------------------------------------------------------
explicit LpSolverL1(const IContext & ctx)
    : m_config(ctx.getConfig()) , m_y() , m_lower(0) , m_upper(0) , m_rank(0) , m_take_nth(1)
    , m_verbose(ctx.getOptions().verbose) , m_one_thread(false) , m_set_name(false) , m_check_sol(false)
//    , m_env()
//    , m_model()
//    , m_vars()
//    , m_ranges()
//    , m_cplex()
//    , m_values()
//    , m_ready(false)
{
    m_lower = m_config.asDouble("background.lower_bound");
    m_upper = m_config.asDouble("background.upper_bound");
    m_rank = m_config.asInt("background.rank");
    m_take_nth = m_config.asInt("lp-solver.take_nth");
    m_one_thread = m_config.asBool("lp-solver.single_thread");
    m_set_name = m_config.asBool("lp-solver.set_name_and_print_lp");
    m_check_sol = m_config.asBool("lp-solver.check_solution");
    m_y.resize((unsigned)m_rank);
    MY_ASSERT((m_rank > 0) && (m_take_nth > 0));
}

//-----------------------------------------------------------------------------
// Destructor.
//-----------------------------------------------------------------------------
virtual ~LpSolverL1() {}

#ifdef MY_ENABLE_CPLEX

//-----------------------------------------------------------------------------
// Function returns vector 'y' that solves the following problem:
// y = arg min_{y} |y*R - image|_1, where R is a column-major (!) matrix
// of size rank-x-N, 'rank' is a parameter of low-rank approximation
// of a matrix of recent images, 'N' is a total number of bytes in image,
// 'image' is a current image (of size N) which we want to match against
// the low-rank approximation y*R. When y == nullptr that means the solver
// failed to initialize the problem or failed to find a feasible solution.
//-----------------------------------------------------------------------------
//#if 1
virtual const flt_arr_t * Solve(const flt_arr_t & R, const ubyte * image, const ubyte * image_end, int take_nth) {
//#if (MY_STRICT_PIXEL_RANGE != 0)
//    const IloInt NUM_CONSTR = 4;    // enforce:  0 <= (c*R)_i <= 255
////#else
    const IloInt NUM_CONSTR = 3;    // relax constraint on (c*R)_i
//#endif
    const int take_each_Nth = (take_nth > 0) ? take_nth : m_take_nth;
    bool      ok = false;
    IloEnv    env;

    try {
        const IloInt Npixels = std::distance(image, image_end);
        const IloInt N = Npixels / take_each_Nth;

        MY_ASSERT(!R.empty());
        MY_ASSERT(R.size() == static_cast<size_t>(m_rank * Npixels));

        IloModel       model(env);
        IloNumVarArray var(env);
        IloRangeArray  con(env);
        IloObjective   obj = IloMinimize(env);

        // Create variables.
        for (IloInt i = 0; i < N; ++i) { var.add(IloNumVar(env, 0.0, IloInfinity));
            if (m_set_name) { var[i].setName(SetName("m", i)()); }
        }
        for (IloInt r = 0; r < m_rank; ++r) { var.add(IloNumVar(env, -IloInfinity, IloInfinity));
            if (m_set_name) { var[r + N].setName(SetName("y", r)()); }
        }
        MY_ASSERT(var.getSize() == (N + m_rank));

        // Set up the objective function.
        for (IloInt i = 0; i < N; ++i) { obj.setLinearCoef(var[i], 1.0); }

        // Create constraints.
        for (IloInt i = 0; i < N; ++i) {
            double f_i = static_cast<double>(image[i * take_each_Nth]);
            con.add(IloRange(env, 0, IloInfinity));
            con.add(IloRange(env, -IloInfinity, f_i));
            con.add(IloRange(env, f_i, +IloInfinity));
//#if (MY_STRICT_PIXEL_RANGE != 0)
//            con.add(IloRange(env, m_lower, m_upper));
//#endif
            if (m_set_name) {
                con[NUM_CONSTR*i + 0].setName(SetName("con.m", i)());
                con[NUM_CONSTR*i + 1].setName(SetName("con.yR.minus.m", i)());
                con[NUM_CONSTR*i + 2].setName(SetName("con.yR.plus.m", i)());
//#if (MY_STRICT_PIXEL_RANGE != 0)
//                con[NUM_CONSTR*i + 3].setName(SetName("con.yR", i)());
//#endif
            }
        }
        MY_ASSERT(con.getSize() == NUM_CONSTR*N);

        // Set up the constraints.
        for (IloInt i = 0; i < N; ++i) {
            con[NUM_CONSTR*i + 0].setLinearCoef(var[i],  1.0);
            con[NUM_CONSTR*i + 1].setLinearCoef(var[i], -1.0);
            con[NUM_CONSTR*i + 2].setLinearCoef(var[i],  1.0);
            for (IloInt r = 0; r < m_rank; ++r) {
                double R_ri = R[(size_t)(r + i * take_each_Nth * m_rank)];
                con[NUM_CONSTR*i + 1].setLinearCoef(var[r + N], R_ri);
                con[NUM_CONSTR*i + 2].setLinearCoef(var[r + N], R_ri);
//#if (MY_STRICT_PIXEL_RANGE != 0)
//                con[NUM_CONSTR*i + 3].setLinearCoef(var[r + N], R_ri);
//#endif
            }
        }

        model.add(obj);
        model.add(con);
        IloCplex cplex(model);

        // Save LP model once in a while.
        if (m_set_name) { m_set_name = false; cplex.exportModel(JoinToPath(TEST_OUTPUT_DIR, "L1.lp").c_str()); }

        // Enforce single-thread mode.
        if (m_one_thread) { if (m_verbose > 0) LOG(INFO) << "LP-solver runs in the single-thread mode";
            cplex.setParam(IloCplex::IntParam::Threads, CPXINT(1));
            MY_ASSERT(cplex.getParam(IloCplex::IntParam::Threads) == 1);
            //cplex.setParam(IloCplex::Param::CPUmask, "1");  // XXX causes error on Mac
        }

        // Optimize the problem and obtain solution.
        if (!cplex.solve()) { throw std::runtime_error("Failed to optimize LP"); }

        // Extract the solution.
        IloNumArray vals(env);
        cplex.getValues(vals, var);
        MY_ASSERT(vals.getSize() == (IloInt)(m_rank + N));
        for (IloInt r = 0; r < m_rank; ++r) { m_y[(size_t)r] = static_cast<float>(vals[r + N]); }

        // Print the status and check the solution.
        if (m_verbose > 0) { LOG(INFO) << "CPUmask: " << cplex.getParam(IloCplex::Param::CPUmask) << "; solution status: " << cplex.getStatus() << "; solution value: " << cplex.getObjValue() << "; #values: " << vals.getSize(); }
        if (m_check_sol) {
            CheckSolution(R, image, Npixels, take_each_Nth, vals);
            cplex.getSlacks(vals, con);
            LOG(INFO) << "#slacks: " << vals.getSize();
            cplex.getDuals(vals, con);
            LOG(INFO) << "#duals: " << vals.getSize();
            cplex.getReducedCosts(vals, var);
            LOG(INFO) << "#reduced costs: " << vals.getSize();
        }
        ok = true;
    } catch (IloException & e) { LOG(ERROR) << "CPlex exception: " << e;
    } catch (std::exception & e) { LOG(ERROR) << "Exception: " << e.what();
    } catch (...) { LOG(ERROR) << "Unknown exception at " << __FILE__ << ":" << __LINE__;
    }
    env.end();
    return (ok ? &m_y : static_cast<flt_arr_t*>(nullptr));
}
//#else
//virtual const flt_arr_t * Solve(const flt_arr_t & R,
//                                const ubyte   * image,
//                                const ubyte   * image_end)
//{
//    bool ok = false;
//    try {
//        const IloInt N = static_cast<IloInt>(std::distance(image, image_end));
//        MY_ASSERT(!R.empty());
//        MY_ASSERT(R.size() == static_cast<size_t>(N * (IloInt)m_rank));
//
//        if (!m_ready) {
//            Init(R, image, N);
//        }
//
//#warning "XXX we recreate environment on each iteration currently"
//m_ready = false; // XXX
//
//        // Optimize the problem and extract the solution.
//        if (m_cplex->solve()) {
//            m_cplex->getValues(*m_values, *m_vars);
//            MY_ASSERT(m_values->getSize() == (IloInt)m_rank + N);
//            for (unsigned r = 0; r < (unsigned)m_rank; ++r) {
//                m_y[r] = static_cast<float>((*m_values)[(IloInt)r + N]);
//            }
//            ok = true;
//        } else {
//            LOG(ERROR) << "Failed to optimize LP";
//        }
//    } catch (IloException & e) {
//        LOG(ERROR) << "CPlex exception: " << e;
//    } catch (std::exception & e) {
//        LOG(ERROR) << "Exception: " << e.what();
//    } catch (...) {
//        LOG(ERROR) << "Unknown exception at " << __FILE__ << ":" << __LINE__;
//    }
//    return (ok ? &m_y : static_cast<flt_arr_t*>(nullptr));
//}
//
//private:
////-----------------------------------------------------------------------------
//// Function initializes CPlex framework and LP-problem given input data.
////-----------------------------------------------------------------------------
//void Init(const flt_arr_t & R, const ubyte * image, const IloInt N)
//{
//    // Close the previous environment first, then create a new one.
//    m_ready = false;
//    if (m_env) {
//        m_values.reset();
//        m_cplex.reset();
//        m_ranges.reset();
//        m_vars.reset();
//        m_model.reset();
//        m_env->end();
//        m_env.reset();
//    }
//    m_env.reset(new IloEnv());
//    m_model.reset(new IloModel(*m_env));
//    m_vars.reset(new IloNumVarArray(*m_env));
//    m_ranges.reset(new IloRangeArray(*m_env));
//
//    IloEnv         & env = *m_env;                  // environment
//    IloNumVarArray & var = *m_vars;                 // variables
//    IloRangeArray  & con = *m_ranges;               // constraints
//    IloObjective     obj = IloMinimize(env);        // objective function
//
//    const IloInt rank = static_cast<IloInt>(m_rank);
//
//    // Auxiliary variables m[i]: 0 <= m[i] < +inf; objective = min sum_i m[i].
//    for (IloInt i = 0; i < N; ++i) {
//        var.add(IloNumVar(env, 0.0, IloInfinity));
//        obj.setLinearCoef(var[i], 1.0);
//        con.add(IloRange(env, 0.0, IloInfinity));
//        con[i].setLinearCoef(var[i], 1.0);
//    }
//    // Target variables y[i].
//    for (IloInt i = 0; i < rank; ++i) {
//        var.add(IloNumVar(env, -IloInfinity, +IloInfinity));
//    }
//    MY_ASSERT(var.getSize() == N + rank);
//    MY_ASSERT(con.getSize() == N);
//
//    // Add the rest of constraints. Note, var[0..N) are auxiliary variables,
//    // and var[N..(N+rank)) are target variables.
//    for (IloInt c = con.getSize(), i = 0; i < N; ++i) {
//        double f_i = static_cast<double>(image[i]);
//        con.add(IloRange(env, -IloInfinity, f_i));
//        con.add(IloRange(env, f_i, +IloInfinity));
//        con.add(IloRange(env, PIXEL_MIN, PIXEL_MAX));
//        for (IloInt r = 0; r < rank; ++r) {
//            double R_ri = _R_(r,i);
//            con[c + 0].setLinearCoef(var[r + N], R_ri);
//            con[c + 1].setLinearCoef(var[r + N], R_ri);
//            con[c + 2].setLinearCoef(var[r + N], R_ri);
//        }
//        con[c + 0].setLinearCoef(var[i], -1.0);
//        con[c + 1].setLinearCoef(var[i],  1.0);
//        c += 3;
//    }
//    MY_ASSERT(con.getSize() == 4*N);
//
//    m_model->add(obj);
//    m_model->add(con);
//    m_cplex.reset(new IloCplex(*m_model));
//    m_values.reset(new IloNumArray(*m_env));
//    m_ready = true;
//}
//#endif
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
virtual void CheckSolution(const flt_arr_t   & R,
                           const ubyte     * image,
                           const IloInt        Npixels,
                           const IloInt        Take_every_n_th,
                           const IloNumArray & vars)
{
    const double EPS = std::pow(std::numeric_limits<double>::epsilon(), 0.25);
    const IloInt N = Npixels / Take_every_n_th;

    double min_yR = +std::numeric_limits<double>::max();
    double max_yR = -std::numeric_limits<double>::max();
    double sum_yR = 0.0;
    int    fail_count = 0;

    MY_ASSERT(vars.getSize() == N + m_rank);
    for (IloInt i = 0; i < N; ++i) {
        double yR = 0.0;
        for (IloInt r = 0; r < m_rank; ++r) {
            yR += m_y[(size_t)r] * R[(size_t)(r + i * Take_every_n_th * m_rank)];
        }

        double f = static_cast<double>(image[i * Take_every_n_th]);
        if (std::fabs(yR - f) > vars[i] + EPS) {
            if (fail_count < 100) {
                LOG(INFO) << "Badness: |yR - image| > m,  yR: " << yR
                          << ",  image: " << f << ",  minimum: " << vars[i];
            } else if (fail_count == 100) {
                LOG(INFO) << "Too many failures ...";
            }
            ++fail_count;
        }
        if ((yR < m_lower - EPS) || (yR > m_upper + EPS)) {
            if (fail_count < 100) {
                LOG(INFO) << "Badness: yR is outside bounds,  yR: " << yR;
            } else if (fail_count == 100) {
                LOG(INFO) << "Too many failures ...";
            }
            ++fail_count;
        }
        if (yR < min_yR) min_yR = yR;
        if (yR > max_yR) max_yR = yR;
        sum_yR += yR;
    }
    if (fail_count == 0) LOG(INFO) << "Solution has no issues";
    LOG(INFO) << "min(yR): " << min_yR << ", max(yR): " << max_yR
              << ", <yR>: " << (sum_yR / (double)N);
}

#else 	// !MY_ENABLE_CPLEX

virtual const flt_arr_t * Solve(const flt_arr_t&, const ubyte*, const ubyte*, int)
{
	MY_VERIFY(false, "CPlex is disabled");
	return static_cast<const flt_arr_t*>(nullptr);
}

#endif 	// MY_ENABLE_CPLEX

private:
#ifdef MY_ENABLE_CPLEX
    typedef std::unique_ptr<IloEnv>         env_t;
    typedef std::unique_ptr<IloModel>       model_t;
    typedef std::unique_ptr<IloNumVarArray> var_arr_t;
    typedef std::unique_ptr<IloRangeArray>  rng_arr_t;
    typedef std::unique_ptr<IloCplex>       cplex_t;
    typedef std::unique_ptr<IloNumArray>    values_t;
#endif

    const ConfigFile & m_config;    // configuration settings

    flt_arr_t m_y;          // placeholder for solution
    double    m_lower;      // lower bound on pixel value
    double    m_upper;      // upper bound on pixel value
    int       m_rank;       // desired rank of approximation
    int       m_take_nth;   // take every n-th element of matrix R
    int       m_verbose;    // level of verbosity
    bool      m_one_thread; // 0/1: enforce single-thread mode
    bool      m_set_name;   // 0/1: assign names to variables and save LP
    bool      m_check_sol;  // 0/1: check LP solution

//    env_t     m_env;        // CPlex environment
//    model_t   m_model;      // CPlex model
//    var_arr_t m_vars;       // CPlex variables
//    rng_arr_t m_ranges;     // CPlex ranges
//    cplex_t   m_cplex;      // CPlex solver
//    values_t  m_values;     // solution
//    bool      m_ready;      // flag of solver readiness

};  // class LpSolverL1

}   // anonymous namespace

//-----------------------------------------------------------------------------
// Factory function create the class instance.
//-----------------------------------------------------------------------------
std::unique_ptr<ILpSolver> CreateLpSolverL1(const IContext & ctx)
{
    return std::unique_ptr<ILpSolver>(new LpSolverL1(ctx));
}

