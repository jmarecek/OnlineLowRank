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

#pragma once

class IContext;

//=============================================================================
// Interface to a solver for low-rank approximation of data matrix.
//=============================================================================
class ILowRankSolver {
public:
    // Destructor.
    virtual ~ILowRankSolver() {}

    // Function updates low-rank approximation of the data matrix X,
    // considering the latter does not change dramatically between iterations.
    // Note, it might happen that this function does not finish a job within
    // prescribed time budget, so one should invoke it many times, with the
    // same "X" and "mask" arguments, until the "true" value is returned. Then
    // the low-rank approximation L*R and its factors L, R can be used for
    // further processing. Some reference data, supplied by user, can be used,
    // for example, for regularization of LR-solver.
    // @param X        pointer to the beginning of data matrix.
    // @param X_end    pointer to the end of data matrix.
    // @param mask     pointer to the beginning of mask.
    // @param mask_end pointer to the end of mask.
    // @param ref      pointer to the beginning of some reference data.
    // @param ref_end  pointer to the end of some reference data.
    virtual bool Update(const ubyte * X, const ubyte * X_end, const ubyte * mask, const ubyte * mask_end, const float * ref, const float   *  ref_end) = 0;

    // Function resets the internal state, thus the updating process will
    // start from scratch upon calling the function Update(..).
    virtual void Reset() = 0;

    // Function returns M-x-rank row-major (!) matrix L of low-rank L*R
    // approximation to X. It makes sense to call this function only after
    // Update() returned "true".
    virtual const flt_arr_t & getL() const = 0;

    // Function returns rank-x-N column-major (!) matrix R of low-rank L*R
    // approximation to X. It makes sense to call this function only after
    // Update() returned "true".
    virtual const flt_arr_t & getR() const = 0;

    // Function returns M-x-N row-major (!) matrix of low-rank L*R
    // approximation to X. It makes sense to call this function only after
    // Update() returned "true".
    virtual void getApproximation(flt_arr_t & A) const = 0;

    // Function prints information about the solver.
    virtual void About() const = 0;

    // Function prints the left matrix of low-rank approximation L*R.
    virtual void PrintL() const = 0;

    // Function enables 100 epochs of optimization (in one go) aiming
    // to collect statistics for presentation, profiling or debugging.
    virtual void EnableOptimStatistics() = 0;
};

typedef std::unique_ptr<ILowRankSolver> lr_solver_t;

//// Factory function(s).
//std::unique_ptr<ILowRankSolver> CreateLowRankSolver(int M, int N, int rank,
//                                                    int num_bulk_iterations,
//                                                    float mu);

//std::unique_ptr<ILowRankSolver> CreateLowRankSolver2(const FeedList & params);

//std::unique_ptr<ILowRankSolver> CreateLowRankSolver3(const IContext & ctx);
//std::unique_ptr<ILowRankSolver> CreateLowRankSolver4(const IContext & ctx);

//lr_solver_t CreateLowRankSolver5(const IContext & ctx);
//lr_solver_t CreateLowRankSolverZeroDelta(const IContext & ctx);
lr_solver_t CreateLowRankSolverTuned(const IContext & ctx);
