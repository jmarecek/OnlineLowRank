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
// Interface to LP (linear-programming) problem solver.
//=============================================================================
class ILpSolver {
public:
    // Destructor.
    virtual ~ILpSolver() {}

    // Function returns vector 'y' that solves the following problem:
    // y = arg min_{y} |y*R - image|_1, where R is a column-major (!) matrix
    // of size rank-x-N, 'rank' is a parameter of low-rank approximation
    // of a matrix of recent images, 'N' is a total number of bytes in image,
    // 'image' is a current image (of size N) which we want to match against
    // the low-rank approximation y*R. When y == nullptr that means the solver
    // failed to initialize the problem or failed to find a feasible solution.
    virtual const flt_arr_t * Solve(const flt_arr_t & R, const ubyte * image, const ubyte * image_end, int take_nth = 0) = 0;
};

std::unique_ptr<ILpSolver> CreateLpSolverL1(const IContext & ctx);

//std::unique_ptr<ILpSolver> CreateLpSolverLinf(const IContext & ctx);
//
//std::unique_ptr<ILpSolver> CreateSolverLinCombRobust(const IContext & ctx);
//
//std::unique_ptr<ILpSolver> CreateSolverLinCombRobustEx(const IContext & ctx);

