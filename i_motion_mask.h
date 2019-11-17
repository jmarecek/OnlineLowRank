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
// Interface to any class constructing a motion mask from a background model
// and a current image frame.
//=============================================================================
class IMotionMask {
public:
    // Destructor.
    virtual ~IMotionMask() {}

    // Function computes motion mask based on background estimation, current
    // image and, possibly, some accumulated history.
    virtual cv::Mat ComputeMask(const cv::Mat & background, const cv::Mat & current_image, const cv::Mat & roi) = 0;
};

std::unique_ptr<IMotionMask> CreateMotionMaskGenerator(const IContext & ctx);

