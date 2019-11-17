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
// Interface to any background model estimator.
//=============================================================================
class IBackground {
public:
    enum { MOVING_POINT = 0, BLACK_POINT  = 1, WHITE_POINT  = static_cast<int>(std::numeric_limits<ubyte>::max()) };

    // Destructor.
    virtual ~IBackground() {}

    // Function updates the current background estimation given a new image
    // and motion mask (0 - foreground pixel, > 0 - background pixel). Note,
    // the new image and mask can be ignored if computation had not been
    // finished since the last call to this function. This is a way to split
    // up a long job into incremental phases over the time line.
    virtual cv::Mat Update(const cv::Mat & img, const cv::Mat & roi) = 0;

    // Function returns so far estimated background as an image. Note, the
    // background can remain the same for quite a long time, if the function
    // Update() is slowly progressing.
    virtual cv::Mat asImage() const = 0;

    // Function returns one of the background model, by its index, from
    // the low-rank approximation. The background is converted to image.
    virtual cv::Mat & asImage(int index, cv::Mat & img) const = 0;

    // Function returns one of the image stored in the recent history.
    virtual const cv::Mat getHistory(int index) const = 0;

    // Function returns a point-wise moving-average image of type CV_32FC3;
    // if moving average is not available, an empty image is returned.
    virtual const cv::Mat getMovingAverage() const = 0;
};

std::unique_ptr<IBackground> CreateBackground(const IContext & ctx);
