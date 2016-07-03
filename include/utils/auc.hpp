/*	=========================================================================
	Author: Leonardo Citraro
	Company: 
	Filename: auc.hpp
	Last modifed: 20.10.2015 by Leonardo Citraro
	Description: AUC helper functions

	=========================================================================
	code partially copied from: http://www.mathworks.com/matlabcentral/fileexchange/41258-faster-roc-auc
	=========================================================================
*/
#ifndef AUC_H
#define AUC_H

#include <cmath>
#include <climits>
#include <vector>
#include <algorithm>
#include <Eigen/Dense> 

using namespace Eigen;

/// Computes the AUC
///
/// @param labels: matrix(n,1) filled with 0 or 1
/// @param scores: matrix(n,1) filled with [0..1]
/// @param posclass: positive label 0 or 1
/// 
/// @return AUC
double calcAUC      (MatrixXd labels, MatrixXd scores, int posclass);


double trapezoidArea(double X1, double X2, double Y1, double Y2);

#endif
