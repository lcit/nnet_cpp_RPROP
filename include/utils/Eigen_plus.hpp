/*	=========================================================================
	Author: Leonardo Citraro
	Company: 
	Filename: Eigen_plus.hpp
	Last modifed: 20.10.2015 by Leonardo Citraro
	Description: Some matrix operations based on Eigen

	=========================================================================

	=========================================================================
*/
#ifndef EIGEN_PLUS_H
#define EIGEN_PLUS_H

#include <Eigen/Dense>

using namespace Eigen;

/// Slices a matrix in the X direction
///
/// The number of columns remain the same
///
/// @param matrix: matrix to slice
/// @param s1: start index (rows)
/// @param e1: end index (rows)
/// 
/// @return resulting matrix
MatrixXd slice(const MatrixXd &matrix, const int s1, const int e1);

/// Slices a matrix in the X direction and permutes 
/// the rows according to the vector indices
///
/// The number of columns remain the same
///
/// @param matrix: matrix to slice
/// @param indices: rows' permutation vector
/// @param s1: start index (rows)
/// @param e1: end index (rows)
/// 
/// @return resulting matrix
MatrixXd slice(const MatrixXd &matrix, const VectorXi indices, const int s1, const int e1);

/// Slices a matrix in both X and Y directions
///
/// @param matrix: matrix to slice
/// @param s1: start index (rows)
/// @param e1: end index (rows)
/// @param s2: start index (columns)
/// @param e3: end index (columns)
/// 
/// @return resulting matrix
MatrixXd slice(const MatrixXd &matrix, const int s1, const int e1, const int s2, const int e2);

/// Slices a matrix in both X and Y directions and permutes 
/// the rows according to the vector indices
///
/// @param matrix: matrix to slice
/// @param indices: rows' permutation vector
/// @param s1: start index (rows)
/// @param e1: end index (rows)
/// @param s2: start index (columns)
/// @param e3: end index (columns)
/// 
/// @return resulting matrix
MatrixXd slice(const MatrixXd &matrix, const VectorXi indices, const int s1, const int e1, const int s2, const int e2);

/// Shuffles matrix's rows
///
/// @param matrix: matrix to shuffle
/// @param random_state: random seed
/// 
/// @return resulting matrix
MatrixXd shuffle(const MatrixXd &matrix, int random_state);

/// Shuffles matrix's rows and permutes 
/// the rows according to the vector indices
///
/// @param matrix: matrix to shuffle
/// @param indices: rows' permutation vector
/// @param random_state: random seed
/// 
/// @return resulting matrix
MatrixXd shuffle(const MatrixXd &matrix, const VectorXi indices, int random_state);

/// Concatenates two matrices
///
/// @param m1: matrix 1
/// @param m2: matrix 2
/// @param axis: concatenation axis 0 or 1
/// 
/// @return resulting matrix
MatrixXd concatenate(MatrixXd m1, MatrixXd m2, int axis);

#endif
