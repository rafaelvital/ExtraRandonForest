/**************************************************************************
*   File:                        rtnode.h                                 *
*   Description:   Basic classes for Tree based algorithms                *
*   Copyright (C) 2007 by  Walter Corno & Daniele Dell'Aglio              *
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program; if not, write to the                         *
*   Free Software Foundation, Inc.,                                       *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
***************************************************************************/
#include "rtLeafLinearInterp.h"

using namespace PoliFitted;

rtLeafLinearInterp::rtLeafLinearInterp() : mCoeffs(0)
{}

/**
  * Basic constructor
  * @param val the value to store in the node
  */
rtLeafLinearInterp::rtLeafLinearInterp( Dataset* data ) : mCoeffs(0)
{
    Fit(data);
}

rtLeafLinearInterp::~rtLeafLinearInterp()
{
    gsl_vector_free(mCoeffs);
}

float rtLeafLinearInterp::Fit( Dataset* data)
{
    unsigned int size = data->size();
    unsigned int input_size = data->GetInputSize() + 1;
    mCoeffs = gsl_vector_alloc(input_size);
    // In case the samples are less than the unknowns
    if (size < input_size) {
        float result = 0.0;
        for ( unsigned int i = 0; i < data->size(); i++ ) {
            result += data->at(i)->GetOutput();
        }
        result /= (float)data->size();
        gsl_vector_set (mCoeffs, 0, result);
        for (unsigned int j = 1; j < input_size; j++) {
            gsl_vector_set (mCoeffs, j, 0);
        }
        return 0.0;
    } else {
        gsl_matrix* X, *cov;
        gsl_vector* y, *w;
        double chisq;

        X = gsl_matrix_alloc (size, input_size);
        y = gsl_vector_alloc (size);
        w = gsl_vector_alloc (size);
        cov = gsl_matrix_alloc (input_size, input_size);

        for (unsigned int i = 0; i < size; i++) {
            gsl_matrix_set (X, i, 0, 1.0);
            for (unsigned int j = 1; j < input_size; j++) {
                gsl_matrix_set (X, i, j, data->at(i)->GetInput(j - 1));
            }

            gsl_vector_set (y, i, data->at(i)->GetOutput());
            //     gsl_vector_set (w, i, 1.0/(ei*ei));
        }

        gsl_multifit_linear_workspace* work = gsl_multifit_linear_alloc (size, input_size);
        //   gsl_multifit_wlinear (X, w, y, c, cov,&chisq, work);
        gsl_multifit_linear (X, y, mCoeffs, cov, &chisq, work);
        gsl_multifit_linear_free (work);

        gsl_matrix_free (X);
        gsl_vector_free (y);
        gsl_vector_free (w);
        gsl_matrix_free (cov);

        return chisq;
    }
}

float rtLeafLinearInterp::getValue(Tuple* input)
{
    float result = gsl_vector_get(mCoeffs, 0);
    for (unsigned int i = 1; i < mCoeffs->size; i++) {
        result += gsl_vector_get(mCoeffs, i) * (*input)[i - 1];
    }
//  if (result < 0) result = 0;
    return result;
}

void rtLeafLinearInterp::WriteOnStream (ofstream& out)
{
    out << "LLI" << endl;
    out << mCoeffs->size << endl;
    for (unsigned int i = 0; i < mCoeffs->size; i++) {
        out << gsl_vector_get(mCoeffs, i) << " ";
    }
}

void rtLeafLinearInterp::ReadFromStream (ifstream& in)
{
    unsigned int size;
    double value;
    in >> size;
    for (unsigned int i = 0; i < size; i++) {
        in >> value;
        gsl_vector_set(mCoeffs, i, value);
    }
}

