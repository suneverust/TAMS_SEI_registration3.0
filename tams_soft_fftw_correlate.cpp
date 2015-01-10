/***************************************************************************
  **************************************************************************
  
  SOFT: SO(3) Fourier Transforms
  Version 2.0

  Copyright (c) 2003, 2004, 2007 Peter Kostelec, Dan Rockmore
  
  This file is part of SOFT.

  SOFT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  SOFT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
  See the accompanying LICENSE file for details.
  
  ************************************************************************
  ************************************************************************/

/*
  to test the correlation routines

  - ASSUMES bwIn >= bwOut

  - uses the Wigner-d symmetries and FFTW
  - uses part of s2kit

  - INTERLEAVED (i.e. real/imaginary) SAMPLES of signal and pattern files
  - [result] -> optional -> filename of all the correlation values
                (if you want all of them)
  - bwIn -> bw of input spherical signals
  - bwOut -> bw of so(3) transform you want to do
  - degLim -> max degree of Wigner-D functions you'll be using


  ASSUMES bwIn >= bwOut

  example: test_soft_fftw_correlate2 signalFile patternFile bwIn bwOut degLim [result]
*/

  /*
  * Modified by Bo Sun
  * From TAMS, University of Hamburg
  * "tams_soft_fftw_correlate.cpp" is modified from "test_soft_fftw_correlate2.c"
  * in package SOFT(3).
  * 1. Rather than read the real and imaginary part of the input files,
  *    the new function just accept the real part (imaginary part is 0) through
  *    the Eigen::VectorXf structure
  * 2. Rather than show the correlation results on the screen or save in a file,
  *    the new function just return the rotaion angles.
  * Author: Bo Sun
  * Afflication: TAMS, University of Hamburg
  * E-Mail: bosun@informatik.uni-hamburg.de
  *         user_mail@QQ.com
  */

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

extern "C"{
#include "fftw3.h"
#include "makeweights.h"
#include "so3_correlate_fftw.h"
#include "soft_fftw.h"

#include "s2_cospmls.h"
#include "s2_legendreTransforms.h"
#include "s2_semi_memo.h"
} // end of extern "C"

#define NORM( x ) ( (x[0])*(x[0]) + (x[1])*(x[1]) )

void tams_soft_fftw_correlate ( const Eigen::VectorXf &TAMS_sei_sig_real,
                                const Eigen::VectorXf &TAMS_sei_pat_real,
                                const int tams_bwIn, const int tams_bwOut, const int tams_degLim,
                                double &alpha,
                                double &beta,
                                double &gamma)
{
  int i ;
  int n, bwIn, bwOut, degLim ;
  fftw_complex *workspace1, *workspace2  ;
  double *workspace3 ;
  double *sigR, *sigI ;
  double *sigCoefR, *sigCoefI ;
  double *patCoefR, *patCoefI ;
  fftw_complex *so3Sig, *so3Coef ;
  fftw_plan p1 ;
  int na[2], inembed[2], onembed[2] ;
  int rank, howmany, istride, idist, ostride, odist ;
  int tmp, maxloc, ii, jj, kk ;
  double maxval, tmpval ;
  double *weights ;
  double *seminaive_naive_tablespace  ;
  double **seminaive_naive_table ;
  fftw_plan dctPlan, fftPlan ;
  int howmany_rank ;
  fftw_iodim dims[1], howmany_dims[1];

  bwIn = tams_bwIn;
  bwOut = tams_bwOut;
  degLim = tams_degLim;

  n = 2 * bwIn ;

  sigR = (double *) calloc( n * n, sizeof(double) );
  sigI = (double *) calloc( n * n, sizeof(double) );
  so3Sig = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * (8*bwOut*bwOut*bwOut) );
  workspace1 = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * (8*bwOut*bwOut*bwOut) );
  workspace2 = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * ((14*bwIn*bwIn) + (48 * bwIn)));
  workspace3 = (double *) malloc( sizeof(double) * (12*n + n*bwIn));
  sigCoefR = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  sigCoefI = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  patCoefR = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  patCoefI = (double *) malloc( sizeof(double) * bwIn * bwIn ) ;
  so3Coef = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * ((4*bwOut*bwOut*bwOut-bwOut)/3) ) ;


  seminaive_naive_tablespace =
    (double *) malloc(sizeof(double) *
		      (Reduced_Naive_TableSize(bwIn,bwIn) +
		       Reduced_SpharmonicTableSize(bwIn,bwIn)));

  weights = (double *) malloc(sizeof(double) * (4*bwIn));

  /****
       At this point, check to see if all the memory has been
       allocated. If it has not, there's no point in going further.
  ****/

  if ( (seminaive_naive_tablespace == NULL) || (weights == NULL) ||
       (sigR == NULL) || (sigI == NULL) ||
       (so3Coef == NULL) ||
       (workspace1 == NULL) || (workspace2 == NULL) ||
       (workspace3 == NULL) ||
       (sigCoefR == NULL) || (sigCoefI == NULL) ||
       (patCoefR == NULL) || (patCoefI == NULL) ||
       (so3Sig == NULL) )
    {
      perror("Error in allocating memory");
      exit( 1 ) ;
    }

  /* create fftw plans for the S^2 transforms */
  /* first for the dct */
  dctPlan = fftw_plan_r2r_1d( 2*bwIn, weights, workspace3,
			      FFTW_REDFT10, FFTW_ESTIMATE ) ;

  /* now for the fft */
  /* 
     IMPORTANT NOTE!!! READ THIS!!!

     Now to make the fft plans.

     Please note that the planning-rigor flag *must be* FFTW_ESTIMATE!
     Why? Well, to try to keep things simple. I am using some of the
     pointers to arrays in rotateFct's arguments in the fftw-planning
     routines. If the planning-rigor is *not* FFTW_ESTIMATE, then
     the arrays will be written over during the planning stage.

     Therefore, unless you are really really sure you know what
     you're doing, keep the rigor as FFTW_ESTIMATE !!!
  */

  /*
    fftw "preamble" ;
    note  that this places in the transposed array
  */

  rank = 1 ;
  dims[0].n = 2*bwIn ;
  dims[0].is = 1 ;
  dims[0].os = 2*bwIn ;
  howmany_rank = 1 ;
  howmany_dims[0].n = 2*bwIn ;
  howmany_dims[0].is = 2*bwIn ;
  howmany_dims[0].os = 1 ;

  fftPlan = fftw_plan_guru_split_dft( rank, dims,
				      howmany_rank, howmany_dims,
				      sigR, sigI,
				      (double *) workspace2,
				      (double *) workspace2 + (n*n),
				      FFTW_ESTIMATE );

  /* create plan for inverse SO(3) transform */
  n = 2 * bwOut ;
  howmany = n*n ;
  idist = n ;
  odist = n ;
  rank = 2 ;
  inembed[0] = n ;
  inembed[1] = n*n ;
  onembed[0] = n ;
  onembed[1] = n*n ;
  istride = 1 ;
  ostride = 1 ;
  na[0] = 1 ;
  na[1] = n ;

  p1 = fftw_plan_many_dft( rank, na, howmany,
			   workspace1, inembed,
			   istride, idist,
			   so3Sig, onembed,
			   ostride, odist,
			   FFTW_FORWARD, FFTW_ESTIMATE );


  fprintf(stdout,"Generating seminaive_naive tables...\n");
  seminaive_naive_table = SemiNaive_Naive_Pml_Table(bwIn, bwIn,
						    seminaive_naive_tablespace,
						    (double *) workspace2);

  /* make quadrature weights for the S^2 transform */
  makeweights( bwIn, weights ) ;

  n = 2 * bwIn ;
  printf("Reading in signal file\n");
  /* read in SIGNAL samples */
  /* first the signal */
  for ( i = 0 ; i < n * n ; i ++ )
    {
	/* first the real part of the sample */
	(*(sigR+i))=TAMS_sei_sig_real(i);
  	/* now the imaginary part */
	(*(sigI+i))=0;
    }

  printf("now taking spherical transform of signal\n");
  FST_semi_memo( sigR, sigI,
		 sigCoefR, sigCoefI,
		 bwIn, seminaive_naive_table,
		 (double *) workspace2, 0, bwIn,
		 &dctPlan, &fftPlan,
		 weights );

  printf("Reading in pattern file\n");
  /* read in SIGNAL samples */
  /* first the signal */
  for ( i = 0 ; i < n * n ; i ++ )
    {
	/* first the real part of the sample */
	(*(sigR+i))=TAMS_sei_pat_real(i);
	/* now the imaginary part */
	(*(sigI+i))=0;
    }

  printf("now taking spherical transform of pattern\n");
  FST_semi_memo( sigR, sigI,
		 patCoefR, patCoefI,
		 bwIn, seminaive_naive_table,
		 (double *) workspace2, 0, bwIn,
		 &dctPlan, &fftPlan,
		 weights ) ;

  printf("freeing seminaive_naive_table and seminaive_naive_tablespace\n");
  
  free( seminaive_naive_table ) ;
  free( seminaive_naive_tablespace ) ;


  printf("about to combine coefficients\n");

  /* combine coefficients */
  so3CombineCoef_fftw( bwIn, bwOut, degLim,
		       sigCoefR, sigCoefI,
		       patCoefR, patCoefI,
		       so3Coef ) ;
  
  printf("about to inverse so(3) transform\n");

  /* now inverse so(3) */
  Inverse_SO3_Naive_fftw( bwOut,
			  so3Coef,
			  so3Sig,
			  workspace1,
			  workspace2,
			  workspace3,
			  &p1,
			  0 ) ;
  printf("finished inverse so(3) transform\n");

  /* now find max value */
  maxval = 0.0 ;
  maxloc = 0 ;
  for ( i = 0 ; i < 8*bwOut*bwOut*bwOut ; i ++ )
    {
      /*
	if (so3Sig[i][0] >= maxval)
	{
	maxval = so3Sig[i][0];
	maxloc = i ;
	}
      */
      tmpval = NORM( so3Sig[i] );
      if ( tmpval > maxval )
	{
	  maxval = tmpval;
	  maxloc = i ;
	}
      
    }

  ii = floor( maxloc / (4.*bwOut*bwOut) );
  tmp = maxloc - (ii*4.*bwOut*bwOut);
  jj = floor( tmp / (2.*bwOut) );
  tmp = maxloc - (ii *4*bwOut*bwOut) - jj*(2*bwOut);
  kk = tmp ;

  alpha = M_PI*jj/((double) bwOut);
  beta  = M_PI*(2*ii+1)/(4.*bwOut);
  gamma = M_PI*kk/((double) bwOut);

  printf("finished correlation!\n");
  fftw_destroy_plan( p1 );
  fftw_destroy_plan( fftPlan );
  fftw_destroy_plan( dctPlan );

  free( weights );
  fftw_free( so3Coef ) ;
  free( patCoefI );
  free( patCoefR );
  free( sigCoefI );
  free( sigCoefR );
  free( workspace3 );
  fftw_free( workspace2 );
  fftw_free( workspace1 );
  fftw_free( so3Sig ) ;
  free( sigI );
  free( sigR );

  return;

} // end of function

