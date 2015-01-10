#ifndef _TAMS_SOFT_FFTW_CORRELATE_H
#define _TAMS_SOFT_FFTW_CORRELATE_H

extern void tams_soft_fftw_correlate (  const Eigen::VectorXf &TAMS_sei_sig_real,
                                        const Eigen::VectorXf &TAMS_sei_pat_real,
                                        const int tams_bwIn, const int tams_bwOut, const int tams_degLim,
                                        double &alpha,
                                        double &beta,
                                        double &gama);

#endif /*_TAMS_SOFT_FFTW_CORRELATE_H*/
