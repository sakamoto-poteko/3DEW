/***
 *  helpers.h
 *
 *  Header for those helper functions
 *
 *  Afa.L Cheng <alpha@tomatoeskit.org>
 */


#ifndef HELPERS_H
#define HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

void zero_matrices(float *u, float *w, float *ws2, float *up2, float *vp1, float *wp1, float *us, float *ws, float *wp,
                   float *us2, float *us1, float *wp2, float *v, float *up1, int nz, int nx, float *up,
                   int ny, float *ws1, float *vs, float *vp2, float *vs1, float *vs2, float *vp);

#ifdef __cplusplus
}
#endif

#endif // HELPERS_H
