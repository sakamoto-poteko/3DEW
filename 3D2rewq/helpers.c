/***
 *  helper.c
 *
 *  Some helper functions
 *
 *  Afa.L Cheng <alpha@tomatoeskit.org>
 */

#include <omp.h>

#include "helpers.h"

//#include "discardme.h"        // This line is for Afa's IDE. Discard it

#ifdef _WITH_PHI
// If we have Xeon Phi
void zero_matrices(float *u, float *w, float *ws2, float *up2, float *vp1, float *wp1, float *us, float *ws, float *wp,
                   float *us2, float *us1, float *wp2, float *v, float *up1, int nz, int nx, float *up, int ny,
                   float *ws1, float *vs, float *vp2, float *vs1, float *vs2, float *vp)
{
    int t = nx * ny * nz;       // [Afa] Total elements number in an array
    // [Afa] That freaking big loop! Really bad for cache and SIMD. Decomposed it
    // [Afa] I don't have any idea on Xeon Phi. Is this correct?
    // [Afa] Might be a really BAD practice, since it takes a lot of bandwidth
    // But should be helpful since CPU needs more time to init the array

    // [Afa] offload documentation:
    // http://software.intel.com/sites/products/documentation/doclib/iss/2013/compiler/cpp-lin/GUID-EAB414FD-40C6-4054-B094-0BA70824E2A2.htm
#pragma offload target(mic) \
    out(u:length(t)) out(v:length(t)) out(w:length(t)) out(up:length(t)) out(up1:length(t)) out(up2:length(t)) \
    out(vp:length(t)) out(vp1:length(t)) out(vp2:length(t)) out(wp:length(t)) out(wp1:length(t)) out(wp2:length(t)) \
    out(us:length(t)) out(us1:length(t)) out(us2:length(t)) out(vs:length(t)) out(vs1:length(t)) out(vs2:length(t)) \
    out(ws:length(t)) out(ws1:length(t)) out(ws2:length(t))
    {
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) u   [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) v   [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) w   [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) up  [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) up1 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) up2 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) vp  [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) vp1 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) vp2 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) wp  [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) wp1 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) wp2 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) us  [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) us1 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) us2 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) vs  [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) vs1 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) vs2 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) ws  [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) ws1 [i] = 0.0f;
        #pragma omp parallel for
        for (int i = 0; i < t; ++i) ws2 [i] = 0.0f;
    }
}
#else
// If we don't have Xeon Phi
void zero_matrices(float *u, float *w, float *ws2, float *up2, float *vp1, float *wp1, float *us, float *ws, float *wp,
                   float *us2, float *us1, float *wp2, float *v, float *up1, int nz, int nx, float *up,
                   int ny, float *ws1, float *vs, float *vp2, float *vs1, float *vs2, float *vp)
{
    int t = nx * ny * nz;       // [Afa] Total elements number in an array
    // [Afa] That freaking big loop! Really bad for cache and SIMD. Decomposed it
    // [Afa] NB: I didn't put OpenMP parallel directive here since I don't know
    //           if we're using 1 core per process or 1 node per process
    for (int i = 0; i < t; ++i) u   [i] = 0.0f;
    for (int i = 0; i < t; ++i) v   [i] = 0.0f;
    for (int i = 0; i < t; ++i) w   [i] = 0.0f;
    for (int i = 0; i < t; ++i) up  [i] = 0.0f;
    for (int i = 0; i < t; ++i) up1 [i] = 0.0f;
    for (int i = 0; i < t; ++i) up2 [i] = 0.0f;
    for (int i = 0; i < t; ++i) vp  [i] = 0.0f;
    for (int i = 0; i < t; ++i) vp1 [i] = 0.0f;
    for (int i = 0; i < t; ++i) vp2 [i] = 0.0f;
    for (int i = 0; i < t; ++i) wp  [i] = 0.0f;
    for (int i = 0; i < t; ++i) wp1 [i] = 0.0f;
    for (int i = 0; i < t; ++i) wp2 [i] = 0.0f;
    for (int i = 0; i < t; ++i) us  [i] = 0.0f;
    for (int i = 0; i < t; ++i) us1 [i] = 0.0f;
    for (int i = 0; i < t; ++i) us2 [i] = 0.0f;
    for (int i = 0; i < t; ++i) vs  [i] = 0.0f;
    for (int i = 0; i < t; ++i) vs1 [i] = 0.0f;
    for (int i = 0; i < t; ++i) vs2 [i] = 0.0f;
    for (int i = 0; i < t; ++i) ws  [i] = 0.0f;
    for (int i = 0; i < t; ++i) ws1 [i] = 0.0f;
    for (int i = 0; i < t; ++i) ws2 [i] = 0.0f;
}
#endif
