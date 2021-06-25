#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdlib.h>
#include "ap_fixed.h"

#define N1 392
#define M1 30

#define N2 30
#define M2 50

#define N3 50
#define M3 392

#define BITS 8		// set bitwidth of multipliers
#define BITS_EXP 1024 //must be set to 2^(BITS+2). Should match tanh_vals size


typedef ap_fixed<BITS+2,2,AP_RND> quantized_type;		// multipliers
typedef ap_fixed<BITS+9,9,AP_RND> l_quantized_type;	// intermediate results

#endif
