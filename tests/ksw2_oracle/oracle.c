// Runs minimap2 2.31's ksw_extd2_sse on each case line and prints every field of the result.
// Case: flag w zdrop end_bonus q e q2 e2 a b ambi target query   (sequences over ACGTN)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ksw2.h"

static int code(char c)
{
	switch (c) {
	case 'A': return 0;
	case 'C': return 1;
	case 'G': return 2;
	case 'T': return 3;
	default: return 4;
	}
}

int main(void)
{
	static char t[100000], q[100000];
	int flag, w, zdrop, eb, gq, ge, gq2, ge2, a, b, ambi;
	while (scanf("%d %d %d %d %d %d %d %d %d %d %d %s %s", &flag, &w, &zdrop, &eb, &gq, &ge, &gq2, &ge2, &a, &b, &ambi, t, q) == 13) {
		int tl = strlen(t), ql = strlen(q), i, j;
		uint8_t *ts = malloc(tl), *qs = malloc(ql);
		for (i = 0; i < tl; ++i) ts[i] = code(t[i]);
		for (i = 0; i < ql; ++i) qs[i] = code(q[i]);
		int8_t mat[25];
		for (i = 0; i < 4; ++i) {
			for (j = 0; j < 4; ++j) mat[i * 5 + j] = i == j ? a : -b;
			mat[i * 5 + 4] = -ambi;
		}
		for (j = 0; j < 5; ++j) mat[20 + j] = -ambi;
		ksw_extz_t ez;
		memset(&ez, 0, sizeof(ez));
		ksw_extd2_sse(0, ql, qs, tl, ts, 5, mat, gq, ge, gq2, ge2, w, zdrop, eb, flag, &ez);
		printf("%d %u %d %d %d %d %d %d %d %d %d ", ez.score, ez.max, ez.max_q, ez.max_t, ez.mqe, ez.mqe_t, ez.mte, ez.mte_q, ez.zdropped, ez.reach_end, ez.n_cigar);
		for (i = 0; i < ez.n_cigar; ++i) printf("%d%c", ez.cigar[i] >> 4, "MIDN"[ez.cigar[i] & 0xf]);
		printf("\n");
		free(ez.cigar);
		free(ts);
		free(qs);
	}
	return 0;
}
